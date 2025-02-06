#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:57:14 2022

@author: evanqu
"""

#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import shlex
import shutil
import subprocess
from Bio import AlignIO

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import ete3

import phlame.helper_functions as helper


#%% Fxns I want

class Tree():

    def __init__(self, tree_file):

        self.tree = ete3.Tree(tree_file, format=0)
        # midpoint_root = self.tree.get_midpoint_outgroup()
        # self.tree.set_outgroup(midpoint_root)
        self.tree.standardize()

        self.tree_samples = self.tree.get_leaf_names()
        
    def readinCMT(self, path_to_cmt):
        
        CMT = helper.CandidateMutationTable(path_to_cmt)
                
        sample_names = self.rphylip(CMT.sample_names)

        match_bool = np.in1d(sample_names, self.tree_samples)
        if np.count_nonzero(match_bool) != len(self.tree_samples):
            raise Exception('At least one sample from tree not found in candidate mutation table!')

        self.CMT_samples = CMT.sample_names[match_bool]
        self.CMT_counts = CMT.counts[:,:,match_bool]

    def rescale(self, path_to_cmt):
        '''
        Rescale a phylogeny from tree distance to # of SNPs using regression.
        '''

        self.readinCMT(path_to_cmt)
        
        tree_dm = self.tip_tip_distmat()

        maNT, _, _, _ = helper.mant(self.CMT_counts)

        snp_dm = helper.distmat(maNT,
                                self.CMT_samples)
        
        # Flatten distmat into 1D array
        snp_dists = self.flatten_distmat(snp_dm)
        tree_dists = self.flatten_distmat(tree_dm)
        
        tree_dists = tree_dists[snp_dists>0] # Remove SNP distances of 0
        snp_dists = snp_dists[snp_dists>0] # Remove SNP distances of 0

        ### Linear regression ###
        
        linreg = stats.linregress(tree_dists, snp_dists)
        
        if linreg.rvalue < 0.75:
            print(f"There is a low correlation between branch length and mutational distances: {linreg.rvalue:.2f}." + \
                  "\nWe do not recommend using the rescaled tree.")
        
        ## Plot linear regression
        fig = self.plot_regression(tree_dists, snp_dists, linreg)

        ### Go through tree and scale all branch lengths ###
        newtree = self.tree.copy()
        
        for node in newtree.traverse(strategy='preorder'):
            if node.dist is not None:
                node.dist = (node.dist*linreg.slope)
                
        return newtree, fig
    
    def flatten_distmat(self, dm):
        '''
        Flatten a distance matrix and remove diagonal values.
        '''

        dm_sorted = self.sort_distmat(dm)
        dists = dm_sorted.to_numpy().flatten()
        dists = np.delete(dists, # Remove values on the diagonal
                              range(0, len(dists), len(dm_sorted) + 1), 0)
        

        return dists

    def tip_tip_distmat(self):
        '''
        Calculate the tip-to-tip distance of every tip on tree.
        '''
        dm = np.zeros((len(self.tree),len(self.tree)))
        
        names = []  
        for idx1, leaf1 in enumerate(self.tree.get_leaves()):
            
            names.append(leaf1.name)
            
            for idx2, leaf2 in enumerate(self.tree.get_leaves()): 
                
                dm[idx1, idx2] = self.tree.get_distance(leaf1, leaf2)
        
        dm_df = pd.DataFrame(dm, index=names, columns=names)
        
        return dm_df

    @staticmethod
    def plot_regression(xs, ys, linreg):
        
        fmt={'fontsize':15,
            'fontname':'Helvetica'}

        fig, axs = plt.subplots()
        
        fig.set_size_inches(5,5)

        axs.scatter(xs, ys, c='k', marker='o', alpha=0.1)
        axs.plot([0,max(xs)], 
                 [linreg.intercept,(max(xs)*linreg.slope)+linreg.intercept], color='r')
        
        axs.set_xlabel('Tree distances',**fmt)
        axs.set_ylabel('# core genome mutations', **fmt)
        axs.tick_params(axis='both', labelsize=12)

        axs.text(0.05, 0.85, f"$r^2$={float(linreg.rvalue):.2f}\np={float(linreg.pvalue):.3e}",
                 transform=axs.transAxes, fontsize=12)
        
        fig.tight_layout()

        return fig
        
    @staticmethod
    def rphylip(sample_names):
        '''Change : to | for consistency with phylip format'''
        
        rename = [sam.replace(':','|') for sam in sample_names]
        
        return np.array(rename)    
    
    @staticmethod
    def sort_distmat(dm):
        
        dm_sorted = dm.sort_index()
        dm_sorted = dm_sorted.reindex(sorted(dm_sorted.columns), axis=1)
        
        return dm_sorted 


class CMT2tree():
    '''
    Main controller of the tree construction step.

    input_cmt_file (str): Path to input candidate mutation table.
    output_phylip (str): Path to output phylip file.
    output_renaming_file (str): Path to output file to rename phylip names to original.
    min_cov_to_include (float, optional): Minimum avg. coverage across positions to include a sample. Defaults to 8.
    min_maf_for_call (float, optional): Minimum major allele frequency to call a major allele for position. Defaults to 0.85.
    min_strand_cov_for_call (int, optional): Minimum coverage per strand to call a major allele for position. Defaults to 2.
    min_qual_for_call (int, optional): Minimum mapping quality to call a major allele for position. Defaults to -30.
    min_presence_core (float, optional): Minimum presence across samples to include a position. Defaults to 0.9.
    min_median_cov_samples (int, optional): Minimum median coverage across samples to include a position. Defaults to 3.
    consider_indels (bool, optional): Consider number of indels when filtering a position. Defaults to False.

    '''

    def __init__(self, 
                 input_cmt_file=None,
                 input_phylip=None,
                 output_phylip=None,
                 output_renaming_file=None,
                 outgroup_str=None,
                 output_tree=None,
                 refGenome_file=None, 
                 rescale_bool=False,
                 min_cov_to_include=10, 
                 min_maf_for_call=0.9,
                 min_strand_cov_for_call=3, 
                 max_qual_for_call=-30,
                 min_presence_core=0.9,
                 min_median_cov_samples=3,
                 max_frac_ambiguous_pos=0.05,
                 max_mean_copynum=2.5,
                 remov_recomb=None):
        

        self.filterby_sample = {\
                                'min_cov_to_include':float(min_cov_to_include),
                                'max_frac_ambiguous_pos':float(max_frac_ambiguous_pos)
                                }
    
        self.filterby_site_per_sample = {\
                                        'min_maf_for_call':float(min_maf_for_call),
                                        'min_strand_cov_for_call':float(min_strand_cov_for_call),
                                        'max_qual_for_call': float(max_qual_for_call),
                                        'max_frac_reads_supporting_indel':0.33
                                        }
        
        self.filterby_site_across_samples = {\
                                            'min_presence_core':float(min_presence_core),
                                            'min_median_cov_samples':float(min_median_cov_samples),
                                            'max_mean_copynum':float(max_mean_copynum)
                                            }
                    
        self.filter_recombination = {\
                                    'distance_for_nonsnp' : 300, #region in bp on either side of goodpos that is considered for recombination
                                    'corr_threshold_recombination' : 0.75 #minimum threshold for correlation
                                    }
        
        self.input_cmt_file = input_cmt_file
        self.input_phylip = input_phylip
        self.output_phylip = output_phylip
        self.output_renaming_file = output_renaming_file
        self.output_tree = output_tree
        self.refGenome_file = refGenome_file
        self.rescale_bool = rescale_bool
        self.remov_recomb = remov_recomb
        self.outgroup_str = outgroup_str
            

        if self.remov_recomb:
            print('Warning: Recombination filtering is not currently implemented. Continuing without filtering.')


        # Require either input or output phylip file
        if not self.input_phylip and not self.output_phylip:
            raise Exception("Path to a phylip file must be specified as either an input or output.")
        
        if self.input_phylip and self.output_phylip:
            raise Exception("Cannot specify both input and output phylip file.")
        
        # For raxml
        if self.input_phylip:
            self.phylip = self.input_phylip
        elif self.output_phylip:
            self.phylip = self.output_phylip
        
        # If output phylip file, also require output renaming file and input CMT file
        if self.output_phylip and not (self.output_renaming_file and self.input_cmt_file):
            if not self.output_renaming_file:
                raise Exception("Path to a renaming file must be specified when output phylip file is specified.")
            if not self.input_cmt_file:
                raise Exception("Path to a candidate mutation table must be specified when output phylip file is specified.")
            
        # If input phylip file, require output tree file
        if self.input_phylip and not self.output_tree:
            raise Exception("Path to an output tree file must be specified when input phylip file is specified.")
            
    def main(self):

        # =========================================================================
        #  First check if just tree building needed
        # =========================================================================

        if self.input_phylip and helper.Phylip.check_valid(self.input_phylip):
            print(f"Valid phylip file found at: {self.input_phylip}.")
            print(f"Building tree with existing file...")

            self.check_raxml()
            
            self.make_tree()

            return
        
        # =========================================================================
        #  Read in input files
        # =========================================================================
        self.CMT = helper.CandidateMutationTable(self.input_cmt_file)
        
        # =========================================================================
        # Filtering
        # =========================================================================

        self.parse_outgroup()

        self.filter_coverage()

        self.filter_basecalls()

        self.filter_positions()

        self.filter_samples()

        # =========================================================================
        # Write phylip file
        # =========================================================================

        if self.output_phylip:
            
            self.write_phylip()

        # =========================================================================
        # If output_tree is True, build tree
        # =========================================================================

        if self.output_tree:

            self.check_raxml()

            self.make_tree()
    
    def check_raxml(self):
        '''
        Check if raxml is installed.
        '''
        # find the location of the program
        loc = shutil.which('raxmlHPC')

        # make sure the help on the program works
        if loc == None:
            raise Exception("Executable raxmlHPC not found. Install RaXML first before creating a tree, or just create a phylip file.")

        works = False
        if loc != None:
            try:
                o = subprocess.check_output([loc, '-h'],stderr=subprocess.STDOUT)
                works = True
            except:
                pass

        if not works:
            raise Exception("Executable raxmlHPC found, but not working. Make sure RaXML is installed correctly, or just create a phylip file.")

    def make_tree(self):
        '''
        All steps to build and process tree including raxml, renaming, rescaling.
        '''

        self.raxml()

        # print("Tree built. Renaming phylip names...")
        self.rename_phylip()

        
        if self.rescale_bool:
            print("Rescaling branch lengths into # of mutations...")

            tree_obj = Tree(self.output_tree)

            tree_scaled, fig = tree_obj.rescale(self.input_cmt_file)

            #prepend 'rescaled_' to basename of output_tree
            rescaled_tree_path = os.path.join(os.path.dirname(self.output_tree),
                                              'rescaled_'+os.path.basename(self.output_tree))
            
            tree_scaled.write(outfile=rescaled_tree_path, format=0)

            # check that the tree is valid
            assert os.path.exists(rescaled_tree_path), 'Rescaled tree file not created.'
            
            fig.savefig(os.path.join(os.path.dirname(self.output_tree),
                                      'tree_snp_distances_linreg.pdf'), format='pdf')


    def parse_outgroup(self):

        if self.outgroup_str:

            self.outgroup_ls = [item.strip() for item in self.outgroup_str.split(",")]
            
            # Check that all outgroup genomes are present in candidate mutation table
            if not np.array([genome in self.CMT.sample_names for genome in self.outgroup_ls]).all():
                raise Exception('The following outgroup genomes are not present in the candidate mutation table: ' +
                                ', '.join([genome for genome in self.outgroup_ls if genome not in self.CMT.sample_names]))

            self.outgroup_bool = np.isin(self.CMT.sample_names, self.outgroup_ls)

        else:
            self.outgroup_bool = np.array([True]*len(self.CMT.sample_names))


    def filter_coverage(self):
        '''
        Remove low-coverage and outgroup samples.
        '''

        # fig = plot_coverage_hist(coverage_all, float(filterby_sample['min_cov_to_include']))
        # fig.show()

        if np.sum(self.outgroup_bool) != len(self.CMT.sample_names):
            print("The following samples were removed as outgroup samples:")
            print( self.CMT.sample_names[~self.outgroup_bool] )

        print("Filtering samples by coverage...")

        # Booleans are all INCLUSION criteria!
        good_cov_bool = ( np.median(self.CMT.coverage,axis=0) >= \
                         float(self.filterby_sample['min_cov_to_include']) )

        print(f"{np.sum(good_cov_bool)}/{len(self.CMT.sample_names)} samples passed median coverage filter.")
        print("The following samples did NOT pass median coverage filter:")
        print( self.CMT.sample_names[np.median(self.CMT.coverage,axis=0) \
                                     < float(self.filterby_sample['min_cov_to_include'])] )
        
        include_bool = good_cov_bool & self.outgroup_bool
        
        # Rationale is not removing low-cov samples first will mess with other filtering
        self.sample_names = self.CMT.sample_names[include_bool]
        self.counts = self.CMT.counts[:,:,include_bool]
        self.quals = self.CMT.quals[:,include_bool]
        self.indels = self.CMT.indels_all[:,include_bool]
        self.pos = self.CMT.pos
        
        ### Create structures from new sample list ###
        self.num_samples=len(self.sample_names)
        
        self.coverage = self.counts.sum(axis=0)
        self.coverage_f_strand = self.counts[0:4,:,:].sum(axis=0)
        self.coverage_r_strand = self.counts[4:8,:,:].sum(axis=0)

        if len(self.sample_names) < 3:
            raise Exception("Too few samples passed the coverage filter!")
        

    def filter_basecalls(self):
        
        print("Filtering basecalls...")
        
        # Order is important!
            # 1. Per position, per sample
            # 2. Per position across samples
            # 3. Per sample
            # 4. Recomb ? ( before sample filter ?)

        # Get major allele at each position
            # 01234=NATCG
        [maNT, maf, _, _] = helper.mant(self.counts)

        calls = np.copy(maNT)
        calls[ self.quals > self.filterby_site_per_sample['max_qual_for_call'] ] = 0
        # remember quals are negative!
        calls[ maf < self.filterby_site_per_sample['min_maf_for_call'] ] = 0
        calls[ self.coverage_r_strand < self.filterby_site_per_sample['min_strand_cov_for_call'] ] = 0
        calls[ self.coverage_r_strand < self.filterby_site_per_sample['min_strand_cov_for_call'] ] = 0

        # Filter positions with indels
        with np.errstate(divide='ignore',invalid='ignore'):
            frac_reads_w_indel = self.indels/self.coverage # sum reads supporting insertion plus reads supporting deletion
            frac_reads_w_indel[ ~np.isfinite(frac_reads_w_indel) ] = 0
        
        calls[ frac_reads_w_indel > self.filterby_site_per_sample['max_frac_reads_supporting_indel'] ] = 0

        self.calls = calls

    def filter_positions(self):

        print("Filtering positions across samples...")
    
        # fig = plot_positions_across_samples_hist(np.count_nonzero(calls, axis=1), 
        #                                         num_samples*float(filterby_site_across_samples['min_presence_core']),
        #                                         'Presence (not N) across samples')
        # fig.show()

        # Booleans are all INCLUSION criteria!
        # Must be present (not N) in some fraction of samples
        min_core_bool = ( np.count_nonzero(self.calls, axis=1) >= \
                            (self.num_samples*self.filterby_site_across_samples['min_presence_core']) ) 

        # Must have some minimum median coverage across samples
        min_medcov_bool = ( np.median(self.coverage, axis=1) >= \
                            self.filterby_site_across_samples['min_median_cov_samples'] )

        # Must not exceed max average copy number per samples
        max_copynum_bool = ( np.mean(self.coverage / np.median(self.coverage, axis=0),axis=1) <= \
                                self.filterby_site_across_samples['max_mean_copynum'] )
        
        self.pos_filter_bool = np.all( (min_core_bool,min_medcov_bool,max_copynum_bool ), axis=0)

        self.goodpos = self.pos[self.pos_filter_bool]
        
        n_goodpos = np.count_nonzero(self.pos_filter_bool)

        print(f"{n_goodpos}/{len(self.pos)} positions ({n_goodpos*100/len(self.pos):.1f}%) passed filtering.")

        if n_goodpos < 10:
            raise Exception("Too few positions retained after filtering! Exiting...")
        
    def filter_samples(self):
        
        print('Filtering samples....')
        
        # fig = plot_samples_hist(1-(np.count_nonzero(goodcalls, axis=0)/len(goodcalls)),
        #                         float(filterby_sample['max_frac_ambiguous_pos']))
        # fig.show()

        self.max_fracNs_bool = ( 1-(np.count_nonzero(self.calls[self.pos_filter_bool], axis=0)/\
                                    len(self.calls[self.pos_filter_bool])) <= \
                                self.filterby_sample['max_frac_ambiguous_pos'] )
            
        n_goodsamples = np.count_nonzero(self.max_fracNs_bool)
        
        print(f"{n_goodsamples}/{len(self.sample_names)} samples passed breadth filtering.")
        print("The following samples did not pass breadth filtering:")
        print( self.sample_names[~self.max_fracNs_bool] )
    
    def write_phylip(self):
        
        print("Writing phylip file...")
        
        # numpy broadcasting of row_array requires np.ix_()
        calls_for_treei = self.calls
        # Convert -10123 to NATCG translation
        calls_for_tree = self.idx2nts(calls_for_treei) 
        
        self.good_sample_names = self.sample_names[self.max_fracNs_bool]

        # Add 0,1,2,3 to beginning of sample names for phylip format
        sample_names_4phylip = self.sample_names_phylip(self.good_sample_names)
        
        #.dnapars.fa > for dnapars...deleted later
        self.write_calls_to_fasta(calls_for_tree,
                             sample_names_4phylip,
                             self.output_phylip+".dnapars.tmp") 
        
        # turn fa to phylip and delete fasta with short tip labels
        aln = AlignIO.read(self.output_phylip+".dnapars.tmp", 'fasta')
        
        AlignIO.write(aln, self.output_phylip, "phylip")
        
        subprocess.run(["rm -f " + self.output_phylip+".dnapars.tmp"],shell=True)

        # Write object to convert phylip names back at the end
        self.phylip2names = dict(zip(sample_names_4phylip, self.good_sample_names))
        
        with open(self.output_renaming_file,'w') as f:
            for key, value in self.phylip2names.items():
                f.write(f"{key}\t{value}\n")

    def raxml(self):
        
        print("Running RAxML...")

        working_dir = os.path.dirname(self.output_tree)
        basename = os.path.basename(self.output_tree)

        print(shlex.quote(working_dir))
        print(shlex.quote(basename))

        # Run RAxML
        print("Running RAxML as follows: " + 
              "raxmlHPC -s " + 
                        shlex.quote(self.phylip) + 
                        " -N 1" + 
                        " -w " + shlex.quote(working_dir) +
                        " -n " + basename + 
                        " -m GTRCAT -p 12345 ")
        
        subprocess.run("raxmlHPC -s " + 
                        shlex.quote(self.phylip) + 
                        " -N 1" + 
                        " -w " + shlex.quote(working_dir) +
                        " -n " + basename + 
                        " -m GTRCAT -p 12345 ", shell=True)


    def rename_phylip(self):
        '''
        Convert 10chr phylip names in a nwk file into long format.
        And get away from raxml naming system!
        '''

        # Read in RAxML tree
        basename = os.path.basename(self.output_tree)
        raxml_outpath = self.output_tree.replace(basename, 
                                                 'RAxML_bestTree.'+basename)
        
        # Replace phylip tree names
        with open(raxml_outpath) as f:
            tre=f.read()


        if not self.output_renaming_file:
            print("No renaming file specified. Continuing without renaming samples...")

            with open(self.output_tree,'w') as f:
                f.write(tre)

            return
        
        if os.path.exists(self.output_renaming_file):
            self.phylip2names = dict()
            with open(self.output_renaming_file) as f:
                for line in f:
                    key, value = line.strip().split('\t')
                    self.phylip2names[key] = value

        else:
            raise Exception("Renaming file not found. Exiting...")
        
        # Replace with representative isolate name
        for i in self.phylip2names.keys():
            tre=tre.replace(i,self.phylip2names[i])
        
        # Write out new tree
        with open(self.output_tree,'w') as f:
            f.write(tre)

    @staticmethod
    def sample_names_phylip(sample_names):
        '''
        Add 0,1,2,3,4 to beginning of sample names for phylip format.
        '''

        nums = np.arange(0,len(sample_names)).astype(str)
        sample_names_4phylip = np.char.add(nums, sample_names.astype(str))
        
        return sample_names_4phylip.astype(object)
    
    @staticmethod
    def idx2nts(calls, missingdata="?"):
        # translate index array to array containing nucleotides
        # add 5th element --> no data! == index -1
        nucl = np.array([missingdata,'A','T','C','G'],dtype=object) 
        palette = [0,1,2,3,4] # values present in index-array
        index = np.digitize(calls.ravel(), palette, right=True)
        
        return nucl[index].reshape(calls.shape)
    
    @staticmethod
    def write_calls_to_fasta(calls, sample_names, output_file):
    
        fa_file = open(output_file, "w")
        
        for i,name in enumerate(sample_names):
            nucl_string = "".join(list(calls[:,i]))
            fa_file.write(">" + name + "\n" + nucl_string + "\n")
        
        fa_file.close()


# # =============================================================================
# #  5. Filter recombinant positions
# # =============================================================================

# #  Check for recombination in p and remove positions from goodpos
# if remov_recomb:
#     print("Filtering recombinant positions...")
#     # When no outgroup defined: refnt ~= ancnt:
#     [chrStarts, genomeLength, scafNames] = genomestats(refGenome_file)

#     refnt = extract_outgroup_mutation_positions(refGenome_file, p2chrpos(goodpos,chrStarts));
#     ancnt = refnt
#     ancnti_m = np.full(ancnt.shape, 9)
    
#     # Change ATCGatcg to numeric 1234 (0 if no allele)
#     for idx, allele in enumerate(ancnt):
        
#         if allele in NTs:
            
#             ancnti_m[idx,] = np.where(NTs==allele)[0][0]+1 # strip down to index number
        
#         else:
#             ancnti_m[idx,] = 0

#     recombpos = findrecombinantSNPs(pos, 
#                                     goodpos, 
#                                     good_counts, 
#                                     ancnti_m, num_samples, 
#                                     filter_recombination['distance_for_nonsnp'],
#                                     filter_recombination['corr_threshold_recombination'])

#     #These are the positions in p that are likely recombinant that we will remove from goodpos
#     print(str(sum(np.isin(goodpos, recombpos))) + ' of a total ' + \
#             str(goodpos.shape[0]) + ' ('  + str(sum(np.isin(goodpos, recombpos))/goodpos.shape[0]*100) + \
#                 '%) positions in goodpos were found to be recombinant.')
    
#     goodcalls = goodcalls[~np.isin(goodpos, recombpos),:]
#     # goodpos = goodpos[~np.isin(goodpos, recombpos)]