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
import gzip
import pickle
import matplotlib.pyplot as plt
import os
import shlex
import subprocess
from Bio import AlignIO
from Bio import SeqIO

import phlame.helper_functions as helper




#%% Fxns I want
    
class Tree():
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

    def __init__(self, input_cmt_file, 
                 output_phylip,
                 output_renaming_file,
                 output_tree=False,
                 refGenome_file=False, 
                 min_cov_to_include=10, 
                 min_maf_for_call=0.9,
                 min_strand_cov_for_call=3, 
                 max_qual_for_call=-30,
                 min_presence_core=0.9,
                 min_median_cov_samples=3,
                 max_frac_ambiguous_pos=0.05,
                 max_mean_copynum=2.5,
                 remov_recomb=False):
        

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
        self.output_phylip = output_phylip
        self.output_renaming_file = output_renaming_file
        self.output_tree = output_tree
        self.refGenome_file = refGenome_file
        self.remov_recomb = remov_recomb

    def main(self):

        # =========================================================================
        #  First check if just tree building needed
        # =========================================================================

        if helper.Phylip.check_valid(self.output_phylip):
            print(f"Valid phylip file found at: {self.output_phylip}.")
            print(f"Building tree with existing file...")

            self.raxml()

            return
        
        # =========================================================================
        #  Read in input files
        # =========================================================================
        self.CMT = helper.CMT()
        
        self.CMT.read_cmt(self.input_cmt_file)

        # =========================================================================
        # Filtering
        # =========================================================================

        self.filter_coverage()

        self.filter_basecalls()

        self.filter_positions()

        self.filter_samples()

        # =========================================================================
        # Write phylip file
        # =========================================================================

        self.write_phylip()


    def filter_coverage(self):
        '''
        Remove low-coverage and outgroup samples.
        '''

        # fig = plot_coverage_hist(coverage_all, float(filterby_sample['min_cov_to_include']))
        # fig.show()

        if np.any(self.CMT.in_outgroup):
            print(f"The following samples were excluded as outgroups: {self.CMT.sample_names[self.in_outgroup]}")

        print("Filtering samples by coverage...")

        # Booleans are all INCLUSION criteria!
        good_cov_bool = ( np.median(self.CMT.coverage,axis=0) >= \
                         float(self.filterby_sample['min_cov_to_include']) )

        print(f"{np.sum(good_cov_bool)}/{len(self.CMT.sample_names)} samples passed median coverage filter.")
        print("The following samples did NOT pass median coverage filter:")
        print( self.CMT.sample_names[np.median(self.CMT.coverage,axis=0) \
                                     < float(self.filterby_sample['min_cov_to_include'])] )
        
        include_bool = good_cov_bool & ~self.CMT.in_outgroup
        
        
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
        sample_names_4phylip = np.char.add(np.arange(0,len(self.good_sample_names)).astype(str), \
                                            self.good_sample_names.astype(str)).astype(object)
        
        #.dnapars.fa > for dnapars...deleted later
        self.write_calls_to_fasta(calls_for_tree,
                             sample_names_4phylip,
                             self.output_phylip+".dnapars.tmp") 
        
        # turn fa to phylip and delete fasta with short tip labels
        aln = AlignIO.read(self.output_phylip+".dnapars.tmp", 'fasta')
        
        AlignIO.write(aln, self.output_phylip, "phylip")
        
        subprocess.run(["rm -f " + self.output_phylip+".dnapars.tmp"],shell=True)

        # Write object to convert phylip names back at the end
        with open(self.output_renaming_file,'w') as f:
            for line in range(len(self.good_sample_names)):
                f.write(f"{sample_names_4phylip[line][:10]}\t{self.good_sample_names[line]}\n")

    def raxml(self):
        
        print("Running RAxML...")

        # Paths for tree
        working_dir = os.path.dirname(self.output_tree)
        basename = os.path.basename(self.output_tree)
        
        # Run RAxML
        print("Running RAxML as follows: " + 
              "raxmlHPC -s " + 
                        shlex.quote(self.output_phylip) + 
                        " -N 1" + 
                        " -w " + shlex.quote(working_dir) +
                        " -n " + basename + 
                        " -m GTRCAT -p 12345 ")
        
        subprocess.run("raxmlHPC -s " + 
                        shlex.quote(self.output_phylip) + 
                        " -N 1" + 
                        " -w " + shlex.quote(working_dir) +
                        " -n " + basename + 
                        " -m GTRCAT -p 12345 ", shell=True)
        


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