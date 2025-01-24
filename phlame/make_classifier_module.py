#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to build a PHLAME classifier.
@author: evanqu
"""

#%%
import numpy as np
import pandas as pd 
# import h5py
import math
import pickle
import gzip
import warnings
import ete3
import itertools    
from scipy import stats

import phlame.helper_functions as helper

#%% FXNs

class MakeDB():
    '''
    Main controller of the makedb step.
    '''

    def __init__(self, 
                 path_to_cmt,
                 path_to_candidate_clades,
                 path_to_output_db,
                 path_to_candidate_clades_tree=False,
                 min_cssnps=10,
                 maxn=0.1,
                 core=0.9,
                 min_maf_for_call=0.75,
                 min_strand_cov_for_call=2,
                 max_qual_for_call=-30,
                 min_presence_core=0.5,
                 max_outgroup=False
                 ):

        self.__path_to_cmt = path_to_cmt
        self.__path_to_candidate_clades = path_to_candidate_clades
        self.__path_to_output_db = path_to_output_db
        if path_to_candidate_clades_tree:
            self.__path_to_candidate_clades_tree = path_to_candidate_clades_tree
        else:
            self.__path_to_candidate_clades_tree = False

        self.min_cssnps = min_cssnps
        self.maxn = maxn
        self.core = core

        self.max_outgroup = max_outgroup

        self.min_maf_for_call = min_maf_for_call
        self.min_strand_cov_for_call = min_strand_cov_for_call
        self.max_qual_for_call = max_qual_for_call
        self.min_presence_core = min_presence_core
            
    def main(self):
        
        # =========================================================================
        # Read in files
        # =========================================================================
        print('Reading in files...')
        sample_names, pos, counts, quals, _, in_outgroup = helper.read_cmt(self.__path_to_cmt)
        sample_names = helper.rphylip(sample_names)
        maNT, maf, _, _ = helper.mant(counts)

        coverage_f_strand=counts[0:4,:,:].sum(axis=0) #should be pxs
        coverage_r_strand=counts[4:8,:,:].sum(axis=0)

        candidate_clades, candidate_clade_names = helper.read_clades_file(self.__path_to_candidate_clades,
                                                                   uncl_marker='-1')

        if self.__path_to_candidate_clades_tree:
            candidate_clades_tree = ete3.Tree(self.__path_to_candidate_clades_tree, format=1)
            #1 includes node names
        else:
            candidate_clades_tree = False

        # =========================================================================
        # Define ingroup and outgroup
        # =========================================================================

        if self.max_outgroup:
            ingroup = maNT[:,~in_outgroup]
            ingroup_maf = maf[:,~in_outgroup]
            ingroup_sample_names = sample_names[~in_outgroup]
            outgroup = maNT[:,in_outgroup]
            outgroup_maf = maf[:,in_outgroup]
            outgroup_sample_names = sample_names[in_outgroup]
        
        else:
            ingroup = maNT; ingroup_sample_names = sample_names; ingroup_maf = maf
            outgroup = np.array([]); outgroup_sample_names = np.array([]); outgroup_maf = np.array([])

        # =========================================================================
        # Do pre-filtering of allele calls
        # =========================================================================

        # Filter for only core genome positions
        is_core_genome = np.count_nonzero(ingroup, axis=1)/len(ingroup[1]) >= self.core
        core_maNT = ingroup[is_core_genome]
        core_pos = pos[is_core_genome]
        print(f"Number of core positions: {len(core_pos)}/{len(pos)}")

        # Mask ambiguous allele calls
        calls = np.copy(core_maNT)
        # calls[ quals[is_core_genome] > self.max_qual_for_call ] = 0
        # calls[ ingroup_maf[is_core_genome] < self.min_maf_for_call ] = 0

        # Mask samples with too many ambiguous allele calls
        mask_fracNs = ( ((calls>0).sum(axis=0)/len(calls)) >= self.min_presence_core )
        print('The following samples have too many ambiguous allele calls and will not be considered:')
        print('\n'.join(ingroup_sample_names[~mask_fracNs]))

        # Moving this to inside the unaminous_to_clade function
        # calls = calls[:,mask_fracNs]
        # ingroup_sample_names = ingroup_sample_names[mask_fracNs]

        # =========================================================================
        # Get csSNPs for every clade
        # =========================================================================
                    
        #Call csSNPs
        print('Getting unanimous alleles...')
        unanimous_alleles = unanimous_to_clade(calls, ingroup_sample_names,
                                               candidate_clades, candidate_clade_names,
                                               self.maxn, self.min_presence_core)

        print('Getting unique alleles...')
        candidate_css = unique_to_clade(calls, unanimous_alleles, ingroup_sample_names,
                                        candidate_clades, candidate_clade_names)
        
        # Remove positions aligning to too many outgroup genomes
        if len(outgroup_sample_names) > 0:
            print('Removing positions present in outgroup genomes...')
            max_outgroup_bool = ( (np.count_nonzero(outgroup, axis=1)/len(outgroup[1])) >= self.max_outgroup )

            candidate_css[max_outgroup_bool[is_core_genome]] = 0
            print(f"Removed {np.count_nonzero(max_outgroup_bool[is_core_genome])}/{len(candidate_css)} positions present >{self.max_outgroup*100}% of outgroup genomes")

        # =========================================================================
        #  Remove clades without enough csSNPs
        # =========================================================================
        
        is_cs_clade = np.count_nonzero(candidate_css,0) > self.min_cssnps
        
        cssnps_arr = candidate_css[:,is_cs_clade]
        clade_names = candidate_clade_names[is_cs_clade]
        
        if np.sum(~is_cs_clade) > 0:
            print(f"The following clades had fewer than {self.min_cssnps} specific SNPs and will be removed:")
            for cname in candidate_clade_names[~is_cs_clade]:
                print(f"{cname}\n")
                # Prune noninformative clades from tree
                if self.__path_to_candidate_clades_tree:
                    delnode = candidate_clades_tree.search_nodes(name=cname)
                    delnode.delete()
        
        # Trim cssnps_arr to just include positions with a csSNP
        cssnps = cssnps_arr[np.count_nonzero(cssnps_arr,1) > 0]
        cssnp_pos = core_pos[np.count_nonzero(cssnps_arr,1) > 0]

        # =========================================================================
        #  Report results
        # =========================================================================

        print('Classifier results:')
        for c in range(len(clade_names)):
            print('Clade ' + clade_names[c] + ': ' + str(np.count_nonzero(cssnps[:,c])) + ' csSNPs found')
        
        # =========================================================================
        #  Save classifier object
        # =========================================================================
        
        with gzip.open(self.__path_to_output_db, 'wb') as f:
            
            pickle.dump({'clades':candidate_clades,
                        'clade_names':clade_names,
                        'cssnps':cssnps,
                        'cssnp_pos':cssnp_pos,
                        'tree': candidate_clades_tree}, f)
        
def unanimous_to_clade(calls, sample_names, candidate_clades, clade_names, n, min_presence_core):
    '''For a list of clades defined by their daughter genomes, return all alleles
    along genomes that are unanimous to members of an individual clade. Clades can
    be ancestors/children of each other.
    
    Args:
        calls (arr): Array of major allele NT for each isolate (p x s) NATCG=01234.
        sample_names (arr): Array of sample names.\n
        candidate_clades (dict): Dictionary of genomes belonging to each clade.\n
        clade_names (list): List of clade names.\n
        n (float): % Ns within clade tolerated to be a csSNP (default 0.1).\n
        core (float): Minimum shared across samples to be a csSNP (default 0.9).\n

    Returns:
        unanimous_alleles (arr): pxc array listing unanimous alleles to an
        individual clade (0 if no allele is unanimous).

    '''
    
    # Filter for only core genome
    # is_core_genome = np.count_nonzero(maNT, axis=1)/len(maNT[1]) >= core
    # core_genome = maNT[is_core_genome]
    
    #Initialize output array (pxc)
    unanimous_alleles = np.zeros([len(calls),len(clade_names)])
    
    for i, cname in enumerate(clade_names):
    
        if not np.array([genome in sample_names for genome in candidate_clades[cname]]).all():
            raise Exception(f'Genomes in Clade: {cname} not found in candidate mutation table!')

        #Get indices of genomes on maNT object        
        clade_idx = [np.where(sample_names==name)[0][0] for name in candidate_clades[cname]]
        
        #maNT matrix containing just genomes belonging to this clade
        clade_calls = calls[:,clade_idx] 

        # Mask samples with too many ambiguous allele calls
        mask_fracNs = ( ((clade_calls>0).sum(axis=0)/len(clade_calls)) >= min_presence_core )
        if np.sum(mask_fracNs) < 2:
            raise Warning(f"After filtering, Clade {cname} does not have enough genomes to call unanimous alleles!")
        
        clade_calls_masked = clade_calls[:,mask_fracNs]

        # Pick unanimous alleles
        # Boolean - which ps are above n threshold   
        n_tol = (np.count_nonzero(clade_calls_masked,axis=1) / clade_calls_masked.shape[1]) >= 1-n
        
        # append a column of ns (0) to every position
        clade_samples_n = np.append(clade_calls_masked, np.zeros((len(calls),1)), axis=1)
        # Count number of unique values in each row. 
        # Good ps have 2 unique values: the unanimous allele and N.
        is_unanimous = np.count_nonzero(np.diff(np.sort(clade_samples_n)), axis=1)+1 == 2
        
        # Write to output array
        unanimous_alleles[:,i] = (n_tol & is_unanimous) * np.max(clade_calls_masked,axis=1)
        
        if np.count_nonzero((n_tol & is_unanimous) * np.max(clade_calls_masked,axis=1)) == 0:
            
            raise Warning(f"Clade {cname} does not have any unanimous alleles!")
            
    return unanimous_alleles

def unique_to_clade(maNT, unanimous_alleles, sample_names, candidate_clades, clade_names):
    '''Search for unique alleles among the unanimous alleles for a particular clade.
    Search occurs only against clades that aren't direct descendants of the target
    clade, nor direct ancestors on the path to the root, plus against any genomes 
    not belonging to any clade.

    Args:
        maNT (arr): Array of major allele NT for each isolate (p x s) NATCG=01234.
                    Note that this function will not do any filtering across positions\n.
        unanimous_alleles (arr): pxc array listing unanimous alleles to an
        individual clade (0 if no allele is unanimous).\n
        sample_names (arr): Array of sample names.\n
        candidate_clades (dict): Dictionary of genomes belonging to each clade.\n
        clade_names (list): List of clade names.\n

    Returns:
        css_mat (arr): DESCRIPTION.

    '''
            
    # Initialize output array
    css_mat = np.zeros((len(unanimous_alleles),len(clade_names)))
    
    # First exclude unclassified 
    cl = []
    for key, val in candidate_clades.items():
        cl = cl + val
    uncl_bool = np.in1d( sample_names,
                         cl)
    
    for c, cname in enumerate(clade_names):
        
        #Genomes to compare this clade against for uniqueness
        
        cp_bool = ~np.in1d( sample_names,
                            np.array(candidate_clades[cname]) ) & uncl_bool
                    
        # cp_bool = ~np.in1d( clade_names, np.unique(np.array(ancdesc)) )
        
        # Get alleles unique to this clade
        this_clade_alleles = unanimous_alleles[:,c]
        is_unique = np.sum( np.expand_dims(this_clade_alleles,1) == maNT[:,cp_bool]
                           ,axis=1) == 0
        
        this_clade_alleles[~is_unique] = 0
        css_mat[:,c] = this_clade_alleles
        # print(f"{cname}: {np.count_nonzero(css_mat[:,c])} csSNPs")
    
    return css_mat


class CladeCaller():
    '''
    Main controller of the Tree step.
    '''
    
    def __init__(self, path_to_nwk_file,
                 path_to_out_tree,
                 path_to_out_cladeIDs,
                 min_branch_len=100,
                 min_nsamples=3,
                 min_support=0.75,
                 path_to_cmt_file=False,
                 rescale=False):
        
        self.__path_to_nwk = path_to_nwk_file
        self.__path_to_cmt = path_to_cmt_file
        self.__path_to_out_cladeIDs = path_to_out_cladeIDs
        self.__path_to_out_tree = path_to_out_tree
        
        self.min_branch_len = min_branch_len
        self.min_nsamples = min_nsamples
        self.min_support = min_support
        
        self.rescale_bool = rescale
        
        if min_support > 1 or min_support < 0:
            raise IOError('Branch support threshold must be between 0 and 1!')
        
        if rescale:
            if not path_to_cmt_file:
                raise IOError('Rescaling a tree requires a candidate mutation table!')
                
    def main(self):
        
        # =====================================================================
        #  Load Data
        # =====================================================================
        print("Reading in file(s)...")

        self.tree = ete3.Tree(self.__path_to_nwk, format=0)
        # midpoint_root = self.tree.get_midpoint_outgroup()
        # self.tree.set_outgroup(midpoint_root)
        self.tree.standardize()

        self.tree_samples = self.tree.get_leaf_names()
        
        if self.__path_to_cmt:
            sample_names, _, counts, _, _, _ = helper.read_cmt(self.__path_to_cmt)
            sample_names = self.rphylip(sample_names)

            match_bool = np.in1d(sample_names, self.tree_samples)
            if np.count_nonzero(match_bool) != len(self.tree_samples):
                raise Exception('At least one sample from tree not found in candidate mutation table!')

            self.cmt_samples = sample_names[match_bool]
            self.counts = counts[:,:,match_bool]
            
        # =====================================================================
        #  Clade calling step
        # =====================================================================

        if self.rescale_bool:
            
            print("Rescaling...")

            scaled_tree, fig = self.rescale()
            
            fig.savefig('tree_linreg_output.pdf',format='pdf')
            
            print("Writing...")
                                    
            self.write_tree(scaled_tree)

        else:
            print("Calling clades...")

            clades, clade_names, clade_tips, called_tree = self.clade_caller()
            
            self.clade_names = clade_names
            self.clade_tips = clade_tips
            self.called_tree = called_tree
        
            print("Writing...")
                
            self.write_cladeIDs()
                    
            self.write_tree(self.called_tree)
        
        # if self.__path_to_out_tree_simplified:
        #     with open(path_to_cladestree_simplified_out,'w') as f:
        #         f.write(clades.write(format=1)) #1 includes node names


    def clade_caller(self):
        '''
        Call candidate clades for every branch of tree passing thresholds.
        '''
        
        # Initialize outputs
        good_nodes=[] # candidate clades
        clade_name=[] # name of candidate clade
        tips_sets=[]; tips_ls=[] # isolates defining clades
        
        #Go through nodes 
        for node in self.tree.traverse("preorder"):
            
            # Cut clades at long enough branch lengths and bootstrap values
            if node.is_leaf() is False and node.dist is not None \
                and node.dist >= self.min_branch_len \
                and len(node.get_leaves()) >= self.min_nsamples \
                and node.support >= self.min_support:
                
                
                good_nodes.append(node) # Save node object
                
                clade_name.append('tmp') # Create temp. name for naming function
                
                leaf_names = []
                for leaf in node.iter_leaves():
                    leaf_names.append(leaf.name)
                
                tips_sets.append(set(leaf_names)) # Save tip names as a set
                
                tips_ls.append(leaf_names)

        def fill_clade_names(parent, parent_name):
            '''
            Name daughter clades iteratively according to their parent 
            and create tree object reflecting structure.
            
            Naming follows this logic:
            1. Starting at the root, find the first daughter node (dnode) for 
            which dnode is a subset of the parent and only the parent 
            (i.e an immediate descendant)
            2. Name it with 'parent_name'.1
            3. Recur onto dnode (first daughter of 'parent'.1 will be 'parent'.1.1)
            4. Then increase number (1->2)
            5. Find the next dnode for which dnode is a subset of the parent 
            and only the parent. (i.e. sister to 'parent_name'.1). These will 
            thus be named 'parent'.2, 'parent'.3, etc.
            6. Continue until all names are filled
                
            '''
            number=1
            # Iterate through goodnodes
            for i in range(len(tips_sets)):
                
                # Is this goodnode a subset of the parent and only the parent?
                if tips_sets[i].issubset(parent) \
                    and sum([tips_sets[i].issubset(tips_sets[c]) for c in range(len(tips_sets)) if clade_name[c] == 'tmp']) == 1: 
                    #^ugly
                        
                    # Name it the parent name + number
                    clade_name[i] = parent_name + '.' + str(number)
                    # Then, recur onto this node
                    fill_clade_names(tips_sets[i], clade_name[i])
                    # Then increase number
                    number = number+1
                
        # Begin at the root
        fill_clade_names(set(self.tree.get_leaf_names()), 'C')
        
        #### Create ete3 graph structure from clade names ####
        # Note that this is NOT a binary tree and can have multifurcations + internal nodes    
        clades = ete3.Tree()
        for name in clade_name:
            # If clade is a direct descendant from root
            if name.rsplit('.', 1)[0] == 'C':
                # Add clade as child of root
                clades.add_child(name=name)
            # If there is another ancestor
            else:
                # Search for the ancestor and add clade as a child of that
                clades.search_nodes(name=name.rsplit('.', 1)[0])[0].add_child(name=name)
    
    
        #### Annotate phylogenetic tree with updated clade names ####
        new_tree = self.tree.copy()
        for new_node in new_tree.traverse('preorder'):
            if not new_node.is_leaf():
                new_node_tips = new_node.get_leaf_names()
                #Do tips of this node match good_nodes?
                if new_node_tips in tips_ls:
                    # print("match")
                    new_node.name = clade_name[tips_ls.index(new_node_tips)]
                else:
                    new_node.name=None
                    
                    
        return clades, clade_name, tips_ls, new_tree
        
    def rescale(self):
        '''
        Rescale a phylogeny from tree distance to # of SNPs using regression.
        '''
        
        tree_dm = self.tip_tip_distmat()

        maNT, _, _, _ = helper.mant(self.counts)

        snp_dm = helper.distmat(maNT,
                                self.cmt_samples)
        
        # Sort by rows AND columns
        tree_dm_sorted = self.sort_distmat(tree_dm)
        snp_dm_sorted = self.sort_distmat(snp_dm)
    
        # Flatten array
        snp_dists = snp_dm_sorted.to_numpy().flatten()
        snp_dists = np.delete(snp_dists, # Remove values on the diagonal
                              range(0, len(snp_dists), len(snp_dm_sorted) + 1), 0)

        tree_dists = tree_dm_sorted.to_numpy().flatten()
        tree_dists = np.delete(tree_dists, 
                               range(0, len(tree_dists), len(tree_dm_sorted) + 1), 0)
        
        tree_dists = tree_dists[snp_dists>0] # Remove SNP distances of 0
        snp_dists = snp_dists[snp_dists>0] # Remove SNP distances of 0

        ### Linear regression ###
        tree_dists_transform = tree_dists[:,np.newaxis]
        slope, _, _, _ = np.linalg.lstsq(tree_dists_transform, snp_dists)

        snp_dists_pred = slope * tree_dists
        
        # Calculate the total sum of squares (SS_tot) and the residual sum of squares (SS_res)
        ss_tot = np.sum((snp_dists - np.mean(snp_dists))**2)
        ss_res = np.sum((snp_dists - snp_dists_pred)**2)

        # Calculate R^2
        rsq = 1 - (ss_res / ss_tot)

        ## Plot linear regression
        fig = self.plot_regression(tree_dists, snp_dists, float(slope), float(rsq))

        ### Go through tree and scale all branch lengths ###
        newtree = self.tree.copy()
        
        for node in newtree.traverse(strategy='preorder'):
            if node.dist is not None:
                node.dist = (node.dist*slope)
                
        return newtree, fig
    
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

    def write_cladeIDs(self):
        '''
        Write clade_IDs file as .tsv
        '''
        
        with open(self.__path_to_out_cladeIDs, 'w') as f:
            for name, tips in zip(self.clade_names, self.clade_tips):
                for tip in tips:
                    f.write(f"{tip}\t{name}\n")
    
    def write_tree(self, tree):
        '''
        Write called tree to newick format
        '''
        with open(self.__path_to_out_tree,'w') as f:
            f.write(tree.write(format=1)) #1 includes node names

    @staticmethod
    def plot_regression(xs, ys, slope, rsq):
        
        fmt={'fontsize':15,
            'fontname':'Helvetica'}

        fig, axs = plt.subplots()
        
        axs.scatter(xs, ys, c='k', marker='o', alpha=0.1)
        axs.plot([0,max(xs)], [0,max(xs)*slope], color='r')
        
        axs.set_xlabel('Tree distances',**fmt)
        axs.set_ylabel('# Core genome SNPs', **fmt)
        axs.tick_params(axis='both', labelsize=12)

        axs.text(0.05, 0.85, f"y={float(slope):.2f}x\nr^2={float(rsq):.2f}",
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

def map_4_repisolates(path_to_candidate_clades, 
                      path_to_cluster_IDs):
    '''
    If tree is built off representative isolates, map clades defined on tree
    onto larger isolate collection.
    '''

    cand_clades, cand_clade_names = helper.read_clades_file(path_to_candidate_clades,
                                                            uncl_marker='-1')
    
    return
    
def rename_phylip(phylip2names_file, intree, outtree, 
                  outclustertree=False, rep_CMT_file=False):
    '''Given a renaming file, rename 10chr phylip names into long format

    Args:
        phylip2names_file (TYPE): DESCRIPTION.
        intree (TYPE): DESCRIPTION.
        outtree (TYPE): DESCRIPTION.

    Returns:
        None.

    '''
    # Get phylip2names as dict
    phylip2names=dict()
    with open(phylip2names_file) as f:
        for line in f:
            key, value = line.strip().split('\t')
            phylip2names[key] = value
            
    # Replace phylip tree names
    with open(intree) as f:
        nwk=f.read()
    #Replace with representative isolate name
    for i in phylip2names.keys():
        nwk=nwk.replace(i,phylip2names[i])
    with open(outtree,'w') as f:
        f.write(nwk)
    
    if outclustertree: # Optionally output tree named by cluster
    
        # Get which tree isolate belongs to which cluster
        with gzip.open(rep_CMT_file,'rb') as f:
            CMT=pickle.load(f); sample_names=CMT[0]; cluster_IDs=CMT[4]
        tree2cluster=dict()
        for sam,clu in zip(sample_names,cluster_IDs):
            tree2cluster[sam]=clu
            
        #Replace with cluster name
        for i in tree2cluster.keys():
            nwk=nwk.replace(i,'Cluster '+tree2cluster[i])
        with open(outclustertree,'w') as f:
            f.write(nwk)