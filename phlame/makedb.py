#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to build a PHLAME classifier.
@author: evanqu
"""

#%%
import os
import numpy as np
# import h5py
import pickle
import gzip
import ete3

import phlame.helper_functions as helper

#%% FXNs

class MakeDB():
    '''
    Main controller of the makedb step.
    '''

    def __init__(self, 
                 path_to_cmt,
                 path_to_tree,
                 path_to_output_clades,
                 path_to_output_db,
                 outgroup_str=None,
                 path_to_input_clades=None,
                 min_branch_len=100,
                 min_nsamples=3,
                 min_support=0.75,
                 min_snps=10,
                 maxn=0.1,
                 core=0.9,
                 min_maf_for_call=0.75, 
                 min_strand_cov_for_call=2,
                 max_qual_for_call=-30,
                 max_frac_ambiguous=0.5,
                 max_outgroup=False
                 ):

        self.__path_to_cmt = path_to_cmt
        self.__path_to_tree = path_to_tree
        self.path_to_output_clades = path_to_output_clades
        self.__path_to_output_db = path_to_output_db
        
        self.path_to_input_clades = path_to_input_clades

        self.outgroup_str = outgroup_str

        # Clade calling parameters
        self.min_branch_len = min_branch_len
        self.min_nsamples = min_nsamples
        self.min_support = min_support
        
        # Database parameters
        self.min_snps = min_snps
        self.maxn = maxn
        self.core = core

        self.max_outgroup = max_outgroup

        self.min_maf_for_call = min_maf_for_call
        self.min_strand_cov_for_call = min_strand_cov_for_call
        self.max_qual_for_call = max_qual_for_call
        self.max_frac_ambiguous = max_frac_ambiguous

    def readin(self):
        '''
        Main function to read in files
        '''

        self.CMT = helper.CandidateMutationTable(self.__path_to_cmt)
        
        self.sample_names = helper.rphylip(self.CMT.sample_names)
        
        self.tree = ete3.Tree(self.__path_to_tree, format=0)
        # midpoint_root = self.tree.get_midpoint_outgroup()
        # self.tree.set_outgroup(midpoint_root)
        self.tree.standardize()
            
    def main(self):
        
        # =========================================================================
        # Read in files
        # =========================================================================
        print('Reading in files...')
        
        self.readin()

        # =========================================================================
        # Clade caller
        # =========================================================================

        if self.path_to_input_clades:

            print(f'Using information from the following file to define clades:')
            print(os.path.basename(self.path_to_input_clades))
            candidate_clades, candidate_clade_names = helper.read_clades_file(self.path_to_input_clades, uncl_marker='-1')

        else:
            
            self.call_clades_check()
            
            candidate_clades, candidate_clade_names, new_tree = self.call_clades()

            # Write clade IDs
            self.write_cladeIDs(candidate_clades,
                                self.path_to_output_clades)
        
        # =========================================================================
        # maNT calls
        # =========================================================================

        self.maNT, maf, _, _ = helper.mant(self.CMT.counts)

        # =========================================================================
        # Define ingroup and outgroup
        # =========================================================================

        self.define_outgroup()
        
        # =========================================================================
        # Do pre-filtering of allele calls
        # =========================================================================

        # Filter for only core genome positions
        is_core_genome = np.count_nonzero(self.ingroup, axis=1)/len(self.ingroup[1]) >= self.core
        core_maNT = self.ingroup[is_core_genome]
        core_pos = self.CMT.pos[is_core_genome]
        print(f"Number of core positions: {len(core_pos)}/{len(self.CMT.pos)}")
        if np.count_nonzero(is_core_genome) < len(is_core_genome)/5:
            print(f'Warning: Less than 20% of positions are core to > {self.core*100}% of samples!')
            print(f'Consider lowering the core genome threshold or checking the input data.')

        # Mask ambiguous allele calls
        calls = np.copy(core_maNT)
        # calls[ quals[is_core_genome] > self.max_qual_for_call ] = 0
        # calls[ ingroup_maf[is_core_genome] < self.min_maf_for_call ] = 0

        # Mask samples with too many ambiguous allele calls
        fracNs_bool = ( ((calls>0).sum(axis=0)/len(calls)) >= self.max_frac_ambiguous )
        
        if np.count_nonzero(~fracNs_bool) > 0:
            print('The following samples have too many ambiguous allele calls and will not be considered:')
            print('\n'.join(self.ingroup_sample_names[~fracNs_bool]))

        if np.count_nonzero(fracNs_bool) < 2:
            raise Warning('After filtering, there are fewer than 3 samples!')

        # Moving this to inside the unaminous_to_clade function
        # calls = calls[:,mask_fracNs]
        # ingroup_sample_names = ingroup_sample_names[mask_fracNs]

        # =========================================================================
        # Get csSNPs for every clade
        # =========================================================================
        
        print(len(self.sample_names))
        print(len(self.ingroup_sample_names))

        #Call csSNPs
        print('Getting unanimous alleles...')
        unanimous_alleles = unanimous_to_clade(calls, self.ingroup_sample_names,
                                               candidate_clades, candidate_clade_names,
                                               self.maxn, self.max_frac_ambiguous)

        print('Getting unique alleles...')
        candidate_css = unique_to_clade(calls, unanimous_alleles, self.ingroup_sample_names,
                                        candidate_clades, candidate_clade_names)
        
        # Remove positions aligning to too many outgroup genomes
        if len(self.outgroup_sample_names) > 0:
            print('Removing positions present in outgroup genomes...')
            max_outgroup_bool = ( (np.count_nonzero(self.outgroup, axis=1)/len(self.outgroup[1])) 
                                 >= self.max_outgroup )

            candidate_css[max_outgroup_bool[is_core_genome]] = 0
            print(f"Removed {np.count_nonzero(max_outgroup_bool[is_core_genome])}/{len(candidate_css)} positions present >{self.max_outgroup*100}% of outgroup genomes")

        # =========================================================================
        #  Remove clades without enough csSNPs
        # =========================================================================
        
        is_cs_clade = np.count_nonzero(candidate_css,0) > self.min_snps
        
        cssnps_arr = candidate_css[:,is_cs_clade]
        clade_names = candidate_clade_names[is_cs_clade]
        
        if np.sum(~is_cs_clade) > 0:
            print(f"The following clades had fewer than {self.min_snps} specific SNPs and will be removed:")
            for cname in candidate_clade_names[~is_cs_clade]:
                print(f"{cname}\n")
       
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
                        'cssnp_pos':cssnp_pos}, f)

    def parse_outgroup(self):

        if self.outgroup_str:

            self.outgroup_ls = [item.strip() for item in self.outgroup_str.split(",")]
            
            # Check that all outgroup genomes are present in candidate mutation table
            if not np.array([genome in self.sample_names for genome in self.outgroup_ls]).all():
                raise Exception('The following outgroup genomes are not present in the candidate mutation table: ' +
                                ', '.join([genome for genome in self.outgroup_ls if genome not in self.sample_names]))
            
            # outgroup_bool = np.in1d(self.sample_names, self.outgroup_ls)
        
            # # Check if any outgroup genomes are not in sample_names
            # if np.sum(outgroup_bool) < len(self.outgroup_ls):
            #     print('Warning: The following outgroup genomes could not be found in the candidate mutation table. Will proceed anyway. ' +
            #           '\n'.join([genome for genome in self.outgroup_ls if genome not in self.sample_names]))
                
            # if np.sum(outgroup_bool) == 0:
            #     raise Exception('None of the outgroup genomes given are present!')

        else:
            self.outgroup_ls = []


    def define_outgroup(self):

        self.parse_outgroup()
        
        # Define outgroup
        outgroup_bool = np.in1d(self.sample_names, self.outgroup_ls)
        
        if np.sum(outgroup_bool) == 0:
            self.ingroup = self.maNT
            self.ingroup_sample_names = self.sample_names
            self.outgroup = np.array([])
            self.outgroup_sample_names = np.array([])
        
        else:
            self.ingroup = self.maNT[:,~outgroup_bool]
            self.ingroup_sample_names = self.sample_names[~outgroup_bool]
            self.outgroup = self.maNT[:,outgroup_bool]
            self.outgroup_sample_names = self.sample_names[outgroup_bool]


    def call_clades_check(self):
        '''
        Check that the parameters passed to call_clades make sense
        '''

        dists = []
        for node in self.tree.traverse("preorder"):
            dists.append(node.dist)

        if self.min_branch_len > max(dists):
            raise IOError(f'Minimum branch length threshold ({self.min_branch_len}) is longer than the longest branch in the tree!')
        
        if len(self.tree.get_leaf_names()) < self.min_nsamples*2:
            raise IOError(f"There are fewer than {self.min_nsamples} samples in the tree. It will not be possible to call more than one clade (Minimum number of samples in a clade is {self.min_nsamples}).")
        

    def call_clades(self):
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
        # graph = ete3.Tree()
        # for name in clade_name:
        #     # If clade is a direct descendant from root
        #     if name.rsplit('.', 1)[0] == 'C':
        #         # Add clade as child of root
        #         graph.add_child(name=name)
        #     # If there is another ancestor
        #     else:
        #         # Search for the ancestor and add clade as a child of that
        #         graph.search_nodes(name=name.rsplit('.', 1)[0])[0].add_child(name=name)
    
    
        # Make clades dict object
        clades = {}
        for name, tips in zip(clade_name, tips_ls):
            clades[name] = tips

        # Annotate phylogenetic tree with updated clade names
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


        return clades, np.array(clade_name), new_tree
    
    @staticmethod
    def write_cladeIDs(clades, output_file):
        '''
        Write clade_IDs file as .tsv
        '''
        
        with open(output_file, 'w') as f:
            for clade_name, tips in zip(clades.keys(), clades.values()):
                for tip in tips:
                    f.write(f"{tip}\t{clade_name}\n")
    
    def write_tree(self, tree):
        '''
        Write called tree to newick format
        '''
        with open(self.__path_to_out_tree,'w') as f:
            f.write(tree.write(format=1)) #1 includes node names

    @staticmethod
    def rphylip(sample_names):
        '''Change : to | for consistency with phylip format'''
        
        rename = [sam.replace(':','|') for sam in sample_names]
        
        return np.array(rename)    

        
def unanimous_to_clade(calls, sample_names, candidate_clades, clade_names, n, max_frac_ambiguous):
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
        mask_fracNs = ( ((clade_calls>0).sum(axis=0)/len(clade_calls)) >= max_frac_ambiguous )
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


# class CladeCaller():
#     '''
#     Main controller of the Tree step.
#     '''
    
#     def __init__(self, path_to_nwk_file,
#                  path_to_out_tree,
#                  path_to_out_cladeIDs,
#                  min_branch_len=100,
#                  min_nsamples=3,
#                  min_support=0.75,
#                  path_to_cmt_file=False,
#                  rescale=False):
        
#         self.__path_to_out_cladeIDs = path_to_out_cladeIDs
#         self.__path_to_out_tree = path_to_out_tree
        
#         self.min_branch_len = min_branch_len
#         self.min_nsamples = min_nsamples
#         self.min_support = min_support
        
#         self.rescale_bool = rescale
        
#         if min_support > 1 or min_support < 0:
#             raise IOError('Branch support threshold must be between 0 and 1!')
        
#         if rescale:
#             if not path_to_cmt_file:
#                 raise IOError('Rescaling a tree requires a candidate mutation table!')
                
#     def main(self):
        
#         print("Calling clades...")

#         clades, clade_names, clade_tips, called_tree = self.clade_caller()
        
#         self.clade_names = clade_names
#         self.clade_tips = clade_tips
#         self.called_tree = called_tree
    
#         # Do the writing in the controller
#         # print("Writing...")
#         # self.write_cladeIDs()
#         # self.write_tree(self.called_tree)


#     def clade_caller(self):
#         '''
#         Call candidate clades for every branch of tree passing thresholds.
#         '''
        
#         # Initialize outputs
#         good_nodes=[] # candidate clades
#         clade_name=[] # name of candidate clade
#         tips_sets=[]; tips_ls=[] # isolates defining clades
        
#         #Go through nodes 
#         for node in self.tree.traverse("preorder"):
            
#             # Cut clades at long enough branch lengths and bootstrap values
#             if node.is_leaf() is False and node.dist is not None \
#                 and node.dist >= self.min_branch_len \
#                 and len(node.get_leaves()) >= self.min_nsamples \
#                 and node.support >= self.min_support:
                
                
#                 good_nodes.append(node) # Save node object
                
#                 clade_name.append('tmp') # Create temp. name for naming function
                
#                 leaf_names = []
#                 for leaf in node.iter_leaves():
#                     leaf_names.append(leaf.name)
                
#                 tips_sets.append(set(leaf_names)) # Save tip names as a set
                
#                 tips_ls.append(leaf_names)

#         def fill_clade_names(parent, parent_name):
#             '''
#             Name daughter clades iteratively according to their parent 
#             and create tree object reflecting structure.
            
#             Naming follows this logic:
#             1. Starting at the root, find the first daughter node (dnode) for 
#             which dnode is a subset of the parent and only the parent 
#             (i.e an immediate descendant)
#             2. Name it with 'parent_name'.1
#             3. Recur onto dnode (first daughter of 'parent'.1 will be 'parent'.1.1)
#             4. Then increase number (1->2)
#             5. Find the next dnode for which dnode is a subset of the parent 
#             and only the parent. (i.e. sister to 'parent_name'.1). These will 
#             thus be named 'parent'.2, 'parent'.3, etc.
#             6. Continue until all names are filled
                
#             '''
#             number=1
#             # Iterate through goodnodes
#             for i in range(len(tips_sets)):
                
#                 # Is this goodnode a subset of the parent and only the parent?
#                 if tips_sets[i].issubset(parent) \
#                     and sum([tips_sets[i].issubset(tips_sets[c]) for c in range(len(tips_sets)) if clade_name[c] == 'tmp']) == 1: 
#                     #^ugly
                        
#                     # Name it the parent name + number
#                     clade_name[i] = parent_name + '.' + str(number)
#                     # Then, recur onto this node
#                     fill_clade_names(tips_sets[i], clade_name[i])
#                     # Then increase number
#                     number = number+1
                
#         # Begin at the root
#         fill_clade_names(set(self.tree.get_leaf_names()), 'C')
        
#         #### Create ete3 graph structure from clade names ####
#         # Note that this is NOT a binary tree and can have multifurcations + internal nodes    
#         clades = ete3.Tree()
#         for name in clade_name:
#             # If clade is a direct descendant from root
#             if name.rsplit('.', 1)[0] == 'C':
#                 # Add clade as child of root
#                 clades.add_child(name=name)
#             # If there is another ancestor
#             else:
#                 # Search for the ancestor and add clade as a child of that
#                 clades.search_nodes(name=name.rsplit('.', 1)[0])[0].add_child(name=name)
    
    
#         #### Annotate phylogenetic tree with updated clade names ####
#         new_tree = self.tree.copy()
#         for new_node in new_tree.traverse('preorder'):
#             if not new_node.is_leaf():
#                 new_node_tips = new_node.get_leaf_names()
#                 #Do tips of this node match good_nodes?
#                 if new_node_tips in tips_ls:
#                     # print("match")
#                     new_node.name = clade_name[tips_ls.index(new_node_tips)]
#                 else:
#                     new_node.name=None
                    
                    
#         return clades, np.array(clade_name), tips_ls, new_tree
            
#     def write_cladeIDs(self):
#         '''
#         Write clade_IDs file as .tsv
#         '''
        
#         with open(self.__path_to_out_cladeIDs, 'w') as f:
#             for name, tips in zip(self.clade_names, self.clade_tips):
#                 for tip in tips:
#                     f.write(f"{tip}\t{name}\n")
    
#     def write_tree(self, tree):
#         '''
#         Write called tree to newick format
#         '''
#         with open(self.__path_to_out_tree,'w') as f:
#             f.write(tree.write(format=1)) #1 includes node names

#     @staticmethod
#     def rphylip(sample_names):
#         '''Change : to | for consistency with phylip format'''
        
#         rename = [sam.replace(':','|') for sam in sample_names]
        
#         return np.array(rename)    
    

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