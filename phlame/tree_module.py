#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:57:14 2022

@author: evanqu
"""

#%%
import phlame.helper_functions as helper
import numpy as np
import pandas as pd
import scipy.stats as stats
import ete3
import gzip
import pickle
import matplotlib.pyplot as plt
import os


#%% Testing
# os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_06_Ecoli/')

# path_to_nwk_file='trees/Ecoli_GTR_isonames_scaled_rooted.tre'
# path_to_cmt_file='CMTs/rep_Ecoli_candidate_mutation_table.pickle.gz'
# path_to_out_tree='trees/Ecoli_GTR_calls.tre'
# path_to_out_cladeIDs='trees/Ecoli_GTR_cladeIDs.txt'

# results = Tree(path_to_nwk_file,
#                 path_to_out_tree,
#                 path_to_out_cladeIDs,
#                 min_branch_len=1000,2
#                 min_nsamples=4,
#                 min_support=0.75,
#                 path_to_cmt_file=path_to_cmt_file,
#                 rescale=False)

# results.main()


# ###############################################################################

# phylip2names_file='trees/Ecoli_phylip2names.txt'
# intree='trees/Ecoli_GTR.tre'
# outtree='trees/Ecoli_GTR_isonames.tre'
# rename_phylip(phylip2names_file, intree, outtree, 
#                   outclustertree=False, rep_CMT_file=False)



# path_to_candidate_clades='Gvaginalis_piotii_HKY85_clades.txt'
# path_to_cluster_IDs='Sepi_acera_lineageIDs.txt'

# path_to_rep_IDs='Sepi_acera_subphylogroupIDs_REPONLY.csv'
# path_to_full_IDs='Sepi_acera_lineageIDs_nodoubletons.txt'
# path_to_out_file='Sepi_acera_subphylogroupIDs.csv'
# rep2full_conversion(path_to_rep_IDs, path_to_full_IDs,
#                         path_to_out_file)

#%% Fxns I want


class Tree():
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

        ### Linear regression ###
        linreg = stats.linregress(tree_dists, snp_dists)
        
        ### Go through tree and scale all branch lengths ###
        newtree = self.tree.copy()
        
        for node in newtree.traverse(strategy='preorder'):
            if node.dist is not None:
                node.dist = node.dist*linreg.slope
                
        ## Plot linear regression
        fig = self.plot_regression(tree_dists, snp_dists, linreg)
                
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
    def plot_regression(xs, ys,
                        linreg):
        
        fig, axs = plt.subplots()
        
        axs.scatter(xs, ys, c='k', marker='o', alpha=0.1)
        axs.plot([0,max(xs)], [0,max(xs)*linreg.slope], color='r')
        
        axs.set_xlabel('Tree distances')
        axs.set_ylabel('# Core genome substitutions')

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

#%% Old code
# def tip_tip_distmat(tree):
    
#     dm=np.zeros((len(tree),len(tree)))
#     names = []
    
#     for idx1, leaf1 in enumerate(tree.get_leaves()):
        
#         names.append(leaf1.name)
        
#         for idx2, leaf2 in enumerate(tree.get_leaves()): 
            
#             dm[idx1, idx2] = tree.get_distance(leaf1, leaf2)
    
#     dm_df = pd.DataFrame(dm, index=names, columns=names)
    
#     return dm_df
    
# def clade_caller(path_to_tree, 
#                  len_threshold, min_iso, min_support=1, 
#                  path_to_clades_out=False, 
#                  path_to_cladestree_out=False,
#                  path_to_cladestree_simplified_out=False):
#     '''Call candidate clades, subclades, sub-subclades etc. for every branch of tree passing thresholds.

#     Args:
#         path_to_tree (str): Path to phylogeny (.nwk file).
#         len_threshold (float): Minimum tree length to be a clade.
#         min_iso (int): Minimum number of isolates to be a clade.
#         min_support (float, optional): Minimum branch support to be a clade. Defaults to 1.
#         path_to_clades_out (str, optional): Path to export clades as .tsv file. Defaults to False.
#         path_to_cladestree_out (str, optional): Path to export clade tree as .nwk file. Defaults to False.

#     Raises:
#         IOError: DESCRIPTION.

#     Returns:
#         clades (TYPE): DESCRIPTION.
#         clade_name (TYPE): DESCRIPTION.
#         tips_ls (TYPE): DESCRIPTION.
#         new_tree (TYPE): DESCRIPTION.

#     '''
#     if min_support > 1 or min_support < 0:
#         raise IOError('Branch support threshold must be between 0 and 1!')

#     tree = ete3.Tree(path_to_tree, format=0)
#     midpoint_root = tree.get_midpoint_outgroup()
#     tree.set_outgroup(midpoint_root)
#     tree.standardize()

#     # Initialize outputs
#     good_nodes=[] # candidate clades
#     clade_name=[] # name of candidate clade
#     tips_sets=[]; tips_ls=[] # isolates defining clades
    
#     #Go through nodes 
#     for node in tree.traverse("preorder"):
        
#         # Cut clades at long enough branch lengths and bootstrap values
#         if node.is_leaf() is False and node.dist is not None and \
#             node.dist >= len_threshold and len(node.get_leaves()) >= min_iso \
#             and node.support >= min_support:
            
#             good_nodes.append(node) # Save node object
#             clade_name.append('tmp') # Create temp. name for naming function
#             leaf_names = []
#             for leaf in node.iter_leaves():
#                 leaf_names.append(leaf.name)
#             tips_sets.append(set(leaf_names)) # Save tip names as a set
#             tips_ls.append(leaf_names)
    
    
#     def fill_clade_names(parent, parent_name):
#         '''Name daughter clades iteratively according to their parent and create tree object reflecting structure.
        
#         This function will iteratively name each goodnode with the following logic:
#         1. Starting at the root, find the first daughter node (dnode) for which dnode 
#           is a subset of the parent and only the parent (i.e an immediate descendant)
#         2. Name it with 'parent'.1
#         3. Recur onto daughter node (e.g. first daughter of 'parent'.1 will be 'parent'.1.1)
#         4. then increase number (1->2)
#         5. Find the second daughter node for which dnode is a subset of the parent and only the parent
#             sister nodes will thus be named 'parent'.2, 'parent'.3, etc.
#         6. Continue until all names are filled
        
#         Args:
#             parent (set): Set of leaves belonging to the parent.
#             parent_name (str): String name of parent.
#             parent_node (class): Node corresponding to parent in ete3
#         Returns:
#             None.

#         '''
#         number=1
#         # Iterate through goodnodes
#         for i in range(len(tips_sets)):
            
#             # Is this goodnode a subset of the parent and only the parent?
#             if tips_sets[i].issubset(parent) \
#                 and sum([tips_sets[i].issubset(tips_sets[c]) for c in range(len(tips_sets)) if clade_name[c] == 'tmp']) == 1: 
#                 #^ugly
                    
#                 # Name it the parent name + number
#                 clade_name[i] = parent_name + '.' + str(number)
#                 # Then, recur onto this node
#                 fill_clade_names(tips_sets[i], clade_name[i])
#                 # Then increase number
#                 number = number+1
            
#     # Begin at the root
#     fill_clade_names(set(tree.get_leaf_names()), 'C')
    
#     #### Create ete3 graph structure from clade names ####
#     # Note that this is NOT a binary tree and can have multifurcations + internal nodes    
#     clades = ete3.Tree()
#     for name in clade_name:
#         # If clade is a direct descendant from root
#         if name.rsplit('.', 1)[0] == 'C':
#             # Add clade as child of root
#             clades.add_child(name=name)
#         # If there is another ancestor
#         else:
#             # Search for the ancestor and add clade as a child of that
#             clades.search_nodes(name=name.rsplit('.', 1)[0])[0].add_child(name=name)


#     #### Annotate phylogenetic tree with updated clade names ####
#     new_tree = tree.copy()
#     for new_node in new_tree.traverse('preorder'):
#         if not new_node.is_leaf():
#             new_node_tips = new_node.get_leaf_names()
#             #Do tips of this node match good_nodes?
#             if new_node_tips in tips_ls:
#                 # print("match")
#                 new_node.name = clade_name[tips_ls.index(new_node_tips)]
#             else:
#                 new_node.name=None
                
#     #### Save ####
#     if path_to_clades_out:
#         with open(path_to_clades_out,'w') as f:
#             for name, tips in zip(clade_name, tips_ls):
#                 for tip in tips:
#                     f.write(f"{tip}\t{name}\n")

#     if path_to_cladestree_out:
#         with open(path_to_cladestree_out,'w') as f:
#             f.write(new_tree.write(format=1)) #1 includes node names
    
#     if path_to_cladestree_simplified_out:
#         with open(path_to_cladestree_simplified_out,'w') as f:
#             f.write(clades.write(format=1)) #1 includes node names

#     return clades, clade_name, tips_ls, new_tree


# def get_delim(path_to_file):
    
#     with open(path_to_file,'r') as file:
#         firstline = file.readline()
       
#     if len(firstline.strip().split('\t'))==2:
#         dlim='\t'
#     elif len(firstline.strip().split(','))==2:
#         dlim=','
#     else:
#         raise Exception(f'File {path_to_file} not in tab-separated or comma-separated format')

#     return dlim

# def rep2full_conversion(path_to_rep_IDs, path_to_full_IDs,
#                         path_to_out_file):
#     '''
#     Convert classification system based on representative isolates
#     into classification system based on full isolates
#     '''
    
    
#     # Load in all isolates > rep isolates conversion
            
#     full_IDs = np.loadtxt(path_to_full_IDs, 
#                           delimiter=get_delim(path_to_full_IDs), 
#                           dtype=str)
    
#     rep_IDs = np.loadtxt(path_to_rep_IDs, 
#                          delimiter=get_delim(path_to_rep_IDs),
#                          dtype=str)
    
#     full_classes = np.copy(full_IDs)

#     # Loop through rep isolates
#     for rep, rep_class in zip(rep_IDs[:,0],rep_IDs[:,1]):
        
#         # Get all other isolates this rep covers
        
#         cluster_idx = np.where(full_IDs==rep)[0][0]
#         cluster = full_IDs[cluster_idx,1]
#         cluster_bool = np.in1d(full_IDs[:,1], np.array(cluster))
    
#         full_classes[cluster_bool,1] = rep_class
    
#     full_classes_df = pd.DataFrame(full_classes)
    
#     full_classes_df.to_csv(path_to_out_file, sep=',', header=False, index=False)            
            
# def rescale(path_to_nwk_file, path_to_cmt_file, path_to_out_nwk=False):
#     ''' Convert branch lengths on core genome SNP tree into number of substitutions
#     based on linear regression.

#     Args:
#         path_to_nwk_file (str): Path to newick file.
#         path_to_cmt_file (str): Path to candidate mutation table file.
#         path_to_out_nwk (str, optional): If specified, write rescaled tree as newick to path. Defaults to False.

#     Returns:
#         newtree (class): skbio TreeNode object.

#     '''
    
#     ### Load in tree ###

#     ete3tree = ete3.Tree(path_to_nwk_file, format=0)
#     midpoint_root = ete3tree.get_midpoint_outgroup()
#     ete3tree.set_outgroup(midpoint_root)
#     ete3tree.standardize()
    
#     # List of samples in tree
#     tree_samples = ete3tree.get_leaf_names()
    
#     # Calc tree distmat
#     tree_dm = tip_tip_distmat(ete3tree)
    
#     ### Load in CMT ###
#     sample_names, _, counts, _, _, _ = helper.read_cmt(path_to_cmt_file)
#     sample_names = rphylip(sample_names)
    
#     maNT, _, _, _ = helper.mant(counts)
    
#     # Check which samples made it into the tree
#     tree_bool = np.in1d(sample_names, tree_samples)
    
#     if np.count_nonzero(tree_bool) != len(tree_samples):
#         raise Exception('At least one tip name from tree not found in candidate mutation table!')
        
#     snp_dm = helper.distmat(maNT[:,tree_bool], 
#                             sample_names[tree_bool])
#     snp_dm = pd.DataFrame(snp_dm, 
#                           index=sample_names[tree_bool], 
#                           columns=sample_names[tree_bool])
        
#     ### Linear regression ###
#     tree_dm_sort = tree_dm.sort_index()
#     tree_dm_sort = tree_dm_sort.reindex(sorted(tree_dm_sort.columns), axis=1)
#     snp_dm_sort = snp_dm.sort_index()
#     snp_dm_sort = snp_dm_sort.reindex(sorted(snp_dm_sort.columns), axis=1)

#     # Flatten array
#     snp_dists = snp_dm_sort.to_numpy().flatten()
#     snp_dists = np.delete(snp_dists, # Remove on the diagonal
#                           range(0, len(snp_dists), len(snp_dm_sort) + 1), 0)
    
#     tree_dists = tree_dm_sort.to_numpy().flatten()
#     tree_dists = np.delete(tree_dists, 
#                            range(0, len(tree_dists), len(snp_dm_sort) + 1), 0)

    
#     linreg = stats.linregress(tree_dists, snp_dists)
    
#     ### Go through tree and scale all branch lengths ###
#     newtree = ete3tree.copy()
    
#     # =========================================================================
#     #  Save data
#     # =========================================================================
    
#     for node in newtree.traverse(strategy='preorder'):
#         if node.dist is not None:
#             node.dist = node.dist*linreg.slope
            
#     if path_to_out_nwk:
#         newtree.write(outfile=path_to_out_nwk, format=0)
        
#     # =========================================================================
#     #  Plot linear regression
#     # =========================================================================
        
#     plt.scatter(tree_dists, snp_dists, c='k', marker='o', alpha=0.1)
#     plt.plot([0,max(tree_dists)], [0,max(tree_dists)*linreg.slope], color='r')
#     plt.xlabel('Tree distances')
#     plt.ylabel('# Core genome substitutions')
    
#     return newtree
