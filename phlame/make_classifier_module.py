#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to build a PHLAME classifier.
@author: evanqu
"""

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

#%% Testing

# Wishlist:
# Check that files exist before running

# import os
# os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/benchmarks/2024_01_curate_assemblies')

# path_to_cmt='Sepi_acera_CMT.pickle.gz'    
# path_to_candidate_clades='Sepi_acera_lineageIDs_curated.txt'
# path_to_output_classifier='test.classifier'


# results = MakeDB(path_to_cmt,
#                  path_to_candidate_clades,
#                  path_to_output_classifier,
#                  path_to_candidate_clades_tree=False,
#                  min_cssnps=10,
#                  maxn=0.1,
#                  core=0.5,
#                  max_outgroup=0.1,
#                  min_maf_for_call=0.75,
#                  min_strand_cov_for_call=2,
#                  max_qual_for_call=-30)

# results.main()

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
        sample_names = rphylip(sample_names)
        maNT, maf, _, _ = helper.mant(counts)

        coverage_f_strand=counts[0:4,:,:].sum(axis=0) #should be pxs
        coverage_r_strand=counts[4:8,:,:].sum(axis=0)

        candidate_clades, candidate_clade_names = read_clades_file(self.__path_to_candidate_clades,
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
            max_outgroup_bool = ( (np.count_nonzero(outgroup_calls, axis=1)/len(outgroup_calls[1])) >= self.max_outgroup )

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


# def make_classifier(path_to_cmt, path_to_output_classifier, path_to_candidate_clades,
#                     path_to_candidate_clades_tree=False, min_cssnps=10, n=0.1, core=0.9):
#     '''Given a list of clades and a corresponding array of polymorphic positions, 
#        find mutations likely to have originated along the branch leading to that
#        clade (clade-specific SNPs). These correspond to mutations that are shared 
#        among all daughters of a particular clade (unanimous) which are also not 
#        found in any other genome (unique).

#     Args:
#         path_to_cmt (str): Path to candidate mutation table file.\n
#         path_to_output_classifier (str): Path to output classifier object.\n
#         candidate_clades (str): Path to file listing candidate clades (csv).\n
#         candidate_clades_tree (str, optional): Path to candidate clades tree (nwk). 
#         Optional, defaults to False.\n
#         min_cssnps (int): Minimum number of csSNPs to be included as a clade.\n
#         n (float): % Ns within clade tolerated to be a csSNP (default 0.1).\n
#         core (float): Minimum shared across samples to be a csSNP (default 0.9).\n
#         multi (bool): 
#     Returns:
#         None.

#     '''
#     # =========================================================================
#     # Read in files
#     # =========================================================================
#     print('Reading in files...')
#     sample_names, pos, counts, _, _, in_outgroup = helper.read_cmt(path_to_cmt)
#     sample_names = rphylip(sample_names)
#     maNT, _, _, _ = helper.mant(counts)

#     candidate_clades, candidate_clade_names = read_clades_file(path_to_candidate_clades,
#                                                                uncl_marker='-1')
    
    
#     if path_to_candidate_clades_tree:
#         candidate_clades_tree = ete3.Tree(path_to_candidate_clades_tree, format=1)
#         #1 includes node names
#     else:
#         candidate_clades_tree = False
        
#     # =========================================================================
#     # Get csSNPs for every clade
#     # =========================================================================
    
#     # Filter for only core genome
#     is_core_genome = np.count_nonzero(maNT, axis=1)/len(maNT[1]) >= core
#     core_maNT = maNT[is_core_genome]
#     core_pos = pos[is_core_genome]
#     print(f"Number of core positions: {len(core_pos)}/{len(pos)}")
        
#     #Call csSNPs
#     print('Getting unanimous alleles...')
#     unanimous_alleles = unanimous_to_clade(core_maNT, sample_names,
#                                            candidate_clades, candidate_clade_names,
#                                            n, core)

#     print('Getting unique alleles...')
#     candidate_css = unique_to_clade(core_maNT, unanimous_alleles, sample_names,
#                                  candidate_clades, candidate_clade_names)
    
#     # =========================================================================
#     #  Remove clades without enough csSNPs
#     # =========================================================================
    
#     is_cs_clade = np.count_nonzero(candidate_css,0) > min_cssnps
    
#     cssnps_arr = candidate_css[:,is_cs_clade]
#     clade_names = candidate_clade_names[is_cs_clade]
    
#     if np.sum(~is_cs_clade) > 0:
#         print(f"The following clades had fewer than {min_cssnps} specific SNPs and will be removed:")
#         for cname in candidate_clade_names[~is_cs_clade]:
#             print(f"{cname}\n")
#             # Prune noninformative clades from tree
#             if path_to_candidate_clades_tree:
#                 delnode = candidate_clades_tree.search_nodes(name=cname)
#                 delnode.delete()
    
#     # Trim cssnps_arr to just include positions with a csSNP
#     cssnps = cssnps_arr[np.count_nonzero(cssnps_arr,1) > 0]
#     cssnp_pos = core_pos[np.count_nonzero(cssnps_arr,1) > 0]

#     # =========================================================================
#     #  Report results
#     # =========================================================================

#     print('Classifier results:')
#     for c in range(len(clade_names)):
#         print('Clade ' + clade_names[c] + ': ' + str(np.count_nonzero(cssnps[:,c])) + ' csSNPs found')
    
#     # =========================================================================
#     #  Save classifier object
#     # =========================================================================
    
#     with gzip.open(path_to_output_classifier, 'wb') as f:
        
#         pickle.dump({'clades':candidate_clades,
#                      'clade_names':clade_names,
#                      'cssnps':cssnps,
#                      'cssnp_pos':cssnp_pos,
#                      'tree': candidate_clades_tree}, f)

# def read_candidate_clades_file(path_to_candidate_clades):
#     '''Read in candidate clades .csv file into dict

#     Args:
#         path_to_candidate_clades (str): .csv file listing candidate clades as 
#                                         follows: name,genome1,genome2,..,genomeN.

#     Returns:
#         candidate_clades_dct (dict): Dictionary giving the genomes 
#                                       belonging to each clade.

#     '''
#     candidate_clades_dct = {}
#     clade_names = []

#     with open(path_to_candidate_clades) as f:
#         for line in f:
#             ls = line.rstrip(',\n').split(',')
#             candidate_clades_dct[ls[0]] = ls[1:]
#             clade_names.append(ls[0])
        
#     return candidate_clades_dct, np.array(clade_names)
    
    
def read_clades_file(path_to_clades_file, uncl_marker):
    '''
    Read in a clades file.
    '''        
    
    # Get delimiter
    with open(path_to_clades_file,'r') as file:
        firstline = file.readline()
   
    if len(firstline.strip().split('\t'))==2:
        dlim='\t'
    elif len(firstline.strip().split(','))==2:
        dlim=','
    
    # Read in file
    clade_ids = np.loadtxt(path_to_clades_file, 
                           delimiter=dlim, 
                           dtype=str)
    
    # Reshape into dictionary
    clades_dct = dict()
    clade_names = []
    # Loop through unique clades & grab samples
    for clade in np.unique(clade_ids[:,1]):
        
        if clade==uncl_marker:
            continue
        
        isclade_bool = np.in1d(clade_ids[:,1], clade)
        clade_samples = clade_ids[isclade_bool,0].tolist()
        
        clades_dct[clade] = clade_samples
        clade_names.append(clade)
            
    return clades_dct, np.array(clade_names)

def rphylip(sample_names):
    '''Change : to | for consistency with phylip format'''
    
    rename = [sam.replace(':','|') for sam in sample_names]
    
    return np.array(rename)

    
    
# %%
