#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:28:18 2022

@author: evanqu
"""
import numpy as np
import pandas as pd
import h5py
import math
import pickle
import os

#%%

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/strainslicer/dev/snakemake_make_classifier')

calls, calls_pos = read_calls_table('Arolyn_data_maNT.mat', calls='Calls_all', pos='p_all')
clade_IDs = np.genfromtxt('Aro_superSLSTs.txt', dtype='str')
clades, clade_names = reshape_clade_IDs(clade_IDs, 'X')

csSNPs, csSNP_pos = find_clade_specific_snps(calls, calls_pos, clades, 0.1, 0.1)

#%%
def make_classifier(matlab_file, clades_file):
    '''
    Wrapper for creating classifiers
    '''
    calls, calls_pos = read_calls_table(matlab_file, calls='Calls_all', pos='p_all')
    
    clade_IDs = np.genfromtxt(clades_file, dtype='str')
    clades, clade_names = reshape_clade_IDs(clade_IDs, 'X')
    
    csSNPs, csSNP_pos = find_clade_specific_snps(calls, calls_pos, clades, 0.1, 0.1)
        
    return csSNPs, csSNP_pos, clade_names

def read_calls_table(matlab_file, calls, pos):
    #Read in candidate mutation table and output variables as individual arrays
    # Note: args are string names of variables
    
    file = h5py.File(matlab_file)
    # read data2dict
    arrays = {}
    for k, v in file.items():
        arrays[k] = np.array(v)
    # assign various variables. NOTE: import transposes arrays (only x/y but not z axis)
    
    calls_table = arrays[calls]
    calls_table = calls_table.transpose().astype(np.int64)
    
    p = arrays[pos] # Note: p is 1-based
    p = p.flatten().astype(np.int64)
    # SampleNames saved in specific object pointer format which requires this technical loop below to be resolved
    #sampleNames = []
    #mygroup = file['SampleNames']
    #for s in mygroup:
    #    obj=file[s[0]]
    #    str1 = ''.join([chr(i[0]) for i in obj])
    #    sampleNames.append(str1)

    return [calls_table, p]

#convert sx1 array of clade IDs to list of clades w/ clade indices
def reshape_clade_IDs(clade_ids, unclassified_marker):
    '''
    Input:
        clade_ids: sx1 array of clade ids
        unclassified_marker: marker indicating unclassified @ that level. 
        Note: if nothing is unclassified, put anything & ignore warning
    '''
    
    clades = []; clade_names = []
    
    for c in np.unique(clade_ids):
        clades.append(np.where(clade_ids==c)[0])
        clade_names.append(str(c))
        
    good_clades = [clades[i] for i in range(len(clade_names)) if unclassified_marker not in clade_names[i] ]
    good_clade_names = [clade_names[i] for i in range(len(clade_names)) if unclassified_marker not in clade_names[i]]
    
    if len(good_clade_names) == len(clade_names):    
        print('Note: Nothing was found as unclassified in clade IDs. Ignore if intentional')
            
    return good_clades, good_clade_names


def find_clade_specific_snps(maNT, maNT_pos, clades, n, core):
    '''
    Input:
        maNT: major allele NT for isolate samples (p x s)
        maNT_pos: position on the reference for each maNT (p x 1)
        clades: list of arrays giving which isolates belong to which clades
        n: % Ns within clade tolerated (default 0.1)
        core: % of isolates tolerated to not have a position (default 0.1)
    '''
    
    is_core_genome = np.count_nonzero(maNT, axis=1)/len(maNT[1]) > 1-core #core if p is in %n of all isolates
    core_genome = maNT[is_core_genome]; core_pos = maNT_pos[is_core_genome]
    
    #finds all    
    def unanimous_to_clade():
        #Initialize output array, size pxc
        unanimous_clade_alleles = np.zeros([len(core_genome),len(clades)])

        for c in range(len(clades)):
        #Loop through every clade and pull out maNT for only samples belonging to that clade
            this_clade_samples = core_genome[:,clades[c]]
                                    
            n_tolerance = (np.count_nonzero(this_clade_samples,axis=1) / this_clade_samples.shape[1]) > 1-n
            
            #Cool trick for allele calling that's tolerant of Ns: append a column of Ns (0) to every position
            clade_samples_add_n = np.append(this_clade_samples, np.zeros((len(core_genome),1)), axis=1)
            #Count number of unique values in each row. Good pos have 2 unique values: the unanimous allele and N.
            is_unanimous_within_clade = np.count_nonzero(np.diff(np.sort(clade_samples_add_n)), axis=1)+1 == 2
            
            #combine to output array
            unanimous_clade_alleles[:,c] = (n_tolerance & is_unanimous_within_clade)*np.max(this_clade_samples,axis=1)
                
        return unanimous_clade_alleles
    
    def unique_to_clade(unanimous_clade_alleles):
        #ask if a position is 'core' - aka present (non-n) in least 90% of all clades
        #is_core_genome_byclade = (np.count_nonzero(unanimous_clade_alleles,axis=1) / unanimous_clade_alleles.shape[1]) > 1-core #core if p is in n% of all clades    
        #is_core_genome = np.count_nonzero(maNT, axis=1)/len(maNT[1]) > 1-core #core if p is in %n of all isolates

        
        #append column of Ns to every position
        core_genome_addn = np.append(unanimous_clade_alleles, np.zeros((len(unanimous_clade_alleles),2)), axis=1)
        
        #initialize some arrays for output
        is_polymorphic = []; candidate_cssnps = np.zeros([len(unanimous_clade_alleles),len(clades)])
        
        #get polymorphic positions
        for p in range(len(unanimous_clade_alleles)):
            alleles_across_clades = np.unique(core_genome_addn[p], return_index=True,return_counts=True)
            is_polymorphic.append( len(alleles_across_clades[0]) > 2 )
            
            #if this position has a unique allele, grab NT and clade info
            #This is UGLY
            if (alleles_across_clades[2] == 1).any():
                unique_allele = alleles_across_clades[0][np.where(alleles_across_clades[2] == 1)]
                unique_clade = alleles_across_clades[1][np.where(alleles_across_clades[2] == 1)]
                
                for nt in range(len(unique_allele)):
                    candidate_cssnps[p,unique_clade[nt]] = unique_allele[nt]
        
        #unique check
        a = np.arange(0, np.size(maNT,1))
        unclassified = a[~np.in1d(a,np.concatenate(clades))] #get unclassified samples
        
        for c in range(len(clades)):
            is_unique = np.sum(np.expand_dims(candidate_cssnps[:,c],1) == np.delete(core_genome,np.concatenate((clades[c], unclassified)),1), 1) > 0  #i think boolean mask would be nominally faster??
            candidate_cssnps[is_unique,c] = 0

        return candidate_cssnps        
    
    UnanimousCladeAlleles = unanimous_to_clade()
    CSS_matrix = unique_to_clade(UnanimousCladeAlleles)
    
    #temporary until i can standardize input
    if len(np.shape(core_pos)) > 1:
        core_pos = np.squeeze(core_pos)
        
    #save only positions with a cssnp allele
    cssnps = CSS_matrix[np.count_nonzero(CSS_matrix,1) > 0]
    cssnp_pos = core_pos[np.count_nonzero(CSS_matrix,1) > 0] 
    
    return cssnps, cssnp_pos

def combine_classifier_positions(cfrs,output_allpos_file,output_chrpos_file,refgenome_folder):
    
    all_pos = np.array([], dtype=np.int32)
    chr_starts, genome_length, scaf_names = genomestats(refgenome_folder)
    
    if len(cfrs) == 1:
        for c in list(cfrs):
            for filename in os.listdir(c):
                if filename.endswith('.classifier'):
                    with open(c+'/'+filename,'rb') as f:
                        csSNPs = pickle.load(f)
                    if len(csSNPs) != 3:
                        raise Exception('csSNP object is not correct shape!')
                    f_pos = csSNPs[1]
                    all_pos = np.unique(np.concatenate([f_pos,all_pos]))
                    
                else:
                    print('Warning! File '+filename+' does not have .classifier ending')
    if len(cfrs) > 1:
        print("Sorry, don't support multiple classifiers per run yet")

    all_pos.sort()
    chr_pos = p2chrpos(all_pos,chr_starts)
    print(str(len(all_pos)) +' total positions found across ' + str(len(os.listdir(list(cfrs)[0]))) + ' classifier(s)')
    
    #save as lists
    np.savetxt(output_allpos_file, all_pos, fmt='%i')
    np.savetxt(output_chrpos_file, chr_pos, fmt='%i')
    
    return
