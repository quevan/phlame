#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:33:43 2022

@author: evanqu
"""

import pickle
import gzip
import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio import SeqIO

# # To do: format --> functions generally not looked at / could be optimized/pruned

#Todo: add formal IO errors
def read_cmt(path_to_cmt_file, version='old'):
    '''Read in candidate mutation table from pickled object file.

    Args:
        path_to_cmt_file (str): String path to candidate mutation table file.

    Returns:
        sample_names (arr): Array of sample names.
        pos (TYPE): DESCRIPTION.
        counts (TYPE): DESCRIPTION.
        quals (TYPE): DESCRIPTION.
        indel_counter (TYPE): DESCRIPTION.

    '''
    
    if path_to_cmt_file.endswith('.pickle.gz'):
        with gzip.open(path_to_cmt_file,'rb') as f:
            CMT = pickle.load(f)
    
    elif path_to_cmt_file.endswith('.pickle'):
        with open(path_to_cmt_file,'rb') as f:
            CMT = pickle.load(f)

            
    if version=='new':
        counts = CMT['counts']
        sample_names = CMT['sample_names']
        pos = CMT['p']
        quals=CMT['quals']
        indel_counter=CMT['indel_counter']
        in_outgroup=CMT['in_outgroup']

    elif version=='old':
        counts = CMT['counts']
        sample_names = CMT['sample_names']
        pos = CMT['p']
        quals=CMT['quals']
        indel_counter=CMT['indel_counter']
        in_outgroup=False            

    if np.size(counts,axis=0) != 8:
        print('Transposing counts table...')
        counts = counts.transpose(1,2,0)
        
    return np.array(sample_names), pos, counts, quals, indel_counter, in_outgroup

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

def genomestats(path_to_refgenome_file):
    '''Extract relevant stats from a reference genome file.

    Args:
        path_to_refgenome_file (str): Path to reference genome file (.fasta).

    Returns:
        ChrStarts (TYPE): DESCRIPTION.
        Genomelength (TYPE): DESCRIPTION.
        ScafNames (TYPE): DESCRIPTION.

    '''
    
    refgenome = SeqIO.parse(path_to_refgenome_file,'fasta')
    
    Genomelength = 0
    ChrStarts = []
    ScafNames = []
    
    for record in refgenome:
        ChrStarts.append(Genomelength) # chr1 starts at 0 in analysis.m
        Genomelength = Genomelength + len(record)
        ScafNames.append(record.id)
    
    # turn to np.arrys
    ChrStarts = np.asarray(ChrStarts,dtype=int)
    Genomelength = np.asarray(Genomelength,dtype=int)
    ScafNames = np.asarray(ScafNames,dtype=object)
    
    return ChrStarts,Genomelength,ScafNames


def mant(counts):
    '''Get major and first minor allele along with frequencies for each position in a counts matrix. 

    Args:
        counts (arr): numpy-compatible array (8xpxs).

    Returns:
        maNT (TYPE): DESCRIPTION.
        maf (TYPE): DESCRIPTION.
        minorNT (TYPE): DESCRIPTION.
        minorAF (TYPE): DESCRIPTION.

    '''
    
    c=counts[0:4,:,:]+counts[4:8,:,:] # combine f and r ATCG counts

    sorted_c = np.sort(c,axis=0) # sort by num. ATCGs 
    argsort_c = np.argsort(c,axis=0)
    
    # Get allele counts for major allele (4th row)
    # Weird "3:4:" indexing required to maintain 3D structure
    maxcount = sorted_c[3:4:,:,:] 
    # Get allele counts for first minor allele (3rd row)
    # tri/quadro-allelic ignored!!
    minorcount = sorted_c[2:3:,:,:] 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        
        maf = maxcount / sorted_c.sum(axis=0,keepdims=True)
        minorAF = minorcount / sorted_c.sum(axis=0,keepdims=True)
    
    # turn 2D; axis=1 to keep 2d structure when only one position!
    maf = np.squeeze(maf,axis=0) 
    maf[np.isnan(maf)]=0 # set to 0 to indicate no data
    
    minorAF = np.squeeze(minorAF,axis=0) 
    minorAF[np.isnan(minorAF)]=0 # set to 0 to indicate no data/no minor AF
    
    # Idx given by argsort_c represents allele position ATCG
    # A=0,T=1,C=2,G=3
    # axis=1 to keep 2d structure when only one position!
    maNT = np.squeeze(argsort_c[3:4:,:,:],axis=0) 
    minorNT = np.squeeze(argsort_c[2:3:,:,:],axis=0)

    # Note: If counts for all bases are zero, then sort won't change the order
    # (since there is nothing to sort), thus maNT/minorNT will be put to -1 (NA)
    # using maf (REMEMBER: minorAF==0 is a value!)
    maNT[maf==0]=-1
    minorNT[maf==0]=-1
    
    # MATLAB conversion to NATCG=01234
    # !Important! This is required for current ver of find_clade_specific_snps
    # as of 3/26/22; Want to change later
    maNT=maNT+1
    minorNT=minorNT+1
    
    return maNT, maf, minorNT, minorAF

#To do: simplify into numpy array
#Resolve whether I need sample_names or not
def distmat(calls, sample_names):
    ''' Calculate the pairwise SNP distance of all samples in a maNT matrix.

    Args:
        calls (arr): Matrix of major allele NT for each sample.
        sample_names (ls): List of sample names.

    Returns:
        distmat (arr): Matrix of pairwise distances.

    '''
    num_samples=len(sample_names)
    
    distmat = np.zeros((num_samples,num_samples))
    
    for i in range(num_samples):
        # print(f"Sample progress: {i+1}/{num_samples} samples done ")
        distmat[i,:] = np.count_nonzero( (calls != np.tile(calls[:,i],(num_samples,1)).T) &
                               (calls > 0) &
                               (np.tile(calls[:,i],(num_samples,1)).T > 0) , axis=0)

    distmat_df = pd.DataFrame(distmat, 
                          index=sample_names, 
                          columns=sample_names)
    
    return distmat_df


def p2chrpos(p, ChrStarts):
    '''# return 2col array with chr and pos on chr
    #p --> continous, ignores chr
    #pos --> like p, 0-based'''

    # get chr and pos-on-chr
    chr = np.ones(len(p),dtype=int)
    if len(ChrStarts) > 1:
        for i in ChrStarts[1:]:
            chr = chr + (p > i) # when (p > i) evaluates 'true' lead to plus 1 in summation. > bcs ChrStarts start with 0...genomestats()
        positions = p - ChrStarts[chr-1] # [chr-1] -1 due to 0based index
        pos = np.column_stack((chr,positions))
    else:
        pos = np.column_stack((chr,p))
    return pos

# To do: format
def idx2nts(calls, missingdata="?"):
    # translate index array to array containing nucleotides
    # add 5th element --> no data! == index -1
    nucl = np.array([missingdata,'A','T','C','G'],dtype=object) 
    palette = [-1,0,1,2,3] # values present in index-array
    index = np.digitize(calls.ravel(), palette, right=True)
    
    return nucl[index].reshape(calls.shape)

# To do: format
def write_calls_to_fasta(calls,sample_names,output_file):
    
    fa_file = open(output_file, "w")
    
    for i,name in enumerate(sample_names):
        nucl_string = "".join(list(calls[:,i]))
        fa_file.write(">" + name + "\n" + nucl_string + "\n")
    
    fa_file.close()    
