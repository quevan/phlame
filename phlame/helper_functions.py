#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions and classes for phlame.

@author: evanqu
"""

import os
import pickle
import gzip
import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio import SeqIO


class Frequencies():
    '''
    Holds clade frequency information from a given sample.
    '''
    
    def __init__(self, path_to_frequencies_file):
        
        self.freqs = pd.read_csv(path_to_frequencies_file,
                                       index_col=0)
                
class FrequenciesData():
    '''
    Holds clade specific SNV counts and modeling information from a given sample.
    '''

    def __init__(self, path_to_data_file):
        
        with gzip.open(path_to_data_file, 'rb') as f:
            
            data_dct, fit_info_dct = pickle.load(f)
            
            # clade_counts structured as follows
            
            self.clade_counts = data_dct['clade_counts']
            self.clade_counts_pos = data_dct['clade_counts_pos']
            
            self.counts_MLE = fit_info_dct['counts_MLE']
            self.total_MLE = fit_info_dct['total_MLE']
            self.counts_MAP = fit_info_dct['counts_MAP']
            self.chain = fit_info_dct['chain']
            self.prob = fit_info_dct['prob']


class CountsMat():
    '''
    Holds data and methods for a counts matrix.
    '''
    def __init__(self, path_to_cts_file):
        
        with gzip.open(path_to_cts_file,'rb') as f:
            counts, pos = pickle.load(f)
                
        self.counts = counts
        self.pos = pos

class PhlameClassifier():
    '''
    Holds data and methods for a single Phlame Classifier object
    '''
    def __init__(self,
                 csSNPs, csSNP_pos,
                 clades, clade_names):

        self.csSNPs = csSNPs
        self.csSNP_pos = csSNP_pos
        self.clades = clades
        self.clade_names = clade_names
        
        # Get allele information
        self.get_alleles()
    
    def read_file(path_to_classifier_file):

        with gzip.open(path_to_classifier_file, 'rb') as f:
            cssnp_dct = pickle.load(f)
            
            csSNPs = cssnp_dct['cssnps']
            csSNP_pos = cssnp_dct['cssnp_pos']
            clades = cssnp_dct['clades']
            clade_names = cssnp_dct['clade_names']
            
        return PhlameClassifier(csSNPs, csSNP_pos, clades, clade_names)

    def grab_level(self, PhyloLevel):
        '''
        Grab just information for a specific level.
        '''
        
        idx=[]
        
        for clade in PhyloLevel.clade_names:
            
            idx.append(np.where(self.clade_names==clade)[0][0])
        
        level_csSNPs = self.csSNPs[:,idx]

        level_csSNP_pos = self.csSNP_pos[~np.all(level_csSNPs == 0, axis=1)]
        
        return PhlameClassifier(level_csSNPs[~np.all(level_csSNPs == 0, axis=1)],
                                level_csSNP_pos,
                                PhyloLevel.clades,
                                PhyloLevel.names)
    
    def get_alleles(self):
        '''
        Get 1D list of every allele and corresponding clade.
        '''
        self.alleles = self.csSNPs[np.nonzero(self.csSNPs)]
        # corresponding clade index
        self.allele_cidx = np.nonzero(self.csSNPs)[1]


class Phylip():
    '''
    Holds data and methods for a phylip object.
    '''

    def check_valid(phylip_file):

        if not os.path.exists(phylip_file):
            return False
        
        with open(phylip_file, 'r') as f:
            line1 = f.readline().strip().split(' ')

        nsamples = int(line1[0])
        npos = int(line1[1])

        valid_samples = nsamples > 0
        valid_positions = npos > 0

        return valid_samples and valid_positions

class CandidateMutationTable():

    def __init__(self, path_to_cmt_file):
        '''
        Read in candidate mutation table from pickled object file.
        '''
        
        if path_to_cmt_file.endswith('.pickle.gz'):
            with gzip.open(path_to_cmt_file,'rb') as f:
                CMT = pickle.load(f)
        
        elif path_to_cmt_file.endswith('.pickle'):
            with open(path_to_cmt_file,'rb') as f:
                CMT = pickle.load(f)

        self.sample_names = np.array(CMT['sample_names'])
        self.counts = CMT['counts']
        self.pos = CMT['p']
        self.quals=CMT['quals']
        self.indel_counter=CMT['indel_counter']

        # FIX THIS!!!
        self.indel_counter = np.zeros((2,len(self.pos),
                                        len(self.sample_names)))
            
        # Note that 1 -> yes outgroup, 0 -> not outgroup
        # Makes booleans more confusing I know
        if 'in_outgroup' in CMT.keys():
            self.in_outgroup=CMT['in_outgroup']
            if type(self.in_outgroup[0]) == np.ndarray:
                self.in_outgroup = np.array(self.in_outgroup[0][0].split(' ')).astype(bool)

        else:
            self.in_outgroup = np.array([False]*len(self.sample_names))

        # Calculate coverage and indels
        self.calc_coverage()
        self.calc_indels_all()

    def calc_coverage(self):
        '''
        Calculate coverage for each sample.
        '''
        self.coverage = np.sum(self.counts,axis=0)

    def calc_indels_all(self):

        self.indels_all = np.sum(self.indel_counter,axis=0)

    # if np.size(self.counts, axis=0) != 8:
    #     print('Transposing counts table...')
    #     self.counts = self.counts.transpose(1,2,0)
            

def read_clades_file(path_to_clades_file, uncl_marker):
    '''
    Read in a clades file.
    '''        
    
    if not os.path.exists(path_to_clades_file):
        raise FileNotFoundError(f'File {path_to_clades_file} not found.')
    # Get delimiter
    with open(path_to_clades_file,'r') as file:
        firstline = file.readline()
   
    if len(firstline.strip().split('\t'))==2:
        dlim='\t'
    elif len(firstline.strip().split(','))==2:
        dlim=','
    else:
        raise ValueError('Delimiter in clades file not recognized. Please specify clades as a tab or comma separated list.')
    
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
