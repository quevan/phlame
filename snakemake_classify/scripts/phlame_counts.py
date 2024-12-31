#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:14:06 2022

@author: evanqu
"""

import numpy as np
import os
import glob
import argparse
import gzip
import pickle
from Bio import SeqIO

#%% Test
# os.chdir("/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/strainslicer/dev/phlame_counts")
# input_pileup = '10X_cacnes_benchmark_s4_ref_Pacnes_C1_aligned.sorted.pileup'
# path_to_ref='Pacnes_C1'
# path_to_classifiers = 'Cacnes_ALL_newClassifier_112922.classifier'
# counts, pos = pileup2counts(input_pileup, path_to_ref)
    
# # Check
# output_diversity = 'KIT_1C_MG1_E2_ref_SepidermidisATCC12228.diversity.pickle.gz'
# with gzip.open(output_diversity, 'rb') as f:
#     truediv = pickle.load(f)
# truecounts = truediv[ pos - 1]

# (counts == truecounts).all()


#%% Functions

def pileup2counts(input_pileup, path_to_ref, path_to_classifiers):
    '''Grabs relevant allele info from mpileupfile and stores as a nice array.

    Args:
        input_pileup (str): Path to input pileup file.
        path_to_ref (str): Path to reference genome file.
        path_to_classifiers (str): Path to classifier file(s).
        
    '''
    # Initial parameters
    nts = 'ATCGatcg'
    num_fields = 8
    
    # Get reference genome information
    chr_starts,genome_length,scaf_names = genomestats(path_to_ref)
    
    # Get classifier position information
    allpos = classifier_stats(path_to_classifiers)
    
    data = np.zeros((len(allpos),num_fields)) #format [[A T C G  a t c g],[...]]
    
    ##### read in mpileup file #####
    mpileup = open(input_pileup)

    for line in mpileup:
        
        lineinfo = line.strip().split('\t')
        
        #holds info for each line before storing in data
        temp = np.zeros((num_fields))
        
        chromo = lineinfo[0]
        
        #position (absolute)
        if chromo not in scaf_names:
            raise ValueError("Contig name in pileup file not found in reference!")

        if len(chr_starts) == 1:
            position=int(lineinfo[1])
        else:
            position=int(chr_starts[np.where(chromo==scaf_names)]) + int(lineinfo[1])
            #chr_starts begins at 0
        pidx = np.searchsorted(allpos, position) # index of position on allpos
        
        #ref allele
        ref=int(np.char.find(nts,lineinfo[2])) # convert to 0123
        if ref > 4:
            ref = ref - 4
        
        #calls info
        #calls=lineinfo[4]
        calls=np.array([ord(l) for l in lineinfo[4]]) #ASCII
        
        #find starts of reads ('^' in mpileup)
        startsk=np.where(calls==94)[0]
        for k in startsk:
            calls[k:k+2]=-1 #WHAT IS -1
            #remove mapping character, absolutely required because the next chracter could be $
        
        #find ends of reads ('$' in mpileup)
        endsk=np.where(calls==36)[0]
        calls[endsk]=-1
        
        #find indels + calls from reads supporting indels ('+-')
        indelk = np.where((calls==43) | (calls==45))[0]
        for k in indelk:
            if (calls[k+2] >=48) and (calls[k+2] < 58): #2 digit indel (size > 9 and < 100)
                indelsize=int(chr(calls[k+1]) + chr(calls[k+2])) 
                #indelsize=str2double(char(calls(k+1:k+2))); MATLAB
                indeld=2
            else: #1 digit indel (size <= 9)
                indelsize=int(chr(calls[k+1]))
                indeld=1
        #remove indel info from counting
            calls[k:(k+1+indeld+indelsize)] = -1 #don't remove base that precedes an indel
        
        #replace reference matches (.,) with their actual calls
        if ref >=0:
            calls[np.where(calls==46)[0]]=ord(nts[ref]) #'.'
            calls[np.where(calls==44)[0]]=ord(nts[ref+4]) #','
        # if ref >=0:
        #     calls[np.where(calls==46)[0]]=ord(nts[int(np.char.find(nts,ref))]) # changed from nts(ref); matlab
        #     calls[np.where(calls==44)[0]]=ord(nts[int(np.char.find(nts,ref))+4]) # changed from=nts(ref+4); matlab

        #index reads for finding scores
        simplecalls=calls[np.where(calls>0)[0]]
        #simplecalls is a tform of calls where each calls position
        #corresponds to its position in bq, mq, td
        
        #count how many of each nt and average scores
        for nt in range(8):
            if not sum(simplecalls == ord(nts[nt])) == 0:
                temp[nt]=sum(simplecalls == ord(nts[nt]))
        
        # Store in big array
        data[pidx]=temp
    
    return data, allpos


def read_fasta(path_to_refgenome): 
    '''Reads in fasta file. If directory is given, reads in dir/genome.fasta
    Args:
        path_to_refgenome (str): Path to reference genome.

    Returns: SeqIO object for reference genome.
    '''
    fasta_file = glob.glob(path_to_refgenome + '/genome.fasta')
    if len(fasta_file) != 1:
        fasta_file_gz = glob.glob(path_to_refgenome + '/genome.fasta.gz')
        if len(fasta_file_gz) != 1:
            raise ValueError('Either no genome.fasta(.gz) or more than 1 genome.fasta(.gz) file found in ' + path_to_refgenome)
        else: # genome.fasta.gz
            refgenome = SeqIO.parse(gzip.open(fasta_file_gz[0], "rt"),'fasta')
    else: # genome.fasta
        refgenome = SeqIO.parse(fasta_file[0],'fasta')
    
    return refgenome

def genomestats(path_to_refgenome):
    '''Parse genome to extract relevant stats

    Args:
        REFGENOMEFOLDER (str): Path to reference genome.

    Returns:
        ChrStarts (arr): DESCRIPTION.
        Genomelength (arr): DESCRIPTION.
        ScafNames (arr): DESCRIPTION.

    '''

    refgenome = read_fasta(path_to_refgenome)
    
    Genomelength = 0
    ChrStarts = []
    ScafNames = []
    for record in refgenome:
        ChrStarts.append(Genomelength) # chr1 starts at 0 in analysis.m
        Genomelength = Genomelength + len(record)
        ScafNames.append(record.id)
    # close file
    #refgenome.close() # biopy update SeqIO has no close attribute anymore.
    # turn to np.arrys!
    ChrStarts = np.asarray(ChrStarts,dtype=int)
    Genomelength = np.asarray(Genomelength,dtype=int)
    ScafNames = np.asarray(ScafNames,dtype=object)
    
    return ChrStarts,Genomelength,ScafNames

def classifier_stats(path_to_classifiers):
    '''Parse classifier file(s) to extract position information

    Args:
        path_to_classifiers (str): Comma separated string of paths to classifiers.

    Returns:
        allpos (arr): Array of absolute positions referenced in classifier(s).

    '''
    
    cat_pos = np.array([], dtype=np.int32)
 
    path_to_cfrs_ls = []
    # Parse whether file or directory of files
    if os.path.isdir(path_to_classifiers):
        
        for filename in os.listdir(path_to_classifiers):
            
            if filename.endswith('.classifier'):
                path_to_cfrs_ls.append(path_to_classifiers+'/'+filename)
    else:
        path_to_cfrs_ls.append(path_to_classifiers)

    for cfr in path_to_cfrs_ls:
        
        with gzip.open(cfr,'rb') as f:
            csSNPs = pickle.load(f)
                            
            pos = csSNPs['cssnp_pos']
            cat_pos = np.concatenate([pos,cat_pos])

    allpos = np.unique(np.sort(cat_pos))
 
    return allpos
    
def chrpos_stats(path_to_pos_file):
    
    genome_length = 0
    scaf_names_ls = []
    chr_pos = []

    with open(path_to_pos_file,'r') as file:
        
        for line in file:
             
            position = line.strip().split('\t')
            scaf_names_ls.append(position[0])
            chr_pos.append(position[1])
    
    scaf_names, lengths = np.unique(scaf_names_ls,return_counts=True)
    chr_starts=[]

    if len(scaf_names) == 1:
        chr_starts.append(genome_length)
        genome_length = lengths[0]
    else:
        for scaf, len_ in zip(scaf_names,lengths):
            chr_starts.append(genome_length)            
            genome_length = genome_length + len_
    
    return chr_starts, genome_length, scaf_names

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', dest='input', type=str, help='Path to input pileup',required=True)
    parser.add_argument('-r', dest='ref', type=str, help='Path to reference genome',required=False)
    parser.add_argument('-o', dest='output', type=str, help='Path to output diversity file', required=True)
    parser.add_argument('-c', dest='classifier', type=str, help='Path to output diversity file', required=True)

    args = parser.parse_args()
            
    counts, pos = pileup2counts(args.input,
                                args.ref, 
                                args.classifier)
            
    with gzip.open(args.output, 'wb') as f:
        pickle.dump([counts, pos],f)