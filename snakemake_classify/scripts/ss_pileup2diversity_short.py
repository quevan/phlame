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

#%%
def read_fasta(REFGENOME_DIR): 
    '''Reads in fasta file. If directory is given, reads in dir/genome.fasta
    Args:
        REFGENOME_DIR (str): Path to reference genome.

    Returns: SeqIO object for reference genome.
    '''
    fasta_file = glob.glob(REFGENOME_DIR + '/genome.fasta')
    if len(fasta_file) != 1:
        fasta_file_gz = glob.glob(REFGENOME_DIR + '/genome.fasta.gz')
        if len(fasta_file_gz) != 1:
            raise ValueError('Either no genome.fasta(.gz) or more than 1 genome.fasta(.gz) file found in ' + REFGENOME_DIR)
        else: # genome.fasta.gz
            refgenome = SeqIO.parse(gzip.open(fasta_file_gz[0], "rt"),'fasta')
    else: # genome.fasta
        refgenome = SeqIO.parse(fasta_file[0],'fasta')
    
    return refgenome

def genomestats(REFGENOME_DIR):
    '''Parse genome to extract relevant stats

    Args:
        REFGENOMEFOLDER (str): Directory containing reference genome file.

    Returns:
        ChrStarts (arr): DESCRIPTION.
        Genomelength (arr): DESCRIPTION.
        ScafNames (arr): DESCRIPTION.

    '''

    refgenome = read_fasta(REFGENOME_DIR)
    
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

#%%
# os.chdir("/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/strainslicer/dev/pileup2diversity_py_script")

# input_pileup = '8AB1K_25YH53_ref_Pacnes_C1_aligned.sorted.pileup'
# input_ref='Pacnes_C1'
# calls='........,+1t,+1t,+1t,+1t'
# calls_chr = [l for l in calls]
# calls=np.array([ord(l) for l in calls]) #ASCII

# startsk=np.where(calls==94)[0]
# for k in startsk:
#     calls[k:k+1]=-1
    
# indelk = np.where((calls==43) | (calls==45))[0]
# for k in indelk:
#     if (calls[k+2] >=48) and (calls[k+2] < 58): #2 digit indel (size > 9 and < 100)
#         indelsize=int(chr(calls[k+1]) + chr(calls[k+3])) 
#         #indelsize=str2double(char(calls(k+1:k+2))); MATLAB
#         indeld=2
#     else: #1 digit indel (size <= 9)
#         indelsize=int(chr(calls[k+1]))
#         indeld=1
#     #remove indel info from counting
#     calls[k:(k+1+indeld+indelsize)] = -1
    
# if ref:
#     calls[np.where(calls==46)[0]]=ord(ref) # '.'
#     calls[np.where(calls==44)[0]]=ord(ref) # ','


#%%
def pileup2diversity(input_pileup, path_to_ref):
    """Grabs relevant allele info from mpileupfile and stores as a nice array 

    Args:
        input_pileup (str): Path to input pileup file.
        path_to_ref (str): Path to reference genome file
        
    """
    #parameters
    nts = 'ATCGatcg'
    num_fields = 8    
    #get reference genome + position information
    chr_starts,genome_length,scaf_names = genomestats(path_to_ref)
    
    #init
    data = np.zeros((genome_length,num_fields)) #format [[A T C G  a t c g],[...]]
        
    #read in mpileup file
    mpileup = open(input_pileup)
    
    #####
    for line in mpileup:
        lineinfo = line.strip().split('\t')
        
        #holds info for each line before storing in data
        temp = np.zeros((num_fields))
        
        chromo = lineinfo[0]
        #position (absolute)
        if len(chr_starts) == 1:
            position=int(lineinfo[1])
        else:
            if chromo not in scaf_names:
                raise ValueError("Scaffold name in pileup file not found in reference")
            position=int(chr_starts[np.where(chromo==scaf_names)]) + int(lineinfo[1])
            #chr_starts begins at 0
        
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
        
        #-1 is needed to turn 1-indexed mpileup to 0-indexed arr
        data[position-1]=temp
        
    #######
    mpileup.close()
    
    #calc coverage
    coverage=np.sum(data,1)
    
    return data, coverage

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', dest='input', type=str, help='Path to input pileup',required=True)
    parser.add_argument('-r', dest='ref', type=str, help='Path to reference genome',required=True)
    parser.add_argument('-o', dest='output', type=str, help='Path to output diversity file', required=True)
    parser.add_argument('-c', dest='coverage', type=str, help='Path to coverage file', required=True)
    
    args = parser.parse_args()
    
    diversity_arr, coverage_arr = pileup2diversity(args.input,args.ref)
    
    with gzip.open(args.output, 'wb') as f:
        pickle.dump(diversity_arr,f)
    
    if args.coverage:
        with gzip.open(args.coverage, 'wb') as f:
            pickle.dump(coverage_arr,f)