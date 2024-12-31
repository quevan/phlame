#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:14:06 2022

@author: evanqu
"""

import numpy as np
import sys
import argparse
import gzip
import pickle

#%%
# Truncated version of larger diversity matrix that only saves 10 fields:
    # counts and indel information for each position on the reference genome
# [0-3] A is the number of forward reads supporting A
# [4-7] a is the number of reverse reads supporting A
# [8] I is number of reads supporting insertions in the +/- (indelregion) bp region
# [9] D is number of reads supporting deletions in the +/- (indelregion) bp region


#%%
def pileup2diversity(input_pileup, path_to_ref):
    """Grabs relevant allele info from mpileupfile and stores as a nice array
    This version ONLY records counts and indel_counter

    Args:
        input_pileup (str): Path to input pileup file.
        path_to_ref (str): Path to reference genome file
        
    """
    #parameters
    nts = 'ATCGatcg'
    nts_dict = {'A':0,'T':1,'C':2,'G':3,'a':4,'t':5,'c':6,'g':7}
    num_fields=10
    indelregion=3 #region surrounding each p where indels recorded 
    #get reference genome + position information
    chr_starts,genome_length,scaf_names = genomestats(path_to_ref)
    
    #initialize output array
    data = np.zeros((genome_length,num_fields)) #format [[A T C G  a t c g],[...]]
    
    #read in mpileup file
    print(f"Reading input file: {input_pileup}")
    mpileup = open(input_pileup)
    
    #####
    loading_bar=0
    
    for line in mpileup:
        
        loading_bar+=1
        if loading_bar % 50000 == 0:
            print('.')
        
        lineinfo = line.strip().split('\t')
        
        #holds info for each position before storing in data
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
        ref=nts_dict[lineinfo[2]] # convert to 0123
        if ref > 4:
            ref = ref - 4
        
        #calls info
        #calls=lineinfo[4]
        calls=np.fromstring(lineinfo[4], dtype=np.int8) #to ASCII
        # spits out a warning
        # calls=np.array([ord(l) for l in lineinfo[4]]) #ASCII
        
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
            #record that indel was found in +/- indelregion nearby
            #indexing is slightly different here from matlab version
            if calls[k]==45: #deletion
                if (position-indelregion-1 >= 0) and (position+indelsize+indelregion-1 < genome_length):
                    #must store directly into data as it affects lines earlier and later
                    data[position-indelregion-1:position+indelsize+indelregion-1,9]+=1
                elif position-indelregion >= 0: #for indels at end
                    data[position-indelregion-1:,9]+=1
                else: #for indels at beg
                    data[:position+indelsize+indelregion-1,9]+=1
            else: #insertion
                #insertion isn't indexed on the chromosome, no need for complex stuff
                if (position-indelregion-1 >= 0) and (position+indelregion-1 < genome_length):
                    data[position-indelregion-1:position+indelregion-1,8]+=1
                elif position-indelregion >= 0:
                    data[position-indelregion-1:,8]+=1
                else:
                    data[:position+indelsize-1,8]+=1

            #remove indel info from counting
            calls[k:(k+1+indeld+indelsize)] = -1 #don't remove base that precedes an indel
        
        #replace reference matches (.,) with their actual calls
        if ref >=0:
            calls[np.where(calls==46)[0]]=ord(nts[ref]) #'.'
            calls[np.where(calls==44)[0]]=ord(nts[ref+4]) #','

        #index reads for finding scores
        simplecalls=calls[np.where(calls>0)[0]]
        #simplecalls is a tform of calls where each calls position
        #corresponds to its position in bq, mq, td
        
        #count how many of each nt and average scores

        for nt in range(8):
            nt_count=np.count_nonzero(simplecalls == ord(nts[nt]))
            if nt_count > 0:
                temp[nt]=nt_count
        
        #-1 is needed to turn 1-indexed mpileup to 0-indexed arr
        data[position-1,0:8]=temp[0:8]
        
    #######
    mpileup.close()
    
    #calc coverage
    coverage=np.sum(data[:,0:8],1)
    
    return data, coverage

#%%
if __name__ == "__main__":
    
    SCRIPTS_DIR="scripts"
    sys.path.insert(0, SCRIPTS_DIR)
    
    from ss_caller_module import genomestats
    
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