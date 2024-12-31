#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:33:30 2021

@author: evanqu
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import numpy as np
import pickle
import glob
import gzip
import argparse

def read_fasta(REFGENOMEFOLDER): 
    
    fasta_file = glob.glob(REFGENOMEFOLDER + '/genome.fasta')
    if len(fasta_file) != 1:
        fasta_file_gz = glob.glob(REFGENOMEFOLDER + '/genome.fasta.gz')
        if len(fasta_file_gz) != 1:
            raise ValueError('Either no genome.fasta(.gz) or more than 1 genome.fasta(.gz) file found in ' + REFGENOMEFOLDER)
        else: # genome.fasta.gz
            refgenome = SeqIO.parse(gzip.open(fasta_file_gz[0], "rt"),'fasta')
    else: # genome.fasta
        refgenome = SeqIO.parse(fasta_file[0],'fasta')
    
    return refgenome

def combine_classifier_positions(cfr,output_path,savefile=False):
    
    if len(cfr) == 1:
        all_pos = np.array([], dtype=np.int32)
        for filename in os.listdir(cfr.pop()):
            if filename.endswith('.classifier'):
                with open(cfr.pop()+'/'+filename,'rb') as f:
                    csSNPs = pickle.load(f)
                if len(csSNPs) != 3:
                    raise Exception('csSNP object is not correct shape!')
                f_pos = csSNPs[1]
                all_pos = np.unique(np.concatenate([f_pos,all_pos]))
                
            else:
                print('Warning! File '+filename+' does not have .classifier ending')
    if len(cfr) > 1:
        print("Sorry, don't support multiple classifiers per run yet")
    
    all_pos.sort()
    print(str(len(all_pos))+' total positions')
    if savefile:
        np.savetxt(output_path, all_pos, fmt='%i')
        return
    
    return all_pos

def genomestats(REFGENOMEFOLDER):
    # parse ref genome to extract relevant stats
    # accepts genome.fasta or genome.fasta.gz (gzip) in refgenomefolder
    refgenome = read_fasta(REFGENOMEFOLDER)
    
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
    return [ChrStarts,Genomelength,ScafNames]
    
    
def p2chrpos(p, ChrStarts):
    '''# return 2col array with chr and pos on chr
    #p...continous, ignores chr
    #chrpos: like p, 0-based'''
        
    # get chr and pos-on-chr
    chromo = np.ones(len(p),dtype=int)
    if len(ChrStarts) > 1:
        for i in ChrStarts[1:]:
            chromo = chromo + (p > i) # when (p > i) evaluates 'true' lead to plus 1 in summation. > bcs ChrStarts start with 0...genomestats()
        positions = p - ChrStarts[chromo-1] # [chr-1] -1 due to 0based index
        chrpos = np.column_stack((chromo,positions))
    else:
        chrpos = np.column_stack((chromo,p))
    return chrpos

###
REFGENOMEFOLDER = 'Pacnes_C1'
all_pos_file = 'allpositions.txt'
###
def trim_ref_genome(REFGENOMEFOLDER, all_pos_file, trim_fasta_file, new_all_pos_file, new_chr_pos_file, readlength=150):
    
    all_pos = np.loadtxt(all_pos_file, dtype=np.int32)
    [ChrStarts, GenomeLength, ScafNames] = genomestats(REFGENOMEFOLDER)    
    chrpos = p2chrpos(all_pos,ChrStarts) #1-indexed
    
    pos2grab = []; chr2grab = []
    #for every position, grab neighboring positions
    #any way to make this faster?
    for chr_num, p in chrpos:
        start_p = p-readlength
        stop_p = p+readlength
        if start_p < 1:
            start_p=1
        if stop_p > np.max(chrpos[chrpos[:,0]==chr_num][:,1]):
            stop_p = np.max(chrpos[chrpos[:,0]==chr_num][:,1])
        chr_neighbors = [chr_num] * (stop_p-start_p)
        pos_neighbors = list(range(start_p, stop_p,1))
        chr2grab.append(chr_neighbors)
        pos2grab.append(pos_neighbors)
        
    #combine into one big array, 0-indexed
    pos_array = np.vstack((np.concatenate(chr2grab),np.concatenate(pos2grab))).T
    pos_nooverlaps = np.unique(pos_array,axis=0) #remove overlaps
    del pos_array
    #new cssnp positions in trimmed fasta 

    cssnp_chr = np.isin(pos_nooverlaps[:,0],chrpos[:,0]) 
    cssnp_pos = np.isin(pos_nooverlaps[:,1],chrpos[:,1]-1) #-1 for change in index
    
    #+1 to make it 1-indexed (because samtools)
    updated_all_pos = np.where(np.all(np.vstack((cssnp_chr,cssnp_pos)).T, axis=1)==True)[0]+1
    # add scaf names for samtools mpileup
    updated_chr_pos = np.array([ScafNames[i-1] for i in chrpos[:,0]])
    
    #load fasta
    refgenome = read_fasta(REFGENOMEFOLDER)
    
    #for each contig, grab basecalls at pos_nooverlaps
    to_write = []
    for idx, record in enumerate(refgenome):
        if idx+1 in np.unique(pos_nooverlaps[:,0]):
            print('test')
            chr_pos2grab = pos_nooverlaps[pos_nooverlaps[:,0]==idx+1][:,1]
            chr_basecalls = "".join(record.seq[int(p-1)] for p in chr_pos2grab) # Seq obj is 0 indexed

            trim_record = SeqRecord(Seq(chr_basecalls), id=record.id, name=record.name, description = record.description)
            to_write.append(trim_record)
    
    #write trimmed genome to fasta file
    SeqIO.write(to_write,trim_fasta_file,format='fasta')
    #write all_positions (for  to text file
    np.savetxt(new_all_pos_file,updated_all_pos,delimiter=',',fmt='%i')
    #write chrpos for samtools mpileup
    np.savetxt(new_chr_pos_file, np.vstack((updated_chr_pos,updated_all_pos)).T, delimiter='\t', fmt="%s")
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='InputFasta', type=str, help='Path to reference genome fasta',required=True)
    parser.add_argument('-p', dest='AllPos', type=str, help='Path to positions file',required=True)
    parser.add_argument('-a', dest='NewAllPos', type=str, help='Path to output chr_pos', required=True)
    parser.add_argument('-c', dest='ChrPos', type=str, help='Path to output chr_pos', required=True)
    parser.add_argument('-o', dest='Output', type=str, help='Path to output trimmed fasta file', required=True)
    parser.add_argument('-r', dest='ReadLength', type=int, help='Read Length')
    args = parser.parse_args()
    #REFGENOMEFOLDER, all_pos_file, trim_fasta_file, new_all_pos_file, new_chr_pos_file readlength=150
    if args.ReadLength:
        trim_ref_genome(args.InputFasta, args.AllPos, args.Output, args.NewAllPos, args.ChrPos, args.ReadLength)
    else:
        trim_ref_genome(args.InputFasta, args.AllPos, args.Output, args.NewAllPos, args.ChrPos)