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
from Bio import SeqIO
import glob

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


# class CountsMat():
#     '''
#     Holds data and methods for a counts matrix.
#     '''
#     def __init__(self, path_to_cts_file):
        
#         with gzip.open(path_to_cts_file,'rb') as f:
#             counts, pos = pickle.load(f)
                
#         self.counts = counts
#         self.pos = pos


class CountsMat():
    '''
    Holds data and methods for a counts matrix.
    '''

    def __init__(self, 
                 path_to_pileup, 
                 path_to_ref, 
                 path_to_classifiers):
        
        self.path_to_pileup = path_to_pileup
        self.path_to_ref = path_to_ref
        self.path_to_classifiers = path_to_classifiers

        self.chr_starts, self.genome_length, self.scaf_names = genomestats(self.path_to_ref)

    def readin(self,
               path_to_cts_file):
        '''
        Read in an existing counts object.
        '''
        
        with gzip.open(path_to_cts_file,'rb') as f:
            counts, pos = pickle.load(f)
                
        self.counts = counts
        self.pos = pos

    def main(self):
        
        print(f"Reading pileup file: {self.path_to_pileup}")
        counts, pos = self.pileup2counts(self.path_to_pileup,
                                         self.path_to_classifiers)
        
        self.counts = counts
        self.pos = pos

    def pileup2counts(self,
                      input_pileup,
                      path_to_classifiers):
        '''Grabs relevant allele info from mpileupfile and stores as a nice array.

        Args:
            input_pileup (str): Path to input pileup file.
            path_to_ref (str): Path to reference genome file.
            path_to_classifiers (str): Path to classifier file(s).
            
        '''
        # Initial parameters
        nts = 'ATCGatcg'
        num_fields = 8
        
        # Get classifier position information
        allpos = self.classifier_stats(path_to_classifiers)
        
        data = np.zeros((len(allpos),num_fields)) #format [[A T C G  a t c g],[...]]
        
        ##### read in mpileup file #####
        mpileup = open(input_pileup)

        loading_bar = 0
        for line in mpileup:
            
            # loading_bar+=1
            # if loading_bar % 500 == 0:
            #     print('.')

            lineinfo = line.strip().split('\t')
            
            #holds info for each line before storing in data
            temp = np.zeros((num_fields))
            
            chromo = lineinfo[0]
            
            #position (absolute)
            if chromo not in self.scaf_names:
                raise ValueError("Contig name in pileup file not found in reference!")

            if len(self.chr_starts) == 1:
                position=int(lineinfo[1])
            else:
                position=int(self.chr_starts[np.where(chromo==self.scaf_names)]) + int(lineinfo[1])
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
                nt_count=np.count_nonzero(simplecalls == ord(nts[nt]))
                if nt_count > 0:
                    temp[nt]=nt_count
            # for nt in range(8):
            #     if not sum(simplecalls == ord(nts[nt])) == 0:
            #         temp[nt]=sum(simplecalls == ord(nts[nt]))

            # Store in big array
            data[pidx]=temp
        
        return data, allpos

    @staticmethod
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
    
    @staticmethod
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

    @staticmethod        
    def chrpos_stats(self, path_to_pos_file):
        
        self.genome_length = 0
        self.scaf_names_ls = []
        chr_pos = []

        with open(path_to_pos_file,'r') as file:
            
            for line in file:
                
                position = line.strip().split('\t')
                self.scaf_names_ls.append(position[0])
                chr_pos.append(position[1])
        
        self.scaf_names, lengths = np.unique(self.scaf_names_ls,return_counts=True)
        self.chr_starts=[]

        if len(self.scaf_names) == 1:
            self.chr_starts.append(self.genome_length)
            self.genome_length = lengths[0]
        else:
            for scaf, len_ in zip(self.scaf_names,lengths):
                self.chr_starts.append(self.genome_length)            
                self.genome_length = self.genome_length + len_
        
        return self.chr_starts, self.genome_length, self.scaf_names


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

def read_fasta(reference_genome_file): 
    '''Reads in fasta file. If directory is given, reads in dir/genome.fasta
    Args:
        reference_genome_file (str): Path to reference genome.

    Returns: SeqIO object for reference genome.
    '''

    if os.path.exists(reference_genome_file):
        refgenome = SeqIO.parse(reference_genome_file,'fasta')

        return refgenome

    fasta_file = glob.glob(reference_genome_file + '/genome.fasta')
    if len(fasta_file) != 1:
        fasta_file_gz = glob.glob(reference_genome_file + '/genome.fasta.gz')
        if len(fasta_file_gz) != 1:
            raise ValueError('Either no genome.fasta(.gz) or more than 1 genome.fasta(.gz) file found in ' + reference_genome_file)
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

def p2chrpos(p, ChrStarts):
    '''Convert 1col list of pos to 2col array with chromosome and pos on chromosome

    Args:
        p (TYPE): DESCRIPTION.
        ChrStarts (TYPE): DESCRIPTION.

    Returns:
        chrpos (TYPE): DESCRIPTION.

    '''
        
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

def parse_file_list(path_to_file):
    with open(path_to_file, 'r') as f:
        file_list = f.read().splitlines()

    for file_ in file_list:
        if not os.path.isabs(file_):
            abs_path = os.path.abspath(file_)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"File not found: {abs_path}")
            file_list[file_list.index(file_)] = os.path.abspath(file_)

    return file_list
