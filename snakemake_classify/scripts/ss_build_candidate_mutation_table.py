# -*- coding: utf-8 -*-
"""
---Gathers everything together for candidate_mutation_table---
NOTE: Still reads in many *.mat files etc. Further purging of matlab necessary!

Output:
# path_candidate_mutation_table: where to write
# candidate_mutation_table.mat, ex. results/candidate_mutation_table.mat

---


# Inputs (changed to argparse usage):
     path_to_p_file: where to find all_positions.txt
     path_to_sample_names_file: where to find text file with sample names
         (space delimited)
     path_to_outgroup_boolean_file: where to find text file with outgroup
         booleans (space delimited, 1=outgroup, 0=not)
     path_to_list_of_quals_files: where to find text file with list of
       quals.mat files for each sample (space delimited)
     path_to_list_of_diversity_files: where to find text file with list of
       diversity.mat files for each sample (space delimited)
# Output:
     path_candidate_mutation_table: where to write
     candidate_mutation_table.pickle.gz

# Note: All paths should be relative to pwd!

## Version history

     This is adapted from TDL's build_mutation_table_master_smaller_file_size_backup.m
  #   Arolyn, 2018.12.19: This script was written as part of the transition to snakemake. 
          It performs the part of the case step that gathers data for
         Quals and counts and saves candidate_mutation_table.mat
  #   Arolyn, 2019.02.12: Added another matlab variable that stores indel
          statistics called 'indel_counter'.
  #   Tami, 2019.12.12: Converted into python and also added ability save coverage data
  #   Felix: 2020.01-04: Continous Debugged and adapted script for streamlined Snakemake implementation. 
  #                      Added argparse for proper argument parsing and optional coverage matrix build.
"""

''' load libraries '''
import numpy as np
import pickle
import scipy.io as sio
import os
import sys,argparse
import gzip
from scipy import sparse

''' positional and optional argument parser'''

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description='''\
                            Gathers everything together for candidate_mutation_table.
                            Optional: Builds coverage matrix (optional w/ double-standardized matrix)
                               ''',
                               epilog="Questions or comments? --> fkey@mit.edu")
parser.add_argument("-p", dest="allpositions", help="All positions p file (1-indexed)",required=True,action='store')
parser.add_argument("-s", dest="sampleNames", help="File with sample names",required=True,action='store')
parser.add_argument("-g", dest="outgroupBool", help="String outgroup bool",required=True,action='store')
#parser.add_argument("-q", dest="qualfiles", help="String qual matrix paths",required=True,action='store')
parser.add_argument("-d", dest="divfiles", help="String diversity paths",required=True,action='store')
parser.add_argument("-o", dest="candidate_mutation_table", help="Output candidate mutation table. Py pickle structure (*.pickle.gz)",required=True,action='store')
parser.add_argument("-c", dest="get_cov", help="Set flag to build raw coverage matrix as sparse csr gzip numpy object (dirname+cov_raw_sparsecsr_mat.npz)",action="store_true", default=False)
parser.add_argument("-n", dest="get_dbl_norm_cov", help="Set flag to build double normalized coverage matrix as sparse csr gzip numpy object (dirname+cov_norm_sparsecsr_mat.npz)",action="store_true", default=False)
args = parser.parse_args()

#%% Functions
def main(path_to_p_file, path_to_sample_names_file, path_to_outgroup_boolean_file, path_to_list_of_diversity_files, path_to_candidate_mutation_table, flag_cov_raw_sparse_matrix,flag_cov_norm_sparse_scale_matrix):
#omit path_to_list_of_quals_files
   
    pwd=os.getcwd()
    
    # p: positions on genome that are candidate SNPs
    print('Processing candidate SNP positions...')
    p=np.loadtxt(path_to_p_file,dtype=np.int32) #1-indexed!
    print('Total number of positions: ' + str(len(p)))
    
    # SampleNames: list of names of all samples
    print('Processing sample names...')
    
    fname=pwd + '/' + path_to_sample_names_file
    fid = open( fname, "r" ) # Input is space separated text file, in one line
    SampleNames = fid.readline().split()
    fid.close()
    numSamples = len(SampleNames) # save number of samples
    print('Total number of samples: ' + str(numSamples))
    
    
    ## in_outgroup: booleans for whether or not each sample is in the outgroup
    print('Processing outgroup booleans...')
    fname =  pwd + '/' + path_to_outgroup_boolean_file  
    fid = open( fname, "r" ) # Input is space separated text file, in one line
    in_outgroup_string = fid.readline().split()
    in_outgroup=np.array(in_outgroup_string)
    in_outgroup = in_outgroup.reshape(1,len(in_outgroup)) # reshape 2d array for analysis.py: 1row and numSamples cols
    fid.close()
    
    ## counts: counts for each base from forward and reverse reads at each candidate position for all samples
    print('Gathering counts data at each candidate position...\n')
    
    # Import list of directories for where to diversity.mat for each sample
    fname =  pwd + '/' + path_to_list_of_diversity_files  
    fid = open( fname, "r" ) # Input is space separated text file, in one line
    paths_to_diversity_files = fid.readline().split()
    fid.close()
    
    #what is this for???
    with gzip.open(paths_to_diversity_files[1],'rb') as f:
        tempfile=pickle.load(f)
    #data=tempfile['data']
    GenomeLength=len(tempfile)
    #GenomeLength=size[1]
        
    # Initialize counts and coverage
    counts = np.zeros((8, len(p), numSamples), dtype='uint')
    all_coverage_per_bp = np.zeros((numSamples,GenomeLength),dtype='uint')
    indel_counter=np.zeros((2, len(p), numSamples), dtype='uint') # Added 2019.02.12
    # Make counts and coverage at the same time
    for i in range (numSamples):
        print('Loading counts matrix for sample: ' + str(i)) 
        print('Filename: '+ paths_to_diversity_files[i]) 
        

        with gzip.open(paths_to_diversity_files[i],'rb') as f:
            data=pickle.load(f)
        data=data.T #Now only 8xp fields!
        counts[:,:,i]=data[:,p-1] #p-1 is for 1-index > 0-index conversion
        
        if flag_cov_raw_sparse_matrix:
            all_coverage_per_bp[i,:]=sum(data[0:8,:])
        
        # old matlab stuff
        # infile=sio.loadmat(paths_to_diversity_files[i])  01.21.22 Evan changed this to take in pickle files
        # data=infile['data']
        # counts[:,:,i]=data[0:8,p]
        #indel_counter[:,:,i]=data[38:40,p] # 01.21.22 Evan removed
    
    #counts = counts.transpose(2,0,1) # 01.21.22 Evan removed
    indel_counter = indel_counter.transpose(2,0,1) # indel_counter reshape for analysis.py: 0:samples,2,p 
    
    #print('Getting all the coverage information...\n')
    #[all_coverage_per_bp, ~, all_maf_per_bp] = get_all_coverage(SampleInfo, GenomeLength)
    
    # Normalize coverage by sample and then position; ignore /0 ; turn resulting inf to 0
    
    if flag_cov_norm_sparse_scale_matrix:
        with np.errstate(divide='ignore',invalid='ignore'):
            array_cov_norm = ( all_coverage_per_bp - np.mean(all_coverage_per_bp,axis=1,keepdims=True) ) / np.std(all_coverage_per_bp,axis=1,keepdims=True) # ,keepdims=True maintains 2D array (second dim == 1), necessary for braodcasting
            array_cov_norm[ ~np.isfinite(array_cov_norm) ] = 0
            
            # 2nd normalisation
            array_cov_norm = ( array_cov_norm - np.mean(array_cov_norm,axis=0,keepdims=True) ) / np.std(array_cov_norm,axis=0,keepdims=True) # ,keepdims=True maintains 2D array (second dim == 1), necessary for braodcasting
            array_cov_norm[ ~np.isfinite(array_cov_norm) ] = 0
    

    ## turn into sparse csr matrices for more efficient computation
    # scale norm matrix by 1000 and save as int64 to slim matrix as much as possible
    # save matrices
    if os.path.dirname(path_to_candidate_mutation_table) == '': # make sure cov matrix goes to same folder as cmt
        outdir = ''
    else:
        outdir = os.path.dirname(path_to_candidate_mutation_table) + '/'
    if flag_cov_raw_sparse_matrix:
        all_coverage_per_bp_csr = sparse.csr_matrix(all_coverage_per_bp)
        sparse.save_npz(outdir+'cov_raw_sparsecsr_mat.npz', all_coverage_per_bp_csr,compressed=True)
    if flag_cov_norm_sparse_scale_matrix:
        array_cov_norm_scaled_csr = sparse.csr_matrix((np.round(array_cov_norm,3)*1000).astype('int64'))
        sparse.save_npz(outdir+'cov_norm_sparsecsr_mat.npz', array_cov_norm_scaled_csr,compressed=True)

    ## Save cmt!   
    with gzip.open(path_to_candidate_mutation_table, 'wb') as f: 
        pickle.dump([SampleNames, p, counts, in_outgroup, indel_counter], f,protocol=4) # protocol=4 for storage of files >4gb
    
    print('DONE')

#%%
if __name__ == "__main__":
    path_to_p_file=args.allpositions
    path_to_sample_names_file=args.sampleNames
    path_to_outgroup_boolean_file=args.outgroupBool
    path_to_list_of_diversity_files=args.divfiles
    path_to_candidate_mutation_table=args.candidate_mutation_table
    flag_cov_raw_sparse_matrix=args.get_cov
    flag_cov_norm_sparse_scale_matrix=args.get_dbl_norm_cov
    if flag_cov_norm_sparse_scale_matrix and not flag_cov_raw_sparse_matrix:
        flag_cov_raw_sparse_matrix = True
        print('Selected to build double normalized coverage matrix. Raw coverage matrix will be build, too.')
    main(path_to_p_file, path_to_sample_names_file, path_to_outgroup_boolean_file, path_to_list_of_diversity_files, path_to_candidate_mutation_table, flag_cov_raw_sparse_matrix,flag_cov_norm_sparse_scale_matrix)

