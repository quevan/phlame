#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing scripts for PhLAMe Snakemake
@author: evanqu
"""

from Bio import SeqIO
import os
import numpy as np
import pickle
import glob
import subprocess
import gzip
import re

# modified read_move_link_samplesCSV.py         
def read_samplesCSV_classify(path_to_samples_csv):

    hdr_check = ['Path','Sample','FileName','Classifier','Reference']

    #initialize lists
    path_ls = []
    spl_ls = []
    file_ls = []
    cfr_ls = []
    refG_ls = []

    with open(path_to_samples_csv) as f:

        for i, line in enumerate(f):

            line = line.strip('\n').split(',')
            if i == 0:
                    # Test Header. 
                    # Note: Even when header wrong code continues (w/ warning)
                if line != hdr_check:

                    print("CSV did NOT pass header check! Parsing will continue as if .csv is in this format: Path,Sample,FileName,Classifier,Reference")
                    continue
        
                else:
                    continue

            # Build lists
            path_ls.append(line[0])
            spl_ls.append(line[1])
            file_ls.append(line[2])
            cfr_ls.append(line[3])
            refG_ls.append(line[4])

    return [path_ls,spl_ls,file_ls,cfr_ls,refG_ls]


def read_samples_CSV_makeclassifier(spls):
    # reads in samples_case.csv file, format: Path,Sample,ReferenceGenome,Outgroup
    hdr_check = ['Path','Sample','FileName','ReferenceGenome']
    switch = "on"
    file = open(spls, 'r')
    #initialize lists
    list_path = []; list_splID = []; list_fileN = []; list_refG = []
    for line in file:
        line = line.strip('\n').split(',')
        # Test Header. Note: Even when header wrong code continues (w/ warning), but first line not read.
        if switch == "on":
            if (line == hdr_check):
                print("Passed CSV header check")
            else:
                Warning("CSV did NOT pass header check! Code continues, but first line ignored")
            switch = "off"
            continue
        # build lists
        list_path.append(line[0])
        list_splID.append(line[1])
        list_fileN.append(line[2])
        list_refG.append(line[3])
    return [list_path,list_splID,list_fileN,list_refG]


def split_samplesCSV_classify(PATH_ls, 
                              SAMPLE_ls, 
                              FILENAME_ls, 
                              CLASSIFIER_ls, 
                              REF_Genome_ls):
    
    # Evan custom for phlame
    
    # takes info extracted form samples.csv
    # + saves each line of samples.csv as sample_info.csv in data/{sampleID}
    for i, sample in enumerate(SAMPLE_ls):
        
        # get info for this sample
        sample_info_csv_text = f"{PATH_ls[i]},{SAMPLE_ls[i]},{FILENAME_ls[i]},{CLASSIFIER_ls[i]},{REF_Genome_ls[i]}"

        # make data directory for this sample if it doesn't already exist
        if not(os.path.isdir('data/' + sample)):
            os.makedirs('data/' + sample, exist_ok=True)
        # check to see if this mini csv with sample info already exists
        if os.path.isfile('data/' + sample + '/sample_info.csv'):
            # if so, read file
            old_file = open('data/' + sample + '/sample_info.csv','r')
            old_info = old_file.readline()
            old_file.close()
            # check to see if the existing file is consistent with samples.csv
            if not(old_info == sample_info_csv_text):
                # if not, remove the old file and save sample info in a new file
                print('Information file must be updated.')  
                os.remove('data/' + sample + '/sample_info.csv')
                f = open('data/' + sample + '/sample_info.csv','w')
                f.write(sample_info_csv_text) 
                f.close()
            #else:
            #print('Information file already updated.')              
        else: # if mini csv with sample info does not already exist
            # save sample info in mini csv
            #print('Information file must be created.')  
            f = open('data/' + sample + '/sample_info.csv','w')
            f.write(sample_info_csv_text) 
            f.close()
            
def split_samplesCSV_makeclassifier(PATH_ls,SAMPLE_ls,FILENAME_ls,REF_Genome_ls):
    # Added by Arolyn, 2019.02.11
    # takes info extracted form samples.csv; saves each line of samples.csv as sample_info.csv in data/{sampleID}
    for i, sample in enumerate(SAMPLE_ls):
        # get info for this sample
        sample_info_csv_text = PATH_ls[i] + ',' + SAMPLE_ls[i] + ',' + REF_Genome_ls[i] + ',' + FILENAME_ls[i]
        #print( sample )
        #print( sample_info_csv_text )
        # make data directory for this sample if it doesn't already exist
        if not(os.path.isdir('data/' + sample)):
            os.makedirs('data/' + sample, exist_ok=True)
        # check to see if this mini csv with sample info already exists
        if os.path.isfile('data/' + sample + '/sample_info.csv'):
            # if so, read file
            old_file = open('data/' + sample + '/sample_info.csv','r')
            old_info = old_file.readline()
            old_file.close()
            # check to see if the existing file is consistent with samples.csv
            if not(old_info == sample_info_csv_text):
                # if not, remove the old file and save sample info in a new file
                #print('Information file must be updated.')
                os.remove('data/' + sample + '/sample_info.csv')
                f = open('data/' + sample + '/sample_info.csv','w')
                f.write(sample_info_csv_text)
                f.close()
            #else:
            #print('Information file already updated.')
        else: # if mini csv with sample info does not already exist
            # save sample info in mini csv
            #print('Information file must be created.')
            f = open('data/' + sample + '/sample_info.csv','w')
            f.write(sample_info_csv_text)
            f.close()


def findfastqfile(dr,smple,filename):

    file_suffixs = ['.fastq.gz', '.fq.gz', '.fastq', '.fq',
                    '_001.fastq.gz', '_001.fq.gz', '_001.fastq', '_001.fq']


    # Search for filename as a prefix
    files_F = [f for f in os.listdir(dr) if re.search(f"{re.escape(filename)}_?R?1({'|'.join(file_suffixs)})",f)]
    files_R = [f for f in os.listdir(dr) if re.search(f"{re.escape(filename)}_?R?2({'|'.join(file_suffixs)})",f)]

    # Search for filename as a directory
    if os.path.isdir(f"{dr}/{filename}"):
        files_F = files_F + [f"{filename}/{f}" for f in os.listdir(f"{dr}/{filename}/") if re.search(f".*_?R?1({'|'.join(file_suffixs)})",f)]
        files_R = files_R + [f"{filename}/{f}" for f in os.listdir(f"{dr}/{filename}/") if re.search(f".*_?R?2({'|'.join(file_suffixs)})",f)]

    if len(files_F) == 0 or len(files_R) == 0:
        raise ValueError(f'No file found in {dr} for sample {smple} with prefix {filename}')
    
    elif len(files_F) > 1 or len(files_R) > 1:
        raise ValueError(f'More than 1 matching files found in {dr} for sample {smple} with prefix {filename}:\n \
                         {",".join(files_F)}\n \
                         {",".join(files_R)}')

    elif len(files_F) == 1 and len(files_R) == 1:
        file_F = f"{dr}/{files_F[0]}"
        file_R = f"{dr}/{files_R[0]}"

        ## Zip fastq files if they aren't already zipped
        if not file_F.endswith('.gz'):
            subprocess.run("gzip " + file_F, shell=True)
        if not file_R.endswith('.gz'):
            subprocess.run("gzip " + file_R, shell=True)
    
    return [file_F, file_R]


def makelink(path,sample,filename):
    #When sample is run on a single lane
    #File name can be either a COMPLETE directory name or a file name in batch(called path in this fx)
    [fwd_file, rev_file]=findfastqfile(path,sample, filename)
    subprocess.run('ln -s -T ' + fwd_file + ' data/' + sample + '/R1.fq.gz', shell=True)    
    subprocess.run('ln -s -T ' + rev_file + ' data/' + sample + '/R2.fq.gz', shell=True)    


def cp_append_files(path_ls,sample,filename_ls):
    #When sample is run on multiple lanes with same barcode
    fwd_list=''
    rev_list=''
    if len(path_ls)>1 and len(path_ls)==len(filename_ls):
        
        for path, file in zip(path_ls,filename_ls):
            [fwd_file, rev_file]=findfastqfile(path,sample,file)
            
            fwd_list=fwd_list+ ' ' + fwd_file
            rev_list=rev_list+ ' ' + rev_file
            print(rev_list)
            print(fwd_list)
            
        subprocess.run("zcat " + fwd_list + ' | gzip > data/' +  sample + '/R1.fq.gz', shell=True)
        subprocess.run("zcat " + rev_list + ' | gzip > data/' +  sample + '/R2.fq.gz', shell=True)

    elif len(path_ls)>1:
        for path in path_ls:
            #Provider name can be either a COMPLETE directory name or a file name in batch(called path in this fx)
            [fwd_file, rev_file]=findfastqfile(path, sample, filename_ls[0])
            fwd_list=fwd_list+ ' ' + fwd_file
            rev_list=rev_list+ ' ' + rev_file
            print(rev_list)
            print(fwd_list)
        subprocess.run("zcat " + fwd_list + ' | gzip > data/' +  sample + '/R1.fq.gz', shell=True)
        subprocess.run("zcat " + rev_list + ' | gzip > data/' +  sample + '/R2.fq.gz', shell=True)
    
    elif len(filename_ls)>1:
        for file in filename_ls:
            [fwd_file, rev_file]=findfastqfile(path_ls[0],sample, file)
            fwd_list=fwd_list+ ' ' + fwd_file
            rev_list=rev_list+ ' ' + rev_file
            print(rev_list)
            print(fwd_list)
        subprocess.run("zcat " + fwd_list + ' | gzip > data/' +  sample + '/R1.fq.gz', shell=True)
        subprocess.run("zcat " + rev_list + ' | gzip > data/' +  sample + '/R2.fq.gz', shell=True)
    else:
        raise IOError('Error: Multiple paths not found!')
        
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

def genomestats(REFGENOMEFOLDER):
    '''Parse genome to extract relevant stats

    Args:
        REFGENOMEFOLDER (str): Directory containing reference genome file.

    Returns:
        ChrStarts (arr): DESCRIPTION.
        Genomelength (arr): DESCRIPTION.
        ScafNames (arr): DESCRIPTION.

    '''

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
    return ChrStarts,Genomelength,ScafNames

def p2chrpos(p, ChrStarts):
    '''Convert 1col list of pos to 2col array with chr and pos on chr

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

def get_positions(path_to_classifiers,
                  output_allpos_file, output_chrpos_file, 
                  refgenome_file):
    ''' From a classifier object, produce a samtools compatible list of positions
        that are informative to calling clades

    Args:
        path_to_classifier (str): Single path to input classifier file, or path 
                                  to directory with multiple classifier files.\n
        output_allpos_file (str): Path to file giving positions as they appear in
                                  the classifier object (NOT samtools compatible).\n 
        output_chrpos_file (str): Path to samtools compatible list of positions.
        refgenome_folder (str): DESCRIPTION.

    Raises:
        Exception: DESCRIPTION.

    Returns:
        None.

    '''
    cat_pos = np.array([], dtype=np.int32)
    chr_starts, _, scaf_names = genomestats(refgenome_file)
    
    path_to_cfrs_ls = []
    # Parse whether file or directory of files
    if os.path.isdir(path_to_classifiers):
        
        for filename in os.listdir(path_to_classifiers):
            
            if filename.endswith('.classifier'):
                path_to_cfrs_ls.append(path_to_classifiers+'/'+filename)
    else:
        path_to_cfrs_ls.append(path_to_classifiers)
        
    
    # Get data from each classifier
    for cfr in path_to_cfrs_ls:
        
        with gzip.open(cfr,'rb') as f:
            csSNPs = pickle.load(f)
                            
            pos = csSNPs['cssnp_pos']
            cat_pos = np.concatenate([pos,cat_pos])

    allpos = np.unique(np.sort(cat_pos))
    
    print(f"{len(allpos)} informative positions found across {len(path_to_cfrs_ls)} classifier(s)")
    
    chr_pos = p2chrpos(allpos,chr_starts)
    chr_names = np.array([scaf_names[i-1] for i in chr_pos[:,0]])
    chr_pos_final = np.vstack((chr_names,chr_pos[:,1])).T
    
    #save as samtools compatible txt file
    np.savetxt(output_allpos_file, allpos, fmt='%i')
    np.savetxt(output_chrpos_file, chr_pos_final, delimiter='\t', fmt='%s')
    
    return
