#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing scripts for StrainSlicer Snakemake
@author: evanqu
"""
from Bio import SeqIO
import os
import numpy as np
import pickle
import glob
import subprocess
import gzip

# modified read_move_link_samplesCSV.py         
def read_samples_CSV_classify(spls):
	# reads in samples_case.csv file, format: Path,Sample,ReferenceGenome,Outgroup
	hdr_check = ['Path','Sample','FileName','Classifier','ReferenceGenome','Outgroup']
	switch = "on"
	file = open(spls, 'r')
    #initialize lists
	list_path = []; list_splID = []; list_fileN = []
	list_cfrN = []; list_refG = []; list_outgroup = []
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
		list_cfrN.append(line[3])
		list_refG.append(line[4])
		list_outgroup.append(line[5])
	return [list_path,list_splID,list_fileN,list_cfrN,list_refG,list_outgroup]

def read_samples_CSV_makeclassifier(spls):
	hdr_check = ['Path','Sample','FileName','ReferenceGenome','Outgroup']
	switch = "on"
	file = open(spls, 'r')
    #initialize lists
	list_path = []; list_splID = []; list_fileN = []; list_refG = []; list_outgroup=[]
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
		list_outgroup.append(line[4])
	return [list_path,list_splID,list_fileN,list_refG,list_outgroup]


def split_samplesCSV_classify(PATH_ls , SAMPLE_ls , FILENAME_ls, CLASSIFIER_ls , REF_Genome_ls):
    # Evan custom for strainslicer
    # takes info extracted form samples.csv; saves each line of samples.csv as sample_info.csv in data/{sampleID}
    for i, sample in enumerate(SAMPLE_ls):
        # get info for this sample
        sample_info_csv_text = PATH_ls[i] + ',' + SAMPLE_ls[i] + ',' + FILENAME_ls[i] + ',' + CLASSIFIER_ls[i] + ',' + REF_Genome_ls[i]
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
            
def split_samplesCSV_makeclassifier(PATH_ls,SAMPLE_ls,REF_Genome_ls,FILENAME_ls,OUTGROUP_ls):
    ''' For make_classifier: takes info from samples.csv & saves each line as sample_info.csv in data/{sampleID}'''
    
    for i, sample in enumerate(SAMPLE_ls):
        # get info for this sample
        sample_info_csv_text = f"{PATH_ls[i]},{SAMPLE_ls[i]},{REF_Genome_ls[i]},{FILENAME_ls[i]},{OUTGROUP_ls[i]}"

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

            
def findfastqfile(dr,ID,filename):
    fwd=[]
    rev=[]
    #search for filename as directory first
    potentialhits_forward=glob.glob(dr + '/' + filename +'/*1.fastq.gz')
    potentialhits_reverse=glob.glob(dr + '/' + filename +'/*2.fastq.gz')
    if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
        fwd=potentialhits_forward[0]
        rev=potentialhits_reverse[0]
    #then search for filename as file.gz
    elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
        potentialhits_forward=glob.glob(dr + '/' + filename +'*1.fastq.gz')
        potentialhits_reverse=glob.glob(dr + '/' + filename +'*2.fastq.gz')
        if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
            fwd=potentialhits_forward[0]
            rev=potentialhits_reverse[0]
        #then search as unzipped file
        elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
            potentialhits_forward=glob.glob(dr + '/' + filename +'*1.fastq')
            potentialhits_reverse=glob.glob(dr + '/' + filename +'*2.fastq')
            if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
                subprocess.run("gzip " + potentialhits_forward[0], shell=True)  
                subprocess.run("gzip " + potentialhits_reverse[0], shell=True)
                fwd=potentialhits_forward[0]+'.gz'
                rev=potentialhits_reverse[0]+'.gz'
            else:
                foldername=glob.glob(dr + '/' + filename + '*')
                if foldername and os.path.isdir(foldername[0]):
                    foldername=foldername[0]
                    potentialhits_forward=glob.glob(foldername + '/*' + filename + '*1*.fastq.gz')
                    potentialhits_reverse=glob.glob(foldername + '/*' + filename + '*2*.fastq.gz')
                    if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
                        fwd=potentialhits_forward[0]
                        rev=potentialhits_reverse[0]
                    elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
                        print(foldername + '/*' + filename + '*2*.fastq.gz')
                        potentialhits_forward=glob.glob(foldername +  '/*' + filename + '*1*.fastq')
                        potentialhits_reverse=glob.glob(foldername + '/*' + filename + '*2*.fastq')
                        if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
                            subprocess.run("gzip " + potentialhits_forward[0], shell=True)  
                            subprocess.run("gzip " + potentialhits_reverse[0], shell=True)
                            fwd=potentialhits_forward[0]+'.gz'
                            rev=potentialhits_reverse[0]+'.gz'
    if not(fwd) or not(rev):
        raise ValueError('Either no file or more than 1 file found in ' + dr + 'for ' + ID)
    ##zip fastq files if they aren't already zipped
    subprocess.run("gzip " + fwd, shell=True)   
    subprocess.run("gzip " + rev, shell=True)   
    return [fwd, rev]
    
def makelink(path,sample,filename):
    #When sample is run on a single lane
    #File name can be either a COMPLETE directory name or a file name in batch(called path in this fx)
    [fwd_file, rev_file]=findfastqfile(path,sample, filename)
    subprocess.run('ln -s -T ' + fwd_file + ' data/' + sample + '/R1.fq.gz', shell=True)    
    subprocess.run('ln -s -T ' + rev_file + ' data/' + sample + '/R2.fq.gz', shell=True)    


def cp_append_files(paths,sample,filename):
    #When sample is run on multiple lanes with same barcode
    fwd_list=''
    rev_list=''
    for path in paths:
        #Provider name can be either a COMPLETE directory name or a file name in batch(called path in this fx)
        [fwd_file, rev_file]=findfastqfile(path,sample, filename)
        fwd_list=fwd_list+ ' ' + fwd_file
        rev_list=rev_list+ ' ' + rev_file
        print(rev_list)
        print(fwd_list)
    subprocess.run("zcat " + fwd_list + ' | gzip > data/' +  sample + '/R1.fq.gz', shell=True)
    subprocess.run("zcat " + rev_list + ' | gzip > data/' +  sample + '/R2.fq.gz', shell=True)

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
    print(str(len(all_pos)) +' total positions found across ' + str(len(os.listdir(list(cfrs)[0]))) + ' classifier(s)')
    
    chr_pos = p2chrpos(all_pos,chr_starts)
    chr_names = np.array([scaf_names[i-1] for i in chr_pos[:,0]])
    chr_pos_final = np.vstack((chr_names,chr_pos[:,1])).T
    #save as lists
    np.savetxt(output_allpos_file, all_pos, fmt='%i')
    np.savetxt(output_chrpos_file, chr_pos_final, delimiter='\t', fmt='%s')
    
    return