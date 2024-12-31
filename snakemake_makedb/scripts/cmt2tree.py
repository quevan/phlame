#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:50:42 2022

@author: evanqu
"""

#%%
import argparse
import os
import subprocess
import pickle
import gzip
import time
import datetime
import glob
import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio import SeqIO
import matplotlib.pyplot as plt

import phlame.helper_functions as helper

#%% Testing
os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_06_Ecoli/')

input_cmt_file="CMTs/rep_Ecoli_candidate_mutation_table.pickle.gz"
output_phylip="Ecoli"
output_name_ids="Ecoli_phylip2names.txt"
refGenome_file="/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_06_Ecoli/Ecoli_ASM584/genome.fasta"

output_renaming_file='trees/Ecoli_phylip2names.txt'

min_cov_to_include=8
min_maf_for_call=0.85
min_strand_cov_for_call=2
max_qual_for_call=-30
min_presence_core=0.95
min_median_cov_samples=3
max_frac_ambiguous_pos=0.085
max_mean_copynum=2.5

#%%
input_cmt_file="CMTs/candidate_mutation_table.pickle.gz.npz"

data = np.load(input_cmt_file)
sample_names = data['sample_names']
pos = data['p']
counts = data['counts']
quals = data['quals']
indel_counter = data['indel_counter']
in_outgroup = data['in_outgroup']

# counts = counts.transpose(2,1,0)
# indel_counter = indel_counter.transpose(2,1,0)
# quals = quals.transpose()

with gzip.open('CMTs/Ecoli_candidate_mutation_table.pickle.gz','wb') as f:
    pickle.dump({'sample_names':sample_names,
                 'p':pos,
                 'counts':counts,
                 'quals':quals,
                 'indel_counter':indel_counter,
                 'in_outgroup':in_outgroup},f) 


#%%

def cmt2phylip(input_cmt_file, 
               output_phylip,
               output_renaming_file, 
               refGenome_file, 
               min_cov_to_include=10, 
               min_maf_for_call=0.9,
               min_strand_cov_for_call=3, 
               max_qual_for_call=-30,
               min_presence_core=0.9,
               min_median_cov_samples=3,
               max_frac_ambiguous_pos=0.05,
               max_mean_copynum=2.5,
               consider_indels=False,
               remov_recomb=False):
    
    '''Finds fixed mutations within a set of samples and outputs phylip format file

    Args:
        input_cmt_file (str): Path to input candidate mutation table.
        output_phylip (str): Path to output phylip file.
        output_renaming_file (str): Path to output file to rename phylip names to original.
        min_cov_to_include (float, optional): Minimum avg. coverage across positions to include a sample. Defaults to 8.
        min_maf_for_call (float, optional): Minimum major allele frequency to call a major allele for position. Defaults to 0.85.
        min_strand_cov_for_call (int, optional): Minimum coverage per strand to call a major allele for position. Defaults to 2.
        min_qual_for_call (int, optional): Minimum mapping quality to call a major allele for position. Defaults to -30.
        min_presence_core (float, optional): Minimum presence across samples to include a position. Defaults to 0.9.
        min_median_cov_samples (int, optional): Minimum median coverage across samples to include a position. Defaults to 3.
        consider_indels (bool, optional): Consider number of indels when filtering a position. Defaults to False.
        
    Returns:
        None.

    '''
    
    filterby_sample = {\
                       'min_cov_to_include':min_cov_to_include,
                       'max_frac_ambiguous_pos':max_frac_ambiguous_pos
                       }
    
    filterby_site_per_sample = {\
                              'min_maf_for_call':min_maf_for_call,
                              'min_strand_cov_for_call':min_strand_cov_for_call,
                              'max_qual_for_call': max_qual_for_call,
                              'max_frac_reads_supporting_indel':0.33
                              }
    
    filterby_site_across_samples = {\
                                  'min_presence_core':min_presence_core,
                                  'min_median_cov_samples':min_median_cov_samples,
                                  'max_mean_copynum':max_mean_copynum
                                  }
        
    filter_recombination = {\
                            'distance_for_nonsnp' : 300, #region in bp on either side of goodpos that is considered for recombination
                            'corr_threshold_recombination' : 0.75 #minimum threshold for correlation
                            }
    
    NTs = np.array(['A','T','C','G'],dtype=object) # NTs='ATCG'

    # =========================================================================
    #  Read in input files
    # =========================================================================
    
    print("Reading in candidate mutation table....")
    
    sample_names, pos, counts, quals, indel_counter, in_outgroup = read_cmt_gzip(input_cmt_file)

    ### Define some structures ###
    num_samples = len(sample_names)

    indels_all = np.sum(indel_counter,axis=0)
        # reduce indel_counter to pxs matrix giving count for indels overall

    coverage_all = counts.sum(axis=0) 
    
    sample_names = np.array(sample_names)

    # alignment_stats_df = parse_alignment_stats(path_to_alignment_stats)
    # alignment_stats_df = alignment_stats_df.reindex(sample_names)
    # plt.hist(alignment_stats_df['Percent Overall Alignment'], bins=50)
    
    # =============================================================================
    #  0. Filter samples by in_outgroup
    # =============================================================================
    
    # This is just temoporary to define a manual in_outgroup
    # in_outgroup_file = 'isolates_typed_metadata_all.csv'
    # in_outgroup = pd.read_csv(in_outgroup_file)['Gleopoldii_outgroup'].values
    # in_outgroup_bool = in_outgroup == 0

    in_outgroup_bool = np.array([True]*num_samples)

    sample_names = sample_names[in_outgroup_bool]
    counts = counts[:,:,in_outgroup_bool]
    quals = quals[:,in_outgroup_bool]
    indels_all = indels_all[:,in_outgroup_bool]
    
    coverage_all = coverage_all[:,in_outgroup_bool]

    # =============================================================================
    #  1. Filter samples by coverage
    # =============================================================================

    fig = plot_coverage_hist(coverage_all, float(filterby_sample['min_cov_to_include']))
    fig.show()

    print("Filtering samples by coverage...")
    # Filter samples by coverage first
        # Rationale is not removing low-cov samples first will mess with 
        # per pos / across sample filtering
    
    # Booleans are all INCLUSION criteria!
    good_cov_bool = ( np.median(coverage_all,axis=0) >= float(filterby_sample['min_cov_to_include']) )

    print(f"{np.sum(good_cov_bool)}/{len(sample_names)} samples passed median coverage filter.")
    print("The following samples did NOT pass median coverage filter:")
    print( sample_names[np.median(coverage_all,axis=0) < float(filterby_sample['min_cov_to_include'])] )
        
    if num_samples < 3:
        raise Exception("Too few samples fullfill filter criteria! Exiting...")

    good_sample_names = sample_names[good_cov_bool]
    good_counts = counts[:,:,good_cov_bool]
    good_quals = quals[:,good_cov_bool]
    good_indels = indels_all[:,good_cov_bool]
    
    ### Create structures from new sample list ###
    num_samples=len(good_sample_names)
    
    coverage=good_counts.sum(axis=0)
    coverage_f_strand=good_counts[0:4,:,:].sum(axis=0) #should be pxs
    coverage_r_strand=good_counts[4:8,:,:].sum(axis=0)

    # =============================================================================
    #  2. Filter on per position, per sample thresholds
    # =============================================================================
    print("Filtering basecalls...")
    # Order is important!
        # 1. Per position, per sample
        # 2. Per position across samples
        # 3. Per sample
        # 4. Recomb ? ( before sample filter ?)

    # Get major allele at each position
        # 01234=NATCG
    [maNT, maf, minorNT, minorAF] = get_major_allele_nt(good_counts)

    calls = np.copy(maNT)
    calls[ good_quals > float(filterby_site_per_sample['max_qual_for_call']) ] = 0
    # remember quals are negative!
    calls[ maf < float(filterby_site_per_sample['min_maf_for_call']) ] = 0
    calls[ coverage_f_strand < float(filterby_site_per_sample['min_strand_cov_for_call']) ] = 0
    calls[ coverage_r_strand < float(filterby_site_per_sample['min_strand_cov_for_call']) ] = 0

    # Filter positions with indels
    with np.errstate(divide='ignore',invalid='ignore'):
        frac_reads_w_indel = good_indels/coverage # sum reads supporting insertion plus reads supporting deletion
        frac_reads_w_indel[ ~np.isfinite(frac_reads_w_indel) ] = 0
    
    calls[ frac_reads_w_indel > float(filterby_site_per_sample['max_frac_reads_supporting_indel']) ] = 0

    # =========================================================================
    #  2.5 Filter samples with poor major allele frequency stats (contaminated)
    # =========================================================================

    # maNT_bool = np.count_nonzero((maf > 0) & (maf < 0.95), axis=0)/len(maf) < 0.05

    # =========================================================================
    #  3. Filter on per position, across sample thresholds
    # =========================================================================
    print("Filtering positions across samples...")
    
    fig = plot_positions_across_samples_hist(np.count_nonzero(calls, axis=1), 
                                              num_samples*float(filterby_site_across_samples['min_presence_core']),
                                              'Presence (not N) across samples')
    fig.show()

    # Booleans are all INCLUSION criteria!
    # Must be present (not N) in some fraction of samples
    min_core_bool = ( np.count_nonzero(calls, axis=1) >= \
                        (num_samples*float(filterby_site_across_samples['min_presence_core'])) )

    # Must have some minimum median coverage across samples
    min_medcov_bool = ( np.median(coverage, axis=1) >= \
                         float(filterby_site_across_samples['min_median_cov_samples']) )

    # Must not exceed max average copy number per samples
    max_copynum_bool = ( np.mean(coverage / np.median(coverage, axis=0),axis=1) <= \
                            float(filterby_site_across_samples['max_mean_copynum']) )
    
    pos_filter_bool = np.all( (min_core_bool,min_medcov_bool),
                                 axis=0)


    good_counts = good_counts[:,pos_filter_bool]
    goodcalls = calls[pos_filter_bool,:]
    goodpos = pos[pos_filter_bool]
    num_samples = np.size(goodcalls,1)

    print(f"{len(goodpos)}/{len(pos)} positions ({len(goodpos)*100/len(pos):.1f}%) passed filtering.")

    if len(goodpos) < 10:
        raise Exception("Too few positions retained after filtering! Exiting...")

    # =============================================================================
    #  4. Filter by sample
    # =============================================================================
    print('Filtering samples....')
    
    fig = plot_samples_hist(1-(np.count_nonzero(goodcalls, axis=0)/len(goodcalls)),
                            float(filterby_sample['max_frac_ambiguous_pos']))
    fig.show()

    max_fracNs_bool = ( 1-(np.count_nonzero(goodcalls, axis=0)/len(goodcalls)) <= \
                           float(filterby_sample['max_frac_ambiguous_pos']) )
        
    
    print(f"{len(good_sample_names[max_fracNs_bool])}/{len(good_sample_names)} samples passed breadth filtering.")
    print("The following samples did not pass breadth filtering:")
    print( good_sample_names[~max_fracNs_bool] )

    good_counts = good_counts[:,:,max_fracNs_bool]
    goodcalls = goodcalls[:,max_fracNs_bool]
    good_sample_names = good_sample_names[max_fracNs_bool]
    num_samples = len(good_sample_names)
    
    # =============================================================================
    #  5. Filter recombinant positions
    # =============================================================================

    #  Check for recombination in p and remove positions from goodpos
    if remov_recomb:
        print("Filtering recombinant positions...")
        # When no outgroup defined: refnt ~= ancnt:
        [chrStarts, genomeLength, scafNames] = genomestats(refGenome_file)

        refnt = extract_outgroup_mutation_positions(refGenome_file, p2chrpos(goodpos,chrStarts));
        ancnt = refnt
        ancnti_m = np.full(ancnt.shape, 9)
        
        # Change ATCGatcg to numeric 1234 (0 if no allele)
        for idx, allele in enumerate(ancnt):
            
            if allele in NTs:
                
                ancnti_m[idx,] = np.where(NTs==allele)[0][0]+1 # strip down to index number
            
            else:
                ancnti_m[idx,] = 0

        recombpos = findrecombinantSNPs(pos, 
                                        goodpos, 
                                        good_counts, 
                                        ancnti_m, num_samples, 
                                        filter_recombination['distance_for_nonsnp'],
                                        filter_recombination['corr_threshold_recombination'])

        #These are the positions in p that are likely recombinant that we will remove from goodpos
        print(str(sum(np.isin(goodpos, recombpos))) + ' of a total ' + \
              str(goodpos.shape[0]) + ' ('  + str(sum(np.isin(goodpos, recombpos))/goodpos.shape[0]*100) + \
                  '%) positions in goodpos were found to be recombinant.')
        
        goodcalls = goodcalls[~np.isin(goodpos, recombpos),:]
        # goodpos = goodpos[~np.isin(goodpos, recombpos)]

    print("Writing phylip file...")
    # grab calls only at goodpos
    # numpy broadcasting of row_array requires np.ix_()
    calls_for_treei=goodcalls
    # Convert -10123 to NATCG translation
    calls_for_tree = idx2nts(calls_for_treei) 
    
    sample_names_4phylip = np.char.add(np.arange(0,len(good_sample_names)).astype(str), \
                                      good_sample_names.astype(str)).astype(object)
    
    #.dnapars.fa > for dnapars...deleted later
    write_calls_to_fasta(calls_for_tree,sample_names_4phylip,output_phylip+".dnapars.fa") 
    # turn fa to phylip and delete fasta with short tip labels    
    aln = AlignIO.read(output_phylip+".dnapars.fa", 'fasta')
    AlignIO.write(aln, output_phylip+".phylip", "phylip")
    # subprocess.run(["rm -f " + output_phylip+".dnapars.fa"],shell=True)

    # Write object to convert phylip names back at the end
    with open(output_renaming_file,'w') as f:
        for line in range(len(good_sample_names)):
            f.write(f"{sample_names_4phylip[line][:10]}\t{good_sample_names[line]}\n")

#%%

def parse_alignment_stats(path_to_alignment_stats):
    
    alignment_stats_df = pd.read_csv(path_to_alignment_stats) 
    alignment_stats_df.index = alignment_stats_df['Sample']
    
    return alignment_stats_df

def findrecombinantSNPs(pos,
                        goodpos,
                        good_counts,
                        ancnti_m,
                        num_samples, 
                        distance_for_nonsnp, 
                        corr_threshold_recombination):

    # tile ancestral nt to size of calls
    anc_nt_tiled = np.tile(ancnti_m, (num_samples, 1)).T
    
    [cmajorNT, cmajorAF, cminorNT, cminorAF] = get_major_allele_nt(good_counts)
    cmajorAF[np.isnan(cmajorAF)] = 0 #set nan values to 0
    cminorAF[np.isnan(cminorAF)] = 0     


    # Compute mutant allele frequency
    # Mutant allele frequency: sum major and minor allele frequencies
    # when they don't match the ancestral allele
    major_nt_mut_freq = cmajorAF
    major_nt_mut_freq[ np.where( cmajorNT == anc_nt_tiled) ] = 0
    minor_nt_mut_freq = cminorAF
    minor_nt_mut_freq[ np.where( cminorNT == anc_nt_tiled) ] = 0
    
    mutantAF = major_nt_mut_freq + minor_nt_mut_freq

    # Find preliminary SNV positions to test for recombination    
    filter_not_N = ( cmajorNT != 0 ) # mutations must be not N
    filter_not_ancestral = ( cmajorNT != anc_nt_tiled ) # mutations must differ from the ancestral allele
    # filter_quals_not_NaN = ( np.tile( mut_qual, (num_samples,1) ) >= 1) # alleles must have strong support
    
    fixedmutation = filter_not_N & filter_not_ancestral #& filter_quals_not_NaN # boolean    
    
    goodpos_bool = np.any( fixedmutation, axis=1 )
    goodpos_idx = np.where( goodpos_bool )[0]
    p_goodpos = pos[goodpos_idx] # extract preliminary SNV positions

    # Downsize mutant allele frequency to goodpos only
    mutantAF_goodpos = mutantAF[ goodpos_idx ]


    #look for recombination regions
    nonsnp = []
    
    for i in range(len(goodpos_idx)):
     
        p_snv = pos[goodpos_idx[i]]
        
        #find nearby snps
        if p_snv > distance_for_nonsnp:
            
            region = np.array(np.where((p_goodpos > p_snv - distance_for_nonsnp) & 
                                       (p_goodpos < p_snv + distance_for_nonsnp)) ).flatten()
            
            if len(region)>1: 
                
                r = mutantAF_goodpos[region,:]
                corrmatrix = np.corrcoef(r) 
                [a,b]=np.where(corrmatrix > corr_threshold_recombination)
                nonsnp.extend(list(region[a[np.where(a!=b)]]))
    
    nonsnp=np.unique(nonsnp)
    p_nonsnp = p_goodpos[ nonsnp ]
    p_keep = np.setdiff1d( p_goodpos, p_nonsnp )
    nonsnp_bool = np.isin( pos, p_nonsnp )

    return p_nonsnp


def read_cov_mat_gzip( raw_cov_mat_file ):
    '''Loads raw coverage matrix from file.'''
    
    # Reads from file
    with gzip.open(raw_cov_mat_file, 'rb') as f:
        raw_cov_mat = pickle.load(f)
        
    return raw_cov_mat

def read_cmt_gzip( path_to_cmt_file ):
    '''Reads in candidate mutation table from pickled object file.'''
    
    # Reads from file
    with gzip.open(path_to_cmt_file, 'rb') as f:
        CMT = pickle.load(f)
    
    counts = CMT['counts']
    sample_names = CMT['sample_names']
    pos = CMT['p']
    quals=CMT['quals']
    indel_counter=CMT['indel_counter'];
    in_outgroup = CMT['in_outgroup'][0]
        
    return sample_names, pos, counts, quals, indel_counter, in_outgroup

#Todo: add formal IO errors
# def read_cmt(path_to_cmt_file):
#     '''Read in candidate mutation table from pickled object file.

#     Args:
#         path_to_cmt_file (str): String path to candidate mutation table file.

#     Returns:
#         sample_names (arr): Array of sample names.
#         pos (TYPE): DESCRIPTION.
#         counts (TYPE): DESCRIPTION.
#         quals (TYPE): DESCRIPTION.
#         indel_counter (TYPE): DESCRIPTION.

#     '''
    
#     if path_to_cmt_file.endswith('.pickle.gz'):
#         with gzip.open(path_to_cmt_file,'rb') as f:
#             CMT = pickle.load(f)
    
#     elif path_to_cmt_file.endswith('.pickle'):
#         with open(path_to_cmt_file,'rb') as f:
#             CMT = pickle.load(f)

            
#     if type(CMT)==dict:
#         counts = CMT['counts']
#         sample_names = CMT['sample_names']
#         pos = CMT['p']
#         quals=CMT['quals']
#         indel_counter=CMT['indel_counter']
#     else:
#         counts = CMT[2] 
#         sample_names=CMT[0]
#         pos = CMT[1]
#         quals = CMT[3]
#         # indel_counter=CMT[4]
#         indel_counter=False

#     if np.size(counts,axis=0) != 8:
#         print('Transposing counts table...')
#         counts = counts.transpose(1,2,0)
        
#     return np.array(sample_names), pos, counts, quals, indel_counter

def read_alignment_stats(alignment_stats_file):
    perc_aligned_dict = dict()
    perc_aligned_ls = []
    with open(alignment_stats_file, 'r') as f:
        for _ in range(1):
            next(f)
        for line in f:
            alignment_stats = line.split(',')
            sam_name = alignment_stats[1]
            perc_aligned = alignment_stats[7].rstrip('\n')
            perc_aligned_dict[sam_name] = float(perc_aligned)
            perc_aligned_ls.append(perc_aligned)
    return np.array(perc_aligned_ls,dtype=np.float32), perc_aligned_dict

def genomestats(REFGENOMEFILE):
    # parse ref genome to extract relevant stats
    refgenome = SeqIO.parse(REFGENOMEFILE,'fasta')
    Genomelength = 0
    ChrStarts = []
    ScafNames = []
    for record in refgenome:
        ChrStarts.append(Genomelength) # chr1 starts at 0 in analysis.m
        Genomelength = Genomelength + len(record)
        ScafNames.append(record.id)
    # turn to np.arrys!
    ChrStarts = np.asarray(ChrStarts,dtype=int)
    Genomelength = np.asarray(Genomelength,dtype=int)
    ScafNames = np.asarray(ScafNames,dtype=object)
    return [ChrStarts,Genomelength,ScafNames]


def get_major_allele_nt(counts):
    
    counts_by_allele = counts[0:4,:,:] + counts[4:8,:,:] # flatten frw and rev ATCG counts    

    counts_sort = np.sort(counts_by_allele,axis=0) #sort by ATCG counts
    counts_argsort = np.argsort(counts_by_allele,axis=0) # return matrix indices of sort
    
    # get allele counts for major allele (4th row)
    # weird "3:4:" indexing required to maintain 3d structure
    majorcount = counts_sort[3:4:,:,:] 
    # get allele counts for first minor allele (3rd row)
    # tri/quadro-allelic ignored!!
    minorcount = counts_sort[2:3:,:,:] 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        
        maf = majorcount / counts_sort.sum(axis=0,keepdims=True)
        minorAF = minorcount / counts_sort.sum(axis=0,keepdims=True)
    maf = np.squeeze(maf,axis=0) # turn 2D; axis=1 to keep 2d structure when only one position!
    maf[np.isnan(maf)]=0 # set to 0 to indicate no data
    minorAF = np.squeeze(minorAF,axis=0) 
    minorAF[np.isnan(minorAF)]=0 # set to 0 to indicate no data/no minor AF
    
    # index position in sortedpositions represents allele position ATCG;
    # A=0,T=1,C=2,G=3
    # axis=1 to keep 2d structure when only one position!
    majorNT = np.squeeze(counts_argsort[3:4:,:,:],axis=0)+1
    minorNT = np.squeeze(counts_argsort[2:3:,:,:],axis=0)+1

    # Note: If counts for all bases are zero, then sort won't change the order
    # (since there is nothing to sort), thus majorNT/minorNT will be put to -1 (NA)
    # using maf (REMEMBER: minorAF==0 is a value!)
    majorNT[maf==0]=0
    minorNT[maf==0]=0
    
    return majorNT, maf, minorNT, minorAF


def ana_mutation_quality(calls,quals):
    # This function calls mutations within the data itself, instead of wrt reference 
    # It takes as input the called nucleotides (calls) and the quality score (qual)
    # and outputs only the mutation call quality (mut_qual) calculated as 
    # max_i(min_j(Qual_i, Qual_j)) for i,j over all samples with different calls
    # It also outputs the samples giving this maxmin quality (mut_qual_isolates)
    
    # If there is no within data mutation in a given position (all nucleotides
    # are equal, but different from the reference), mut_qual returns 0 at that
    # position.
    
    [n_muts, n_strain] = calls.shape ;
    mut_qual = np.zeros((n_muts,1)) ; 
    mut_qual_isolates = np.zeros((n_muts,2)); 
    
    # generate template index array to sort out strains gave rise to reported FQ values
    s_template=np.zeros( (len(calls[0,:]),len(calls[0,:])) ,dtype=object)
    for i in range(s_template.shape[0]):
        for j in range(s_template.shape[1]):
            s_template[i,j] = str(i)+"_"+str(j)

    for k in range(n_muts):
        if len(np.unique(np.append(calls[k,:], 4))) <= 2: # if there is only one type of non-N (4) call, skip this location
            mut_qual[k] = np.nan ;
            mut_qual_isolates[k,:] = 0; 
        else:
            c = calls[k,:] ; c1 = np.tile(c,(c.shape[0],1)); c2 = c1.transpose() # extract all alleles for pos k and build 2d matrix and a transposed version to make pairwise comparison
            q = quals[k,:] ; q1 = np.tile(q,(q.shape[0],1)); q2 = q1.transpose() # -"-
            g = np.all((c1 != c2 , c1 != 4 , c2 != 4) ,axis=0 )  # no data ==4; boolean matrix identifying find pairs of samples where calls disagree (and are not N) at this position
            #positive_pos = find(g); # numpy has no find; only numpy where, which does not flatten 2d array that way
            # get mut_qual + logical index for where this occurred
            mut_qual[k] = np.max(np.minimum(q1[g],q2[g])) # np.max(np.minimum(q1[g],q2[g])) gives lower qual for each disagreeing pair of calls, we then find the best of these; NOTE: np.max > max value in array; np.maximum max element when comparing two arryas
            MutQualIndex = np.argmax(np.minimum(q1[g],q2[g])) # return index of first encountered maximum!
            # get strain ID of reorted pair (sample number)
            s = s_template
            strainPairIdx = s[g][MutQualIndex]
            mut_qual_isolates[k,:] = [strainPairIdx.split("_")[0], strainPairIdx.split("_")[1]]
            
    return [mut_qual,mut_qual_isolates]


def idx2nts(calls, missingdata="?"):
    # translate index array to array containing nucleotides
    # add 5th element --> no data! == index -1
    nucl = np.array([missingdata,'A','T','C','G'],dtype=object) 
    palette = [0,1,2,3,4] # values present in index-array
    index = np.digitize(calls.ravel(), palette, right=True)
    
    return nucl[index].reshape(calls.shape)

def write_calls_to_fasta(calls,sample_names,output_file):
    
    fa_file = open(output_file, "w")
    
    for i,name in enumerate(sample_names):
        nucl_string = "".join(list(calls[:,i]))
        fa_file.write(">" + name + "\n" + nucl_string + "\n")
    
    fa_file.close()

def extract_outgroup_mutation_positions(REFGENOMEFILE,position):
    # extracts the ref nucleotide for every position. positions needs to be sorted by chr
    # CMTpy=True: if old matlab build_candidate_mutation.mat used, put flag False. p 1-based correction
    refgenome = SeqIO.parse(REFGENOMEFILE,'fasta')
    refnt = np.zeros(position.shape[0],dtype=object)
    pos_counter = 0
    chr_counter = 1
    for record in refgenome:
        poschr = position[ position[:,0]==chr_counter , 1]
        for sglpos in poschr:
            refnt[pos_counter] = str(record.seq)[sglpos] 

            pos_counter += 1
        chr_counter += 1
    return refnt

def p2chrpos(p, ChrStarts):
    '''# return 2col array with chr and pos on chr
    #p...continous, ignores chr
    #pos: like p, 0-based'''

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

def generate_dnapars_tree(path_to_phylip_file,path_to_output_tree,
                          path_to_renaming_file=False):
    # Write alignment file (as fasta)
    # calc NJ or Parsimonous tree or None
    # writeDnaparsAlignment==True for writing dnapars input for usage on cluster
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    if not path_to_phylip_file.endswith('.phylip'):
        path_to_phylip_file += '.phylip'

    # Find dnapars executable; searches up to 5 directories back
    dnapars_path = glob.glob('dnapars')
    path_extension = "../"
    backstop = 0
    while len(dnapars_path) == 0 and backstop <= 5:
        dnapars_path = glob.glob(path_extension+'dnapars')
        path_extension = path_extension + "../"
        backstop = backstop + 1
    if len(dnapars_path) == 0:
        raise ValueError('Error: dnapars executable could not be located.')
    elif dnapars_path[0]=='dnapars':
        dnapars_path[0] = './dnapars'
    
    # Write parameters file
    with open(f"{path_to_output_tree}_{timestamp}.options.txt",'w') as file:
        file.write(path_to_phylip_file+"\n")
        file.write("f"+"\n")
        file.write(f"{path_to_output_tree}_{timestamp}.dnapars"+"\n")
        file.write("5"+"\n")
        file.write("V"+"\n")
        file.write("1"+"\n")
        file.write("y"+"\n")
        file.write("f"+"\n")
        file.write(f"{path_to_output_tree}_{timestamp}.tre"+"\n"+"\n") #Path to tree

    # Run dnapars
    print("Building parsimony tree...")
    print( f"{dnapars_path[0]} < {path_to_output_tree}_{timestamp}.options.txt > {path_to_output_tree}_{timestamp}.dnapars.log")
    subprocess.run([ "touch outtree"  ],shell=True)
    subprocess.run([ f"{dnapars_path[0]} < {path_to_output_tree}_{timestamp}.options.txt > {path_to_output_tree}_{timestamp}.dnapars.log"  ],shell=True)
    print("Done!")
    
    # Re-write tree with new long tip labels  
    if path_to_renaming_file:
        print("Renaming tree with original labels...")
        path_to_output_renamed_tree=f"{path_to_output_tree}_{timestamp}_isonames.tre"
        rename_phylip(path_to_renaming_file,f"{path_to_output_tree}_{timestamp}.tre",path_to_output_renamed_tree)

    return timestamp

def rename_phylip(phylip2names_file, intree, outtree, 
                  outclustertree=False, rep_CMT_file=False):
    '''Given a renaming file, rename 10chr phylip names into long format

    Args:
        phylip2names_file (TYPE): DESCRIPTION.
        intree (TYPE): DESCRIPTION.
        outtree (TYPE): DESCRIPTION.

    Returns:
        None.

    '''
    # Get phylip2names as dict
    phylip2names=dict()
    with open(phylip2names_file) as f:
        for line in f:
            key, value = line.strip().split('\t')
            phylip2names[key] = value
            
    # Replace phylip tree names
    with open(intree) as f:
        nwk=f.read()
    #Replace with representative isolate name
    for i in phylip2names.keys():
        nwk=nwk.replace(i,phylip2names[i])
    with open(outtree,'w') as f:
        f.write(nwk)
    
    if outclustertree: # Optionally output tree named by cluster
    
        # Get which tree isolate belongs to which cluster
        with gzip.open(rep_CMT_file,'rb') as f:
            CMT=pickle.load(f); sample_names=CMT[0]; cluster_IDs=CMT[4]
        tree2cluster=dict()
        for sam,clu in zip(sample_names,cluster_IDs):
            tree2cluster[sam]=clu
            
        #Replace with cluster name
        for i in tree2cluster.keys():
            nwk=nwk.replace(i,'Cluster '+tree2cluster[i])
        with open(outclustertree,'w') as f:
            f.write(nwk)


def plot_coverage_hist(coverage_matrix,
                       coverage_cutoff):
    
    coverage_median = np.median(coverage_matrix,axis=0)

    fig, axs = plt.subplots()
    
    maxcov=coverage_median.max()
    maxcovbin=np.ceil(maxcov/10)*10+10
    my_bins = np.arange(0,int(maxcovbin),5)
    n, bins, patches = plt.hist(x=coverage_median, bins=my_bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Median coverage')
    plt.axvline(coverage_cutoff, color='r')
    plt.ylabel('Number of samples')
    plt.title('Median coverage across samples')
    
    return fig

def plot_positions_across_samples_hist(presence_arr,
                                       max_ns_cutoff,
                                       filter_name):
    
    fig, axs = plt.subplots()

    my_bins = np.linspace( np.min(presence_arr), np.max(presence_arr), 100 )
    axs.hist(x=presence_arr, bins=my_bins, 
             color='#0504aa', alpha=0.7, rwidth=0.85)
    
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(filter_name)
    plt.ylabel('Number of positions')
    # Add a line at filter cutoff
    plt.axvline(x = max_ns_cutoff, color = 'r')
    
    return fig

def plot_samples_hist(breadth_bysample_arr,
                      min_breadth_cutoff):
    
    fig, axs = plt.subplots()
    
    # max()
    # my_bins = np.arange(0,1,0.01)
    
    n, bins, patches = plt.hist(x=breadth_bysample_arr, bins=20, 
                                color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Percentage of positions with Ns per sample')
    plt.axvline(min_breadth_cutoff, color='r')
    plt.ylabel('Number of samples')
    # plt.title('Median coverage across samples')
    
    return fig

    

#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='Input', type=str, help='Path to input candidate mutation table.',required=True)
    parser.add_argument('-p', dest='Phylip', type=str, help='Path to output phylip.', required=True)
    parser.add_argument('-o', dest='Output', type=str, help='Path to output tree.', required=True)
    parser.add_argument('-n', dest='NameIDs', type=str, help='Path to output renaming file.', required=True)
    parser.add_argument('-r', dest='RefGenome', type=str, help='Path to reference genome file.', required=True)
    parser.add_argument('--min_cov', dest='MinCov', type=str, help='Minimum average coverage across positions to include a sample. Default=8.',required=False, default=8)
    parser.add_argument('--min_maf', dest='MinMAF', type=str, help='Minimum major allele frequency to call a major allele. Default=0.85.', required=False, default=0.85)
    parser.add_argument('--min_strand_cov', dest='MinStrandCov', type=str, help='Minimum coverage per strand to call a major allele. Default=2.', required=False, default=2)
    parser.add_argument('--min_qual', dest='MinQual', type=str, help='Minimum mapping quality to call a major allele. Default=-30.',required=False,default=-30)
    parser.add_argument('--min_presence_core', dest='MinCore', type=str, help='Minimum presence across samples to include a position. Default=0.9.', required=False,default=0.9)
    parser.add_argument('--min_median_cov_samples', dest='MinCovSamples', type=str, help='Minimum median coverage across samples to include a position. Default=3.', required=False, default=3)
    parser.add_argument('--filter_indels', dest='FilterIndels', type=str, help='Filter positions based on number of indels. Defaults to False.', required=False, default=False)
    parser.add_argument('--remov_recomb', dest='Recomb', type=bool, help='Filter positions based on number of indels. Defaults to False.', required=False, default=False)


    args = parser.parse_args()
    
    if args.Recomb:
        print('Remove recombinant regions: Yes')

    cmt2phylip(args.Input, args.Phylip, args.NameIDs, args.RefGenome,
           min_cov_to_include=args.MinCov, min_maf_for_call=args.MinMAF,
           min_strand_cov_for_call=args.MinStrandCov, min_qual_for_call=args.MinQual,
           min_presence_core=args.MinCore,min_median_cov_samples=args.MinCovSamples,
           consider_indels=args.FilterIndels, remov_recomb=args.Recomb)

    generate_dnapars_tree(args.Phylip,args.Output,
                          path_to_renaming_file=args.NameIDs)



    

    