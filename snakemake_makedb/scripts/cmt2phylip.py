#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:50:42 2022

@author: evanqu
"""
import argparse
import subprocess
import pickle
import gzip
import numpy as np
from Bio import AlignIO

#%% Testing
# import os
# os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/strainslicer/benchmarks/2022_03_subsample_community_benchmarking/Cacnes_strainphlan_test/true_trees')
# input_cmt_file='coarse_candidate_mutation_table.pickle.gz'
# output_name_ids='veryfine_tree_rename.txt'
# input_tree_file='veryfine_res_truetree_pars.tre'
# rep_tree_file='veryfine_res_truetree_pars_isonames.tre'

# phylip2names=dict()
# with open(output_name_ids) as f:
#     for line in f:
#         key, value = line.strip().split('\t')
#         phylip2names[key] = value

# # Replace phylip tree names
# with open(input_tree_file) as f:
#     nwk=f.read()
# #Replace with representative isolate name
# for i in phylip2names.keys():
#     nwk=nwk.replace(i,phylip2names[i])
# with open(rep_tree_file,'w') as f:
#     f.write(nwk)
    
#%% Fxns
def read_cmt(input_cmt_file):
    with gzip.open(input_cmt_file) as f:
        CMT=pickle.load(f)
    sample_names=CMT[0]
    pos=CMT[1]
    counts=CMT[2]
    quals=CMT[3]
    indel_counter=CMT[4]
    
    return [sample_names, pos, counts, quals, indel_counter]

def get_major_allele_nt(counts):
    
    c=counts[0:4,:,:]+counts[4:8,:,:]; # flatten frw and rev ATCG counts    

    sorted_arr = np.sort(c,axis=0) #sort by ATCG counts
    sortedpositions = np.argsort(c,axis=0) # return matrix indices of sort
    
    # get allele counts for major allele (4th row)
    # weird "3:4:" indexing required to maintain 3d structure
    maxcount = sorted_arr[3:4:,:,:] 
    # get allele counts for first minor allele (3rd row)
    # tri/quadro-allelic ignored!!
    minorcount = sorted_arr[2:3:,:,:] 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        maf = maxcount / sorted_arr.sum(axis=0,keepdims=True)
        minorAF = minorcount / sorted_arr.sum(axis=0,keepdims=True)
    maf = np.squeeze(maf,axis=0) # turn 2D; axis=1 to keep 2d structure when only one position!
    maf[np.isnan(maf)]=0 # set to 0 to indicate no data
    minorAF = np.squeeze(minorAF,axis=0) 
    minorAF[np.isnan(minorAF)]=0 # set to 0 to indicate no data/no minor AF
    
    # index position in sortedpositions represents allele position ATCG;
    # A=0,T=1,C=2,G=3
    # axis=1 to keep 2d structure when only one position!
    majorNT = np.squeeze(sortedpositions[3:4:,:,:],axis=0) 
    minorNT = np.squeeze(sortedpositions[2:3:,:,:],axis=0)

    # Note: If counts for all bases are zero, then sort won't change the order
    # (since there is nothing to sort), thus majorNT/minorNT will be put to -1 (NA)
    # using maf (REMEMBER: minorAF==0 is a value!)
    majorNT[maf==0]=-1
    minorNT[maf==0]=-1
    
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
    palette = [-1,0,1,2,3] # values present in index-array
    index = np.digitize(calls.ravel(), palette, right=True)
    
    return nucl[index].reshape(calls.shape)

def write_calls_to_fasta(calls,sample_names,output_file):
    
    fa_file = open(output_file, "w")
    
    for i,name in enumerate(sample_names):
        nucl_string = "".join(list(calls[:,i]))
        fa_file.write(">" + name + "\n" + nucl_string + "\n")
    
    fa_file.close()


def cmt2phylip(input_cmt_file, output_phylip, output_name_ids,
               min_cov_to_include=8, min_maf_for_call=0.85,
               min_strand_cov_for_call=2, min_qual_for_call=-30,
               min_presence_core=0.85,min_median_cov_samples=3,
               consider_indels=False):
    '''Finds fixed mutations within a set of samples and outputs phylip format file

    Args:
        input_cmt_file (str): Path to input candidate mutation table.
        output_phylip (str): Path to output phylip file.
        output_name_ids (str): Path to output file to rename phylip names to original.
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
                       'min_cov_to_include':min_cov_to_include
                       }
    
    filterby_site_per_sample = {\
                              'min_maf_for_call':min_maf_for_call,
                              'min_strand_cov_for_call':min_strand_cov_for_call,
                              'min_qual_for_call': min_qual_for_call,
                              }
    
    filterby_site_across_samples = {\
                                  'min_presence_core':min_presence_core,
                                  'min_median_cov_samples':min_median_cov_samples,
                                  }
    # nts = np.array(['A','T','C','G'],dtype=object) # NTs='ATCG'
    
    print("Reading in candidate mutation table....")
    
    sample_names, pos, counts, quals, indel_counter = read_cmt(input_cmt_file)
    
    ### Modify some structures ###
    sample_names_all=np.asarray(sample_names,dtype=object)
    #reduce indel_counter to pxs matrix giving count for indels overall
    # indels_all=np.sum(indel_counter,axis=0)
    coverage_all=counts.sum(axis=0) #pxs
    
    print("Filtering samples...")
    
    #samples that pass sample-wise filters
    good_sample_bool = coverage_all.mean(axis=0) > float(filterby_sample['min_cov_to_include'])
    
    good_sample_names = sample_names_all[good_sample_bool]
    good_counts = counts[:,:,good_sample_bool]
    good_quals = quals[:,good_sample_bool]
    # good_indels = indels_all[:,good_sample_bool]
    
    num_samples=len(good_sample_names)
    coverage=good_counts.sum(axis=0)
    coverage_f_strand=good_counts[0:4,:,:].sum(axis=0) #should be pxs
    coverage_r_strand=good_counts[4:8,:,:].sum(axis=0)

    print(f"{num_samples}/{len(sample_names)} samples passed coverage filter.")
    print("The following samples did NOT pass coverage filter; NOT included:")
    print( sample_names_all[coverage_all.mean(axis=0) < float(filterby_sample['min_cov_to_include'])] )
    
    if num_samples < 3:
        raise Exception("Too few samples fullfill filter criteria!")
        
    print("Filtering positions...")
    #Get major allele at each position
    [maNT, maf, minorNT, minorAF] = get_major_allele_nt(good_counts)
    
    # Define mutations that pass filters in each and across samples.
    calls = np.copy(maNT)
    #remember quals are negative!
    calls[ good_quals > float(filterby_site_per_sample['min_qual_for_call']) ] = -1
    calls[ maf < float(filterby_site_per_sample['min_maf_for_call']) ] = -1
    calls[ coverage_f_strand < float(filterby_site_per_sample['min_strand_cov_for_call']) ] = -1
    calls[ coverage_r_strand < float(filterby_site_per_sample['min_strand_cov_for_call']) ] = -1
    
    if consider_indels:
        indels_all=np.sum(indel_counter,axis=0)
        good_indels = indels_all[:,good_sample_bool]
        calls[good_indels > (0.5*coverage)] = -1 #50%+ of reads @ site cannot support indel
        
    max_fracNs_bool = ( ((calls>-1).sum(axis=1)) <= \
                        (num_samples*float(filterby_site_across_samples['min_presence_core'])) )
    min_med_cov_bool = ( np.median(coverage, axis=1) < \
                         float(filterby_site_across_samples['min_median_cov_samples']) )
    
    site_filter_bool=np.any((max_fracNs_bool,min_med_cov_bool),axis=0)
    calls[site_filter_bool,:] = -1 # sites that fail qc ->-1, for all samples      
    
    print('Done filter.')
    
    print("Finding within-sample SNPs...")
    mut_qual,mut_qual_isolates = ana_mutation_quality(calls,quals)
    
    # nan in mutQual gives a warning, which is fine (nan=False=excluded)
    fixedmutation = (calls>-1)&(np.tile(mut_qual,(1,num_samples)) <= 1)
    hasmutation = np.any((fixedmutation == True,),axis=0)
    # NOTE: filteredpos is INDEX of filtered positions for p!
    filteredpos = np.where(np.sum(hasmutation, axis=1)> 0)[0] 
    
    # remove pos that are not variable between samples 
    # (fixed different to outgroup or N)
    bool_allN_outgroup = (calls>-1)
    # For outgroup functionality: NOT added yet!
    # bool_allN_outgroup[:,outgroup_idx] = False 
    
    #indices of nonvariable positions
    non_variable_idx = []
    for i,fp in enumerate(filteredpos):
        # identify sites non-variable between samples
        if len(np.unique(calls[fp,:][bool_allN_outgroup[fp,:]])) == 1:
            non_variable_idx.append(i)
    
    # NOTE: goodpos is INDEX of good positions for p!
    goodpos = np.delete(filteredpos,non_variable_idx)
    
    print(str(len(goodpos)) + ' goodpos found.')
    
    if len(goodpos) < 3:
        raise Exception("Too few positions after filter!")
    
    print("Writing phylip file...")
    # grab calls only at goodpos
    # numpy broadcasting of row_array requires np.ix_()
    calls_for_treei=calls[np.ix_(goodpos)]
    # Convert -10123 to NATCG translation
    calls_for_tree = idx2nts(calls_for_treei) 
    
    sample_names_4phylip = np.char.add(np.arange(0,len(good_sample_names)).astype(str), \
                                      good_sample_names.astype(str)).astype(object)
    
    #.dnapars.fa > for dnapars...deleted later
    write_calls_to_fasta(calls_for_tree,sample_names_4phylip,output_phylip+".dnapars.fa") 
    # turn fa to phylip and delete fasta with short tip labels    
    aln = AlignIO.read(output_phylip+".dnapars.fa", 'fasta')
    AlignIO.write(aln, output_phylip+".phylip", "phylip")
    subprocess.run(["rm -f " + output_phylip+".dnapars.fa"],shell=True)

    # Write object to convert phylip names back at the end
    with open(output_name_ids,'w') as f:
        for line in range(len(good_sample_names)):
            f.write(f"{sample_names_4phylip[line][:10]}\t{good_sample_names[line]}\n")
    

#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='Input', type=str, help='Path to input candidate mutation table.',required=True)
    parser.add_argument('-o', dest='Output', type=str, help='Path to output phylip file.', required=True)
    parser.add_argument('-n', dest='NameIDs', type=str, help='Path to output renaming file.', required=True)
    parser.add_argument('--min_cov', dest='MinCov', type=str, help='Minimum average coverage across positions to include a sample. Default=8.',required=False, default=8)
    parser.add_argument('--min_maf', dest='MinMAF', type=str, help='Minimum major allele frequency to call a major allele. Default=0.85.', required=False, default=0.85)
    parser.add_argument('--min_strand_cov', dest='MinStrandCov', type=str, help='Minimum coverage per strand to call a major allele. Default=2.', required=False, default=2)
    parser.add_argument('--min_qual', dest='MinQual', type=str, help='Minimum mapping quality to call a major allele. Default=-30.',required=False,default=-30)
    parser.add_argument('--min_presence_core', dest='MinCore', type=str, help='Minimum presence across samples to include a position. Default=0.9.', required=False,default=0.9)
    parser.add_argument('--min_median_cov_samples', dest='MinCovSamples', type=str, help='Minimum median coverage across samples to include a position. Default=3.', required=False, default=3)
    parser.add_argument('--filter_indels', dest='FilterIndels', type=str, help='Filter positions based on number of indels. Defaults to False.', required=False, default=False)

    args = parser.parse_args()

    cmt2phylip(args.Input, args.Output, args.NameIDs,
               min_cov_to_include=args.MinCov, min_maf_for_call=args.MinMAF,
               min_strand_cov_for_call=args.MinStrandCov, min_qual_for_call=args.MinQual,
               min_presence_core=args.MinCore,min_median_cov_samples=args.MinCovSamples,
               consider_indels=args.FilterIndels)






    

    