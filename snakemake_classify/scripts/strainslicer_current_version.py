############################
# strainSLICer v.0.2.2
# last updated: 11/5/2021    
############################
# updates:

    
import numpy as np
import pandas as pd
import h5py
import math
import pickle
import gzip
import warnings
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

def version_num():
    print('Version 0.2.2 Last updated 11/5/21')

def classify(path_to_cmt_file, path_to_classifier_file, path_to_output=None, fitmodel='zinbinom'):
    '''Wrapper for classifying MGs using a given classifier

    Args:
        path_to_cmt_file (str): DESCRIPTION.
        path_to_classifier_file (str): DESCRIPTION.
        path_to_output (str): DESCRIPTION.

    Returns:
        cts_cov (TYPE): DESCRIPTION.
        cts_cov_pos (TYPE): DESCRIPTION.
        sample_frequencies (TYPE): DESCRIPTION.

    '''
    
    print("Reading in candidate mutation table...")
    if path_to_cmt_file.endswith('.pickle.gz'):
        with gzip.open(path_to_cmt_file,'rb') as f:
            CMT = pickle.load(f)
            counts = CMT[2]; sample_names=CMT[0]; pos = CMT[1]
            
    elif path_to_cmt_file.endswith('.pickle'):
        with open(path_to_cmt_file,'rb') as f:
            CMT = pickle.load(f)
            counts = CMT[2]; sample_names=CMT[0]; pos=CMT[1]
    
    elif path_to_cmt_file.endswith('.mat'):
        counts, pos, sample_names = read_counts_table(path_to_cmt_file, 
                                                                   counts='counts', 
                                                                   pos='p', 
                                                                   sample_names='SampleNames')
    else:
        raise IOError("Error: candidate mutation table does not end in .mat or .pickle.gz!")
    
    print("Reading in classifier file...")
    with open(path_to_classifier_file, 'rb') as f:
        csSNPs, csSNP_pos, clade_names = pickle.load(f)
    
    print("Classifying...")
    cts_cov, fit_info, sample_frequencies = counts_CSS_ZINB(counts, 
                                                            pos, 
                                                            sample_names,
                                                            csSNPs, 
                                                            csSNP_pos,
                                                            clade_names,
                                                            5, model=fitmodel)
    #Save output
    if path_to_output:
        sample_frequencies.to_csv(path_to_output, sep=',')
    
    return cts_cov, fit_info, sample_frequencies

def reshape_clade_IDs_1col(clade_ids, unclassified_marker):
    '''
    Input:
        clade_ids: sx1 array of clade ids
        unclassified_marker: marker indicating unclassified @ that level. 
        Note: if nothing is unclassified, put anything & ignore warning
    '''
    
    clades = []; clade_names = []
    
    for c in np.unique(clade_ids):
        clades.append(np.where(clade_ids==c)[0])
        clade_names.append(str(c))
        
    good_clades = [clades[i] for i in range(len(clade_names)) if unclassified_marker != clade_names[i] ]
    good_clade_names = [clade_names[i] for i in range(len(clade_names)) if unclassified_marker != clade_names[i]]
    
    if len(good_clade_names) == len(clade_names):    
        print('Note: Nothing was found as unclassified in clade IDs. Ignore if intentional')
            
    return good_clades, good_clade_names

def reshape_cluster_IDs_2col(cluster_IDs, unclassified_marker):
    '''
    Input:
        cluster_IDs: 2 column array formatted as: sample_name, cluster_ID
        unclassified_marker: marker indicating unclassified @ that level. 
        Note: if nothing is unclassified, put anything & ignore warning
    '''
    
    clades = []; clade_names = []
    
    for c in np.unique(cluster_IDs[:,1]):
        clades.append(cluster_IDs[np.where(cluster_IDs==c)[0],0])
        clade_names.append(str(c))
        
    good_clades = [clades[i] for i in range(len(clade_names)) if unclassified_marker != clade_names[i] ]
    good_clade_names = [clade_names[i] for i in range(len(clade_names)) if unclassified_marker != clade_names[i]]
    
    if len(good_clade_names) == len(clade_names):    
        print('Note: Nothing was found as unclassified in clade IDs. Ignore if intentional')
            
    return good_clades, good_clade_names


def find_clade_specific_snps(maNT, maNT_pos, clades, n, core):
    '''
    Input:
        maNT: major allele NT for isolate samples (p x s) NATCG=01234
        maNT_pos: position on the reference for each maNT (p x 1)
        clades: list of arrays giving which isolates belong to which clades
        n: % Ns within clade tolerated (default 0.1)
        core: % of isolates tolerated to not have a position (default 0.1)
    '''
    
    is_core_genome = np.count_nonzero(maNT, axis=1)/len(maNT[1]) >= 1-core #core if p is in %n of all isolates
    core_genome = maNT[is_core_genome]; core_pos = maNT_pos[is_core_genome]
    print(f"Number of core positions: {len(core_pos)}/{len(maNT_pos)}")
    
    #finds all    
    def unanimous_to_clade():
        #Initialize output array, size pxc
        unanimous_clade_alleles = np.zeros([len(core_genome),len(clades)])

        for c in range(len(clades)):
        #Loop through every clade and pull out maNT for only samples belonging to that clade
            this_clade_samples = core_genome[:,clades[c]]
                                    
            n_tolerance = (np.count_nonzero(this_clade_samples,axis=1) / this_clade_samples.shape[1]) > 1-n
            
            #Cool trick for allele calling that's tolerant of Ns: append a column of Ns (0) to every position
            clade_samples_add_n = np.append(this_clade_samples, np.zeros((len(core_genome),1)), axis=1)
            #Count number of unique values in each row. Good pos have 2 unique values: the unanimous allele and N.
            is_unanimous_within_clade = np.count_nonzero(np.diff(np.sort(clade_samples_add_n)), axis=1)+1 == 2
            
            print(f"Number of alleles unanimous to clade {c}: {np.sum(is_unanimous_within_clade)}") #Testing
            #combine to output array
            unanimous_clade_alleles[:,c] = (n_tolerance & is_unanimous_within_clade)*np.max(this_clade_samples,axis=1)
                
        return unanimous_clade_alleles
    
    def unique_to_clade(unanimous_clade_alleles):
        #ask if a position is 'core' - aka present (non-n) in least 90% of all clades
        #is_core_genome_byclade = (np.count_nonzero(unanimous_clade_alleles,axis=1) / unanimous_clade_alleles.shape[1]) > 1-core #core if p is in n% of all clades    
        #is_core_genome = np.count_nonzero(maNT, axis=1)/len(maNT[1]) > 1-core #core if p is in %n of all isolates

        #append column of Ns to every position
        core_genome_addn = np.append(unanimous_clade_alleles, np.zeros((len(unanimous_clade_alleles),2)), axis=1)
        
        #initialize some arrays for output
        is_polymorphic = []; candidate_cssnps = np.zeros([len(unanimous_clade_alleles),len(clades)])
        
        #get polymorphic positions
        for p in range(len(unanimous_clade_alleles)):
            alleles_across_clades = np.unique(core_genome_addn[p], return_index=True,return_counts=True)
            is_polymorphic.append( len(alleles_across_clades[0]) > 2 )
            
            #if this position has a unique allele, grab NT and clade info
            #This is UGLY
            if (alleles_across_clades[2] == 1).any():
                unique_allele = alleles_across_clades[0][np.where(alleles_across_clades[2] == 1)]
                unique_clade = alleles_across_clades[1][np.where(alleles_across_clades[2] == 1)]
                
                for nt in range(len(unique_allele)):
                    candidate_cssnps[p,unique_clade[nt]] = unique_allele[nt]
        
        #unique check
        a = np.arange(0, np.size(maNT,1))
        unclassified = a[~np.in1d(a,np.concatenate(clades))] #get unassigned samples
        
        for c in range(len(clades)):
            #checks if unique in all isolates that have a clade assigned
            is_unique = np.sum(np.expand_dims(candidate_cssnps[:,c],1) == np.delete(core_genome,np.concatenate((clades[c], unclassified)),1), 1) > 0  #i think boolean mask would be nominally faster??
            candidate_cssnps[is_unique,c] = 0

        return candidate_cssnps        
    
    UnanimousCladeAlleles = unanimous_to_clade()
    CSS_matrix = unique_to_clade(UnanimousCladeAlleles)
    
    #temporary until i can standardize input
    if len(np.shape(core_pos)) > 1:
        core_pos = np.squeeze(core_pos)
        
    #save only positions with a cssnp allele
    cssnps = CSS_matrix[np.count_nonzero(CSS_matrix,1) > 0]
    cssnp_pos = core_pos[np.count_nonzero(CSS_matrix,1) > 0] 
    
    return cssnps, cssnp_pos

#First snakemake compatible version
def counts_CSS_snakemake(counts,pos,clade_names,CSS_matrix,CSS_pos,min_snps):
    
    if len(np.shape(CSS_pos)) > 1:
        raise Exception('Structure for CSS_pos incorrect')
    
    # Pull out 'informative positions': positions with both a candidate mutations and a csSNP   
    pos_with_CSS = CSS_pos[np.where(CSS_matrix != 0)[0]] 
    counts_CSS_positions = [p for p in pos_with_CSS if p in pos] # list of informative positions with csSNP and CM
    
    #corresponding index on CSS_matrix
    #I can't find a good solution to this - currently using weird searchsorted method
    CSS_pos_sorted = np.argsort(CSS_pos); counts_CSS_positions_pos = np.searchsorted(CSS_pos[CSS_pos_sorted], counts_CSS_positions)
    counts_CSS_index = CSS_pos_sorted[counts_CSS_positions_pos]
        #counts_CSS_index = np.where(CSS_pos == counts_CSS_positions)[0] old version
        #counts_CSS_index = np.nonzero(CSS_matrix)[0]
    
    if len(counts_CSS_positions) != len(pos_with_CSS):
        raise Exception('Counts and csSNP positions do not match. Check that references are same and that csSNP positions are being included during the case step')
        
    #Grab cluster and allele info for each informative alleles
    clusters = np.nonzero(CSS_matrix[np.unique(counts_CSS_index)])[1]
    alleles = CSS_matrix[counts_CSS_index,clusters]
    
    #location of informative pos on counts matrix        
    counts_index=[ np.where(np.isin(pos,p))[0] for p in counts_CSS_positions if p in pos]
        #my original line: counts_index=np.where(np.isin(pos,counts_CSS_positions))[0] got deprecated (7/25)
        #sample_mode = stats.mode(np.sum(counts[:,:,:],axis=0),axis=0)[0].flatten() #mode counts of each sample
        #sample_mean = np.mean(np.sum(counts[:,:,:],axis=0),axis=0)
        #sample_std = np.std(np.sum(counts[:,:,:],axis=0),axis=0)
    
    
    #Initialize a bunch of arrays for next step
    counts_fwd=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)]); counts_rev=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)])
    cov_fwd=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)]); cov_rev=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)])

    # Grab # of counts supporting the csSNP allele and frac of total reads for informative positions
    np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 warnings
    for s in range(np.size(counts, axis=2)):
        for p in range(len(counts_CSS_positions)):
            counts_fwd[p,s] = counts[int(alleles[p]-1),counts_index[p][0],s]
            counts_rev[p,s] = counts[int(alleles[p]+3),counts_index[p][0],s]
            cov_fwd[p,s] = counts[int(alleles[p]-1),counts_index[p][0],s] / np.sum(counts[:4,counts_index[p][0],s], axis=0) #coverage = reads for csSNP allele/total reads
            cov_rev[p,s] =  counts[int(alleles[p]+3),counts_index[p][0],s] / np.sum(counts[4:,counts_index[p][0],s], axis=0)
            
    #filter for nans (0/0 counts) before moving on
    #cov_fwd[np.isnan(cov_fwd)] = 0; 
    #cov_rev[np.isnan(cov_rev)] = 0;
    
    #Initialize arrays for filtering and frequency calling
    #Make a 2d lists size sxc with only pos we want to consider, removing pos too far from the peak of distribution
    filtered_pos = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    filtered_counts_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    filtered_counts_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    filtered_cov_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    filtered_cov_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    
    #output data structure
    frequencies = np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    
    for s in range(np.size(counts_fwd,axis=1)):

        for c in range(np.size(CSS_matrix, axis=1)):
            
            #grab fwd and rev counts for every sample, cluster pair
            SC_counts_fwd = counts_fwd[np.where(clusters==c),s][0]
            SC_counts_rev = counts_rev[np.where(clusters==c),s][0]
            SC_cov_fwd = cov_fwd[np.where(clusters==c),s][0]
            SC_cov_rev = cov_rev[np.where(clusters==c),s][0]
            #SC_pos = CSS_pos[np.where(clusters==c)]
                
            # if 0s aren't expected (coverage is high enough), take nonzero mean. Otherwise take mean with zeros
            if np.mean(np.concatenate((SC_counts_fwd,SC_counts_rev))) > 6:
                SC_counts_mean = np.nanmean(np.concatenate((SC_counts_fwd[np.nonzero(SC_counts_fwd)],
                                                            SC_counts_rev[np.nonzero(SC_counts_rev)])))
            else:
                SC_counts_mean = np.nanmean(np.concatenate((SC_counts_fwd,SC_counts_rev)))
                
            # filter out positions that are not within mean +/2SD mean counts for s,c pair
            poiss_expect = 2*math.sqrt(SC_counts_mean)
            
            if poiss_expect > 1:
                fwd_filter_bool = (SC_counts_fwd > SC_counts_mean - poiss_expect) & (SC_counts_fwd < SC_counts_mean + poiss_expect) 
                rev_filter_bool = (SC_counts_rev > SC_counts_mean - poiss_expect) & (SC_counts_rev < SC_counts_mean + poiss_expect)
            else:
                fwd_filter_bool = (SC_counts_fwd > SC_counts_mean - 1) & (SC_counts_fwd < SC_counts_mean + 1) 
                rev_filter_bool = (SC_counts_rev > SC_counts_mean - 1) & (SC_counts_rev < SC_counts_mean + 1)
            
            #filtered_pos[s][c] = SC_pos[fwd_filter_bool & rev_filter_bool]
            filtered_counts_fwd[s][c] = SC_counts_fwd[fwd_filter_bool & rev_filter_bool]
            filtered_counts_rev[s][c] = SC_counts_rev[fwd_filter_bool & rev_filter_bool]
            filtered_cov_fwd[s][c] = SC_cov_fwd[fwd_filter_bool & rev_filter_bool]
            filtered_cov_rev[s][c] = SC_cov_rev[fwd_filter_bool & rev_filter_bool]
            
            #to have a frequency you must have @ least 3 pos with positive counts in both f and r reads
            if (np.size(filtered_counts_fwd[s][c]) > min_snps) & (np.size(filtered_counts_rev[s][c]) > min_snps):

                frequencies[s,c] = np.nanmean(np.concatenate((filtered_cov_fwd[s][c], filtered_cov_rev[s][c])))
           
            else:
                
                frequencies[s,c] = 0.
    
    #Output everything as nice dataframes
    filtered_counts_coverage = [filtered_counts_fwd, filtered_counts_rev, filtered_cov_fwd, filtered_cov_rev]
    
    frequencies[np.isnan(frequencies)] = 0
    sample_frequencies = pd.DataFrame(frequencies)
    sample_frequencies.columns = clade_names
    
    filtered_pos = []
    return filtered_counts_coverage, filtered_pos, sample_frequencies
    
#Read in candidate mutation table and output variables as individual arrays
def read_counts_table(matlab_file, counts, pos, sample_names,quals=None):
    # Note: SampleNames == list[]
    
    file = h5py.File(matlab_file)
    # read data2dict
    arrays = {}
    for k, v in file.items():
        arrays[k] = np.array(v)
    # assign various variables. NOTE: import transposes arrays (only x/y but not z axis)
    
    p = arrays[pos] # Note: p is 1-based
    p = p.flatten().astype(np.int64)
    
    counts_mat = arrays[counts] #[level,row,col] > in matlab: level=sample,row=ATGCatgc, col=pos; NOTE: Arolyns PDF 3d object wrong axis!
    if len(counts_mat.shape) != 3: # only prints correct output when used after indel_counter implemented
        print('Attention: counts 3D matrix is not 3D. Skip subject!')
    counts_mat = counts_mat.transpose(2, 1, 0) # transpose: (level==; row>col; col>row)
    
    # SampleNames saved in specific object pointer format which requires this technical loop below to be resolved
    sampleNames = []
    mygroup = file[sample_names]
    
    for s in mygroup:
        obj=file[s[0]]
        str1 = ''.join([chr(i[0]) for i in obj])
        sampleNames.append(str1)
    
    if quals != None:
        quals_ = arrays[quals]
        quals_ = quals_.transpose()
        return [counts_mat, p, quals_, sampleNames]
    
    return [counts_mat, p, sampleNames]

def read_calls_table(matlab_file, calls, pos):
    #Read in candidate mutation table and output variables as individual arrays
    # Note: args are string names of variables
    
    file = h5py.File(matlab_file)
    # read data2dict
    arrays = {}
    for k, v in file.items():
        arrays[k] = np.array(v)
    # assign various variables. NOTE: import transposes arrays (only x/y but not z axis)
    
    calls_table = arrays[calls]
    calls_table = calls_table.transpose().astype(np.int64)
    
    p = arrays[pos] # Note: p is 1-based
    p = p.flatten().astype(np.int64)
    # SampleNames saved in specific object pointer format which requires this technical loop below to be resolved
    #sampleNames = []
    #mygroup = file['SampleNames']
    #for s in mygroup:
    #    obj=file[s[0]]
    #    str1 = ''.join([chr(i[0]) for i in obj])
    #    sampleNames.append(str1)

    return [calls_table, p]

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
    
    # MATLAB conversion to NATCG=01234
    # !Important! This is required for current ver of find_clade_specific_snps
    # as of 3/26/22; Want to change later
    majorNT=majorNT+1
    minorNT=minorNT+1
    
    return majorNT, maf, minorNT, minorAF

def counts_CSS_ZINB(counts, pos, sample_names,
                    CSS_matrix, CSS_pos, clade_names,
                    min_snps, do_filter=False, model='zinbinom'):
    '''
    Args:
        counts (arr): Counts object from CMT (8 x p x s).
        pos (arr): Corresponding positions on the reference for counts (p x 1).
        sample_names (list): List of sample names.
        CSS_matrix (arr): p x c matrix giving csSNPs (01234=NATCG).
        CSS_pos (arr): Corresponding positions on the reference for csSNPs.
        clade_names (list): List of all clade names.
        min_snps (int): Minimum csSNPs with non-zero counts in sample to fit a model.
        do_filter (bool, optional): Filter positions with high or low counts. Defaults to False.
        model (str, optional): Regression model to fit counts data. Options are 'zipois', 'zinbinom'.

    Raises:
        Exception: DESCRIPTION.

    Returns:
        counts_coverage (TYPE): DESCRIPTION.
        filter_counts_coverage (TYPE): DESCRIPTION.
        fit_info (TYPE): DESCRIPTION.
        sample_frequencies (TYPE): DESCRIPTION.

    '''
    
    if len(np.shape(CSS_pos)) > 1:
        raise Exception('Structure for CSS_pos incorrect')
        
    # =========================================================================
    #     Make some data structures to index through arrays & store results
    # =========================================================================
    
    # List of positions with a csSNP   
    pos_with_CSS = CSS_pos[np.where(CSS_matrix != 0)[0]]
    # List of 'informative positions' with csSNP and candidate mutation
    counts_CSS_positions = [p for p in pos_with_CSS if p in pos] 
    
    #corresponding index on CSS_matrix
    #I can't find a good solution to this, currently using weird searchsorted method
    CSS_pos_sorted = np.argsort(CSS_pos)
    counts_CSS_positions_pos = np.searchsorted(CSS_pos[CSS_pos_sorted], 
                                               counts_CSS_positions)
    counts_CSS_index = CSS_pos_sorted[counts_CSS_positions_pos]
    #counts_CSS_index = np.where(CSS_pos == counts_CSS_positions)[0] old version
    #counts_CSS_index = np.nonzero(CSS_matrix)[0]
    
    if len(counts_CSS_positions) != len(pos_with_CSS):
        raise Exception('Counts and csSNP positions do not match. Check that references are same and that csSNP positions are being included during the case step')

    #Grab cluster and allele info for each informative alleles
    clusters = np.nonzero(CSS_matrix[np.unique(counts_CSS_index)])[1]
    alleles = CSS_matrix[counts_CSS_index,clusters]
    
    #location of informative pos on counts matrix        
    counts_index=[ np.where(np.isin(pos,p))[0] for p in counts_CSS_positions if p in pos]
    
    #Initialize arrays to grab raw csSNP data
    counts_fwd=np.empty([len(pos_with_CSS),len(sample_names)]) 
    counts_rev=np.empty([len(pos_with_CSS),len(sample_names)])
    total_fwd=np.empty([len(pos_with_CSS),len(sample_names)])
    total_rev=np.empty([len(pos_with_CSS),len(sample_names)])
    cov_fwd=np.empty([len(pos_with_CSS),len(sample_names)])
    cov_rev=np.empty([len(pos_with_CSS),len(sample_names)])
    
    #Initialize output data structures
    frequencies=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    counts_mu_arr=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    counts_alpha_arr=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    counts_pi_arr=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    total_mu_arr=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    total_pi_arr=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    total_alpha_arr=np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])

    save_counts_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    save_counts_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    save_total_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    save_total_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    save_cov_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    save_cov_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    
    if do_filter:
        filtered_pos = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        filtered_counts_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        filtered_counts_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        filtered_total_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        filtered_total_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        filtered_cov_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        filtered_cov_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
        
    # =========================================================================
    #     Grab counts info for every csSNP position and allele
    # =========================================================================

    np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 warnings
    for s in range(len(sample_names)):
        for p in range(len(pos_with_CSS)):
            counts_fwd[p,s]=counts[int(alleles[p]-1),counts_index[p][0],s]
            counts_rev[p,s]=counts[int(alleles[p]+3),counts_index[p][0],s]
            total_fwd[p,s]=np.sum(counts[:4,counts_index[p][0],s], axis=0)
            total_rev[p,s]=np.sum(counts[4:,counts_index[p][0],s], axis=0)
            cov_fwd[p,s]=counts_fwd[p,s]/total_fwd[p,s]
            cov_rev[p,s]=counts_rev[p,s]/total_rev[p,s]
                
    # =========================================================================
    #     For each sample, clade - fit data to expected counts model
    # =========================================================================
    
    #First, reshape data to group csSNPs by clade
    for s in range(np.size(counts_fwd,axis=1)):
        for c in range(np.size(CSS_matrix, axis=1)):
            #grab fwd and rev counts for every sample, cluster pair
            SC_counts_fwd = counts_fwd[np.where(clusters==c),s][0]
            SC_counts_rev = counts_rev[np.where(clusters==c),s][0]
            SC_total_fwd = total_fwd[np.where(clusters==c),s][0]
            SC_total_rev = total_rev[np.where(clusters==c),s][0]
            SC_cov_fwd = cov_fwd[np.where(clusters==c),s][0]
            SC_cov_rev = cov_rev[np.where(clusters==c),s][0]
            
            # SC_counts = np.concatenate((SC_counts_fwd,SC_counts_rev))
            # SC_total = np.concatenate((SC_total_fwd,SC_total_rev))
            # SC_cov = np.concatenate((SC_cov_fwd,SC_cov_rev))
            SC_pos = np.array(pos_with_CSS)[np.where(clusters==c)]
            
            ## FILTERING ##
            if do_filter:

                # Remove positions with 95th percentile depth of coverage
                total_fwd_95th=np.percentile(SC_total_fwd[np.nonzero(SC_total_fwd)], 95)
                total_rev_95th=np.percentile(SC_total_rev[np.nonzero(SC_total_rev)], 95)
                                
                fwd_filter_bool = (SC_counts_fwd < total_fwd_95th) 
                rev_filter_bool = (SC_counts_rev < total_rev_95th)
                
                #Make a 2d lists size sxc with only pos we want to consider
                filtered_pos[s][c] = SC_pos[fwd_filter_bool & rev_filter_bool]
                save_counts_fwd[s][c] = SC_counts_fwd[fwd_filter_bool & rev_filter_bool]
                save_counts_rev[s][c] = SC_counts_rev[fwd_filter_bool & rev_filter_bool]
                save_total_fwd[s][c] = SC_total_fwd[fwd_filter_bool & rev_filter_bool]
                save_total_rev[s][c] = SC_total_rev[fwd_filter_bool & rev_filter_bool]
                save_cov_fwd[s][c] = SC_cov_fwd[fwd_filter_bool & rev_filter_bool]
                save_cov_rev[s][c] = SC_cov_rev[fwd_filter_bool & rev_filter_bool]
                
            else: # no filtering
                save_counts_fwd[s][c] = SC_counts_fwd
                save_counts_rev[s][c] = SC_counts_rev
                save_total_fwd[s][c] = SC_total_fwd
                save_total_rev[s][c] = SC_total_rev
                save_cov_fwd[s][c] = SC_cov_fwd
                save_cov_rev[s][c] = SC_cov_rev
            
            # To have a frequency, a sample must have min_SNP positions with 
            # nonzero counts in either fwd OR rev reads
            # To implement: If filter, at least 70% of positions can't be filtered out
                # (len(filtered_pos[s][c])/len(SC_pos) < 0.9)
            if np.count_nonzero(save_counts_fwd[s][c] + save_counts_rev[s][c]) > min_snps:
                
                cts2model=np.concatenate((save_counts_fwd[s][c], save_counts_rev[s][c]))
                total2model=np.concatenate((save_total_fwd[s][c], save_total_rev[s][c]))

                
                print(f"Fit results for Sample: {sample_names[s]} Clade: {clade_names[c]}")
                #Fit to chosen model
                if model=='zinbinom':
                    counts_mu, counts_alpha, counts_pi = zinbinom_fit(cts2model)
                    total_mu, total_alpha, total_pi = zinbinom_fit(total2model)
                    
                    #save fit info
                    counts_mu_arr[s,c]=counts_mu
                    counts_alpha_arr[s,c]=counts_alpha
                    counts_pi_arr[s,c]=counts_pi
                    total_mu_arr[s,c]=total_mu
                    total_pi_arr[s,c]=total_pi
                    total_alpha_arr[s,c]=total_alpha
                    
                    #if pi is too high, count the clade as not present
                    if total_pi < 0.2 and counts_pi-total_pi < 0.2: # and total_alpha > 2:
                        frequencies[s,c] = counts_mu/total_mu
                    else:
                        print("Pi parameter exceeds allowed values")
                        frequencies[s,c] = 0.
                
                if model=='zipois':
                    counts_lamb, counts_pi, counts_lamb_CI = zip_fit(cts2model)
                    total_lamb, total_pi, total_lamb_CI = zip_fit(total2model)
                
                    #save fit info
                    counts_mu_arr[s,c]=counts_lamb
                    counts_pi_arr[s,c]=counts_pi
                    total_mu_arr[s,c]=total_lamb
                    total_pi_arr[s,c]=total_pi
                    
                    #if pi is too high, count the clade as not present
                    if total_pi < 0.2 and counts_pi-total_pi < 0.2:
                        frequencies[s,c] = counts_lamb/total_lamb
                    else:
                        frequencies[s,c] = 0.
            
            #Otherwise is zero
            else:
                print(f"Sample: {sample_names[s]} Clade: {clade_names[c]} not enough SNPs to model")
                frequencies[s,c] = 0.
                
    #Output everything as nice data structures
    frequencies[np.isnan(frequencies)] = 0
    sample_frequencies = pd.DataFrame(frequencies)
    sample_frequencies.columns = clade_names
    sample_frequencies.index=sample_names

    if do_filter:
        counts_coverage = [save_counts_fwd, save_counts_rev, save_total_fwd,save_total_rev,save_cov_fwd,save_cov_rev,filtered_pos]
    else:
        counts_coverage = [save_counts_fwd, save_counts_rev, save_total_fwd,save_total_rev,save_cov_fwd,save_cov_rev]
    fit_info = [counts_mu_arr,counts_alpha_arr,counts_pi_arr,total_mu_arr,total_alpha_arr,total_pi_arr]

    return counts_coverage, fit_info, sample_frequencies

def zinb_pmf(x, mu, alpha_, pi):
    
    #scipy will only take n and p as shape parameters, need to convert
    #n = number of successes
    #p = probability of success
    n_ = alpha_
    p_ =  alpha_/(alpha_+mu)

    if pi < 0 or pi > 1 or mu <= 0:
        return np.zeros_like(x)
    else:
        return ((x == 0) * pi) + ((1 - pi) * stats.nbinom.pmf(x, n_, p_))

def nb_pmf(x, mu, alpha_):
    
    #scipy will only take n and p as shape parameters, need to convert
    #n = number of successes
    #p = probability of success
    n_ = alpha_
    p_ =  alpha_/(alpha_+mu)
    
    return stats.nbinom.pmf(x, n_, p_)

class ZeroInflatedNB(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(ZeroInflatedNB, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        mu = params[0]
        alpha_ = params[1]
        pi = params[2]
        # alph = params[-1]
        # beta = params[:-1]
        # ll = _ll_nb2(self.endog, self.exog, beta, alph)
        nll = -np.log(zinb_pmf(self.endog, mu=mu, alpha_=alpha_, pi=pi))
        
        return nll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        # self.exog_names.append('alpha')
        if start_params == None:
            # Reasonable starting values
            mu_start = self.endog.mean()
            alpha_start = 1/(((self.endog.std()**2/mu_start)-1) / mu_start)
            pi_start = (self.endog == 0).mean() - nb_pmf(0, mu_start, alpha_start)
            
            start_params = np.array([mu_start, alpha_start, pi_start])

        return super(ZeroInflatedNB, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
def zinbinom_fit(counts):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedNB(counts)
        fit = model.fit()
        mu, alpha_, pi = fit.params
        
        if np.isnan(mu):
            print("Warning: mu parameter returned NaN value")
            mu=0
        if np.isnan(alpha_):
            print("Warning: alpha parameter returned NaN value")
            alpha_=0
        if np.isnan(pi):
            print("Warning: pi parameter returned NaN value")
            pi=0
        
    return abs(mu), abs(alpha_), abs(pi)

def zip_pmf(x, pi, lambda_):
    '''zero-inflated poisson function, pi is prob. of 0, lambda_ is the fit parameter'''
    if pi < 0 or pi > 1 or lambda_ <= 0:
        return np.zeros_like(x)
    else:
        return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)

class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]

        return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            lambda_start = self.endog.mean()
            excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            
            start_params = np.array([excess_zeros, lambda_start])
            
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)
def zip_fit(data):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedPoisson(data)
        fit = model.fit()
        pi, lambda_ = fit.params
    
    # return lambda_, pi
        CI_pi, CI_lambda_ = fit.conf_int()
        # range_CI_pi = CI_pi[1] - CI_pi[0]
        range_CI_lambda_ = CI_lambda_[1] - CI_lambda_[0]
    
    return lambda_, pi, range_CI_lambda_

'''
OLD csSNP calling code - dont use this
'''
# def find_cluster_mutations_counts(maNT, maNT_pos, cluster, core):
#     #For a dataset of isolates, get allele frequencies across clades for every position
#         # Input:
#             #maNT: pxiso matrix with major allele nt (maNT) calls for every isolate
#             #maNT_pos: px1 matrix with reference pos for every maNT call
#             #cluster: 1xiso list mapping each isolate to a clade.
#             #core: How prevalent a pos must be across isolates to be included (default: 0.9)
            
#         # Output:    
#             #maNT_frequencies: matrix that is positions (p) long and clusters (c) wide.
#                 #Each p,c is a 5-long set (N,A,T,G,C) that has % major nt freq for each isolate 
#                 #in the cluster. 5 numbers in set should add up to 1
#             #core_maNT_pos: updated maNT_pos excluding positions that were trimmed
    
#     #pull out only positions that are part of the core genome
#     core_maNT = maNT[np.count_nonzero(maNT, axis=1)/len(maNT[1]) > core]
#     core_maNT_pos = maNT_pos[np.count_nonzero(maNT, axis=1)/len(maNT[1]) > core]
    
#     maNT_frequencies=np.empty([len(core_maNT),np.max(cluster)+1,5]) # NOTE: code getting the # of clusters is ugly

#     for c in range(np.max(cluster)+1): #To do: change this to range(unique values of cluster)
        
#         maNT_bycluster = core_maNT[:,np.where(cluster==c)[1]]
#         # slice out all isolates in cluster c
        
#         for p in range(len(maNT_bycluster)):
            
#             maNT_byposition = maNT_bycluster[p,:]
            
#             for n in range(5):
#                 maNT_frequencies[p,c,n] = np.mean(maNT_byposition==n)
    
#     return maNT_frequencies, core_maNT_pos

# def get_cluster_specific_snps(maNT_frequencies, n):
#     #Find cluster-specific SNPs (SNPs that are unique and unanimous to a clade)
#         #Input: 
#             #maNT_frequencies: pxcx5 matrix containing within-cluster nucleotide frequencies
#             #n: What frequency of Ns are tolerated in calling a csSNP (default 0.1)
#         #Output: 
#             #CSS_matrix: pxc matrix, each p,c is ATCG if unique SNP to a cluster, 0 otherwise

#     # Find positions where all isolates in ONE cluster have the same allele (unanimous) AND
#     # only one cluster has any isolates with this allele (unique)
#     unanimous_to_cluster = (maNT_frequencies >= 1-n)
#     just_one_cluster = (np.sum(maNT_frequencies>0, axis=1) == 1)[:,np.newaxis,:]
    
#     # tolerate a certain number of Ns to call an allele as 'unanimous'
#     n_tolerance = (maNT_frequencies[:,:,0] <= n)[:,:,np.newaxis]
    
#     #combine into 3D matrix
#     CSS_matrix_3d = (unanimous_to_cluster & 
#                      np.tile(just_one_cluster, (1,maNT_frequencies.shape[1],1)) & 
#                      np.tile(n_tolerance, (1,1,maNT_frequencies.shape[2])) )
    
#     # flatten 3D matrix into 2D pxc matrix with ATCG where a unique and unanimous allele is present, 0 otherwise
#     CSS_matrix = np.zeros((len(maNT_frequencies), len(maNT_frequencies[1])))
    
#     for p,c in zip(CSS_matrix_3d.nonzero()[0],CSS_matrix_3d.nonzero()[1]):
#         # for every pos on 3d matrix with a 1, get just the width index
#         CSS_matrix[p,c] = (CSS_matrix_3d[p,c].nonzero()[0])[0]
    
#     return CSS_matrix.astype(int)

# #OLD counts_CSS before snakemake
# def counts_CSS(counts,pos,clade_names,CSS_matrix,CSS_pos,min_snps):
#     '''
#     #Input: 
#         #counts: output from case step (8 x p x s)
#         #pos: corresponding positions on the reference for counts (p x 1)
#         #clade_names: str list of all clade names. 
#         #CSS_matrix: p x c matrix of clade specific SNPs (1234 if ATCG, 0 otherwise)
#         #CSS_pos: positions on the reference for CSSes (same as core_pos)
#         #min_snps: minimum num of csSNPs with non-zero counts to assign a clade as present in a sample (default 3)
#     '''
    
#     if len(np.shape(CSS_pos)) > 1:
#         raise Exception('Structure for CSS_pos incorrect')
    
#     # Pull out 'informative positions': positions with both a candidate mutations and a csSNP   
#     pos_with_CSS = CSS_pos[np.where(CSS_matrix != 0)[0]] 
#     counts_CSS_positions = [p for p in pos_with_CSS if p in pos] # list of informative positions with csSNP and CM
    
#     #corresponding index on CSS_matrix
#     #I can't find a good solution to this - currently using weird searchsorted method
#     CSS_pos_sorted = np.argsort(CSS_pos); counts_CSS_positions_pos = np.searchsorted(CSS_pos[CSS_pos_sorted], counts_CSS_positions)
#     counts_CSS_index = CSS_pos_sorted[counts_CSS_positions_pos]
#         #counts_CSS_index = np.where(CSS_pos == counts_CSS_positions)[0] old version
#         #counts_CSS_index = np.nonzero(CSS_matrix)[0]
    
#     if len(counts_CSS_positions) != len(pos_with_CSS):
#         raise Exception('Counts and csSNP positions do not match. Check that references are same and that csSNP positions are being included during the case step')
        
#     #Grab cluster and allele info for each informative alleles
#     clusters = np.nonzero(CSS_matrix[np.unique(counts_CSS_index)])[1]
#     alleles = CSS_matrix[counts_CSS_index,clusters]
    
#     #location of informative pos on counts matrix        
#     counts_index=[ np.where(np.isin(pos,p))[0] for p in counts_CSS_positions if p in pos]
#         #my original line: counts_index=np.where(np.isin(pos,counts_CSS_positions))[0] got deprecated (7/25)
#         #sample_mode = stats.mode(np.sum(counts[:,:,:],axis=0),axis=0)[0].flatten() #mode counts of each sample
#         #sample_mean = np.mean(np.sum(counts[:,:,:],axis=0),axis=0)
#         #sample_std = np.std(np.sum(counts[:,:,:],axis=0),axis=0)
    
    
#     #Initialize a bunch of arrays for next step
#     counts_fwd=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)]); counts_rev=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)])
#     cov_fwd=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)]); cov_rev=np.empty([len(counts_CSS_positions),np.size(counts, axis=2)])

#     # Grab # of counts supporting the csSNP allele and frac of total reads for informative positions
#     np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 warnings
#     for s in range(np.size(counts, axis=2)):
#         for p in range(len(counts_CSS_positions)):
#             counts_fwd[p,s] = counts[int(alleles[p]-1),counts_index[p][0],s]
#             counts_rev[p,s] = counts[int(alleles[p]+3),counts_index[p][0],s]
#             cov_fwd[p,s] = counts[int(alleles[p]-1),counts_index[p][0],s] / np.sum(counts[:4,counts_index[p][0],s], axis=0) #coverage = reads for csSNP allele/total reads
#             cov_rev[p,s] =  counts[int(alleles[p]+3),counts_index[p][0],s] / np.sum(counts[4:,counts_index[p][0],s], axis=0)
            
#     #filter for nans (0/0 counts) before moving on
#     #cov_fwd[np.isnan(cov_fwd)] = 0; 
#     #cov_rev[np.isnan(cov_rev)] = 0;
    
#     #Initialize arrays for filtering and frequency calling
#     #Make a 2d lists size sxc with only pos we want to consider, removing pos too far from the peak of distribution
#     filtered_pos = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
#     filtered_counts_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
#     filtered_counts_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
#     filtered_cov_fwd = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
#     filtered_cov_rev = [[0 for c in range(len(CSS_matrix[1]))] for s in range(np.size(counts, axis=2))]
    
#     #output data structure
#     frequencies = np.zeros([np.size(counts, axis=2) , np.size(CSS_matrix, axis=1)])
    
#     for s in range(np.size(counts_fwd,axis=1)):

#         for c in range(np.size(CSS_matrix, axis=1)):
            
#             #grab fwd and rev counts for every sample, cluster pair
#             SC_counts_fwd = counts_fwd[np.where(clusters==c),s][0]
#             SC_counts_rev = counts_rev[np.where(clusters==c),s][0]
#             SC_cov_fwd = cov_fwd[np.where(clusters==c),s][0]
#             SC_cov_rev = cov_rev[np.where(clusters==c),s][0]
#             SC_pos = CSS_pos[np.where(clusters==c)]
                
#             # if 0s aren't expected (coverage is high enough), take nonzero mean. Otherwise take mean with zeros
#             if np.mean(np.concatenate((SC_counts_fwd,SC_counts_rev))) > 6:
#                 SC_counts_mean = np.nanmean(np.concatenate((SC_counts_fwd[np.nonzero(SC_counts_fwd)],
#                                                             SC_counts_rev[np.nonzero(SC_counts_rev)])))
#             else:
#                 SC_counts_mean = np.nanmean(np.concatenate((SC_counts_fwd,SC_counts_rev)))
                
#             # filter out positions that are not within mean +/2SD mean counts for s,c pair
#             poiss_expect = 2*math.sqrt(SC_counts_mean)
            
#             if poiss_expect > 1:
#                 fwd_filter_bool = (SC_counts_fwd > SC_counts_mean - poiss_expect) & (SC_counts_fwd < SC_counts_mean + poiss_expect) 
#                 rev_filter_bool = (SC_counts_rev > SC_counts_mean - poiss_expect) & (SC_counts_rev < SC_counts_mean + poiss_expect)
#             else:
#                 fwd_filter_bool = (SC_counts_fwd > SC_counts_mean - 1) & (SC_counts_fwd < SC_counts_mean + 1) 
#                 rev_filter_bool = (SC_counts_rev > SC_counts_mean - 1) & (SC_counts_rev < SC_counts_mean + 1)
            
#             filtered_pos[s][c] = SC_pos[fwd_filter_bool & rev_filter_bool]
#             filtered_counts_fwd[s][c] = SC_counts_fwd[fwd_filter_bool & rev_filter_bool]
#             filtered_counts_rev[s][c] = SC_counts_rev[fwd_filter_bool & rev_filter_bool]
#             filtered_cov_fwd[s][c] = SC_cov_fwd[fwd_filter_bool & rev_filter_bool]
#             filtered_cov_rev[s][c] = SC_cov_rev[fwd_filter_bool & rev_filter_bool]
            
#             #to have a frequency you must have @ least 3 pos with positive counts in both f and r reads
#             if (np.size(filtered_counts_fwd[s][c]) > min_snps) & (np.size(filtered_counts_rev[s][c]) > min_snps):

#                 frequencies[s,c] = np.nanmean(np.concatenate((filtered_cov_fwd[s][c], filtered_cov_rev[s][c])))
           
#             else:
                
#                 frequencies[s,c] = 0.
    
#     #Output everything as nice dataframes
#     filtered_counts_coverage = [filtered_counts_fwd, filtered_counts_rev, filtered_cov_fwd, filtered_cov_rev]
    
#     frequencies[np.isnan(frequencies)] = 0
#     sample_frequencies = pd.DataFrame(frequencies)
#     sample_frequencies.columns = clade_names
    
#     return filtered_counts_coverage, filtered_pos, sample_frequencies
