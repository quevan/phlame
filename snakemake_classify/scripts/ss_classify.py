#!/usr/bin/env python3

import os
import argparse
import pickle
import gzip
import glob
import pandas as pd
import strainslicer_current_version as strainslicer

#%% Testing
#%% Functions
def classify(path_to_cmt_file, path_to_classifier_file, path_to_output):
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
        counts, pos, sample_names = strainslicer.read_counts_table(path_to_cmt_file, 
                                                                   counts='counts', 
                                                                   pos='p', 
                                                                   sample_names='SampleNames')
    else:
        raise IOError("Error: candidate mutation table does not end in .mat or .pickle.gz!")
    
    print("Reading in classifier file...")
    with open(path_to_classifier_file, 'rb') as f:
        csSNPs, csSNP_pos, clade_names = pickle.load(f)
    
    print("Classifying...")
    cts_cov, fit_info, sample_frequencies = strainslicer.counts_CSS_ZINB(counts, 
                                                                        pos, 
                                                                        sample_names,
                                                                        csSNPs, 
                                                                        csSNP_pos,
                                                                        clade_names,
                                                                        10)
    #Save output
    sample_frequencies.to_csv(path_to_output, sep=',')
    
    return cts_cov, fit_info

#%% Main
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', dest='Input', type=str, help='Path to input MG candidate mutation table',required=True)
    parser.add_argument('-c', dest='Classifier', type=str, help='Path to classifier directory', required=True)
    parser.add_argument('-o', dest='Output', type=str, help='Path to output directory', required=True)

    args = parser.parse_args()
    
    CMT=args.Input
    
    if not os.path.isdir(args.Output):
        raise IOError("Error: Missing output directory!")
    
    print(args.Output+"/*.classifier")
    
    for cfr in glob.iglob(args.Classifier+"/*.classifier"):
        out_file=args.Output+'/'+os.path.basename(cfr).strip('.classifier')+'_frequencies.csv'
        print(out_file)
        cts_cov, fit_info = classify(CMT, cfr, out_file)
        
        with gzip.open(args.Output+'/'+os.path.basename(cfr).strip('.classifier')+'_frequencies.data','wb') as f:
            pickle.dump([cts_cov, fit_info],f)
