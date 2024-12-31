'''
Inputs:
    paths_to_samples

Outputs:
    alignment_stats.csv
'''
import math
import os
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description='''\
                            Generates QC stats on the bowtie2 rule
                               ''',
                               epilog="Questions or comments? --> aholton@mit.edu")
parser.add_argument("-s", dest="samples", help="csv file with samples",required=True,action='store')
parser.add_argument("-d", dest="currentDirectory", help="current directory of Snakefile", required=True, action="store")
args = parser.parse_args()

def main(path_to_samples, current_directory):

    cwd = current_directory
    home_dir = cwd.split("/")[-1]

    # make copy of samples.csv to add to
    print("Copying samples csv...")
    shutil.copyfile(cwd + "/" + path_to_samples, cwd + "/3-bowtie2/alignment_stats.csv")

    # make new columns in csv for the alignment stats
    alignment_stats = pd.read_csv(cwd + "/3-bowtie2/alignment_stats.csv")
    alignment_stats = alignment_stats.astype(str)
    alignment_stats["Number Aligned Once"] = ""
    alignment_stats["Percent Aligned Once"] = ""
    alignment_stats["Percent Overall Alignment"] = ""
    
    # grab relevant info from log files
    os.system("grep exactly logs/*bowtie2_* >> " + cwd + "/3-bowtie2/alignment_stats.txt")
    os.system("grep overall logs/*bowtie2_* >> " + cwd + "/3-bowtie2/alignment_stats.txt") 

    sample_id_to_number_once  = {}        
    sample_id_to_percent_once = {}
    sample_id_to_overall      = {}

    stats_file = open(cwd + "/3-bowtie2/alignment_stats.txt")
    for line in stats_file:
        split_on_rule_name = line.split("bowtie2_")
        split_on_ref = split_on_rule_name[1].split("_ref_")
        sample_id = split_on_ref[0]
        split_on_colon = split_on_ref[1].split(":")
        stat_info = split_on_colon[1]
        if "exactly" in stat_info:
            info = stat_info.split()
            number = info[0]
            percent = info[1].replace("(", "").replace(")","").replace("%", "")
            sample_id_to_number_once[sample_id] = number
            sample_id_to_percent_once[sample_id] = percent
        elif "overall" in stat_info:
            info = stat_info.split()
            percent = info[0].replace("%", "")  
            sample_id_to_overall[sample_id] = percent
	
    for _, row in alignment_stats.iterrows():
        sample_id_from_csv = row["Sample"]
        if sample_id_from_csv in sample_id_to_number_once:
            row["Number Aligned Once"] = sample_id_to_number_once[sample_id_from_csv]
            row["Percent Aligned Once"] = sample_id_to_percent_once[sample_id_from_csv]
            row["Percent Overall Alignment"] = sample_id_to_overall[sample_id_from_csv]
        else:
            row["Number Aligned Once"] = "-1"
            row["Percent Aligned Once"] = "-1"
            row["Percent Overall Alignment"] = "-1"

    # for each stat, make a histogram across the samples
    number_samples = alignment_stats.shape[0]
    number_bins = math.ceil(math.sqrt(number_samples))
    # number aligned once histogram
    number_aligned_once = alignment_stats["Number Aligned Once"].tolist()
    number_aligned_once = [int(number) for number in number_aligned_once]
    number_aligned_once = np.array(number_aligned_once)
    hist, bin_edges = np.histogram(number_aligned_once, bins=number_bins)
    plt.hist(number_aligned_once, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Number of Paired Reads That Aligned Once")
    plt.xlabel("Number of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/3-bowtie2/" + home_dir + "_number_aligned_once_histogram.png")

    # percent aligned once histogram
    percent_aligned_once = alignment_stats["Percent Aligned Once"].tolist()
    percent_aligned_once = [float(percent) for percent in percent_aligned_once]
    percent_aligned_once = np.array(percent_aligned_once)
    hist, bin_edges = np.histogram(percent_aligned_once, bins=number_bins)
    plt.clf()
    plt.hist(percent_aligned_once, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Percent of Paired Reads That Aligned Once")
    plt.xlabel("Percent of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/3-bowtie2/" + home_dir + "_percent_aligned_once_histogram.png")

    # overall alignment histogram
    overall_alignment = alignment_stats["Percent Overall Alignment"].tolist()
    overall_alignment = [float(percent) for percent in overall_alignment]
    overall_alignment = np.array(overall_alignment)
    hist, bin_edges = np.histogram(overall_alignment, bins=number_bins)
    plt.clf()
    plt.hist(overall_alignment, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Overall Alignment Percent")
    plt.xlabel("Percent of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/3-bowtie2/" + home_dir + "_overall_alignment_histogram.png")

    # save the csv file in the bowtie2 step folder
    path_to_save_csv = cwd + "/3-bowtie2/alignment_stats.csv"
    alignment_stats.to_csv(path_to_save_csv, index = False)
    print("Done with bowtie2 QC")

if __name__ == "__main__":
    path_to_samples=args.samples
    current_directory=args.currentDirectory
    main(path_to_samples, current_directory)
