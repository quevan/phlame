'''
Inputs:
    path_to_samples

Outputs:
    filtering_stats.csv
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
                            Generates QC stats on the cutadapt and sickle rules
                                ''',
                               epilog="Questions or comments? --> aholton@mit.edu")
parser.add_argument("-s", dest="samples", help="csv file with ",required=True,action='store')
parser.add_argument("-d", dest="currentDirectory", help="current directory of Snakefile", required=True, action="store")
args = parser.parse_args()

def main(path_to_samples, current_directory):

    cwd = current_directory
    home_dir = cwd.split("/")[-1]

    # make copy of samples.csv to add to
    print("Copying samples csv...")
    shutil.copyfile(cwd + "/" + path_to_samples, cwd + "/1-data_processed/filtering_stats.csv")

    # make new columns in csv for the alignment stats
    filtering_stats = pd.read_csv(cwd + "/1-data_processed/filtering_stats.csv")
    filtering_stats = filtering_stats.astype(str)
    filtering_stats["Paired Kept"] = ""
    filtering_stats["Paired Discarded"] = ""
    filtering_stats["Percent Reads With Adapters"] = ""
    filtering_stats["Reads Passed Filters"] = ""
    filtering_stats["Percent Bp Written Filtered"] = ""
    
    # grab relevant info from log files
    os.system("grep pairs logs/*sickle2050_* >> " + cwd + "/1-data_processed/filtering_stats.txt")
    os.system("grep -w \"with adapters\" logs/*cutadapt_* >> " + cwd + "/1-data_processed/filtering_stats.txt")
    os.system("grep filters logs/*cutadapt_* >> " + cwd + "/1-data_processed/filtering_stats.txt") 
    os.system("grep filtered logs/*cutadapt_* >> " + cwd + "/1-data_processed/filtering_stats.txt") 

    sample_id_to_paired_kept        = {}        
    sample_id_to_paired_discarded   = {}
    sample_id_to_percent_adapters   = {}
    sample_id_to_reads_passed       = {}        
    sample_id_to_percent_bp_written = {}

    stats_file = open(cwd + "/1-data_processed/filtering_stats.txt")
    for line in stats_file:
        split_on_rule_name = []
        if ("sickle" in line):
            split_on_rule_name = line.split("sickle2050_")
        elif ("cutadapt" in line):
            split_on_rule_name = line.split("cutadapt_")
        split_on_txt = split_on_rule_name[1].split(".txt")
        sample_id = split_on_txt[0]
        split_on_colon = split_on_txt[1].split(":")
        stat_description = split_on_colon[1]
        stat_value_info = split_on_colon[2]
        if "pairs" in stat_value_info:
            value = stat_value_info.split("(")[0]
            if "kept" in stat_description:
                sample_id_to_paired_kept[sample_id] = int(value)
            else:
                sample_id_to_paired_discarded[sample_id] = int(value)
        elif "adapters" in stat_description:
            value = float(stat_value_info.split()[1].split("(")[1].replace(")", "").replace("%", "").replace(",", ""))
            if (sample_id in sample_id_to_percent_adapters):
                sample_id_to_percent_adapters[sample_id] = sample_id_to_percent_adapters[sample_id] + value
            else:    
                sample_id_to_percent_adapters[sample_id] = value
        elif "filters" in stat_description:
            value = int(stat_value_info.split()[0].replace(",", ""))
            if (sample_id in sample_id_to_reads_passed):
                sample_id_to_reads_passed[sample_id] = sample_id_to_reads_passed[sample_id] + value
            else:    
                sample_id_to_reads_passed[sample_id] = value
        elif "filtered" in stat_description:
            value = float(stat_value_info.split()[2].replace("(", "").replace(")", "").replace("%", "").replace(",", ""))
            if (sample_id in sample_id_to_percent_bp_written):
                sample_id_to_percent_bp_written[sample_id] = sample_id_to_percent_bp_written[sample_id] + value
            else:    
                sample_id_to_percent_bp_written[sample_id] = value

    for _, row in filtering_stats.iterrows():
        sample_id_from_csv = row["Sample"]
        if sample_id_from_csv in sample_id_to_paired_kept:
	        row["Paired Kept"] = sample_id_to_paired_kept[sample_id_from_csv]
	        row["Paired Discarded"] = sample_id_to_paired_discarded[sample_id_from_csv]
        else:
            row["Paired Kept"] = "-1"
            row["Paired Discarded"] = "-1"
        if sample_id_from_csv in sample_id_to_percent_adapters:
	        row["Percent Reads With Adapters"] = sample_id_to_percent_adapters[sample_id_from_csv]
	        row["Reads Passed Filters"] = sample_id_to_reads_passed[sample_id_from_csv]
	        row["Percent Bp Written Filtered"] = sample_id_to_percent_bp_written[sample_id_from_csv]
        else:
	        row["Percent Reads With Adapters"] = "-1"
	        row["Reads Passed Filters"] = "-1"
	        row["Percent Bp Written Filtered"] = "-1"
    
    # for each stat, make a histogram across the samples
    number_samples = filtering_stats.shape[0]
    number_bins = math.ceil(math.sqrt(number_samples))

    # paired kept histogram
    paired_kept = filtering_stats["Paired Kept"].tolist()
    paired_kept = [int(number) for number in paired_kept]
    paired_kept = np.array(paired_kept)
    hist, bin_edges = np.histogram(paired_kept, bins=number_bins)
    plt.hist(paired_kept, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Number of Paired Reads Kept")
    plt.xlabel("Number of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/1-data_processed/" + home_dir + "_paired_kept_histogram.png")

    # paired discarded histogram
    paired_discarded = filtering_stats["Paired Discarded"].tolist()
    paired_discarded = [int(number) for number in paired_discarded]
    paired_discarded = np.array(paired_discarded)
    hist, bin_edges = np.histogram(paired_discarded, bins=number_bins)
    plt.clf()
    plt.hist(paired_discarded, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Number of Paired Reads Discarded")
    plt.xlabel("Number of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/1-data_processed/" + home_dir + "_paired_discarded_histogram.png")

    # percent reads with adapters histogram
    percent_adapters = filtering_stats["Percent Reads With Adapters"].tolist()
    percent_adapters = [float(percent) for percent in percent_adapters]
    percent_adapters = np.array(percent_adapters)
    hist, bin_edges = np.histogram(percent_adapters, bins=number_bins)
    plt.clf()
    plt.hist(percent_adapters, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Percent Paired Reads With Adapters")
    plt.xlabel("Percent of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/1-data_processed/" + home_dir + "_percent_adapters_histogram.png")

    # reads passed filters histogram
    reads_passed = filtering_stats["Reads Passed Filters"].tolist()
    reads_passed = [int(number) for number in reads_passed]
    reads_passed = np.array(reads_passed)
    hist, bin_edges = np.histogram(reads_passed, bins=number_bins)
    plt.clf()
    plt.hist(reads_passed, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Paired Reads that Passed Filters")
    plt.xlabel("Number of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/1-data_processed/" + home_dir + "_reads_passed_histogram.png")

    # percent bp written filtered histogram
    percent_bp = filtering_stats["Reads Passed Filters"].tolist()
    percent_bp = [int(number) for number in percent_bp]
    percent_bp = np.array(percent_bp)
    hist, bin_edges = np.histogram(percent_bp, bins=number_bins)
    plt.clf()
    plt.hist(percent_bp, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Percent of Base Pairs Written that Passed Filters")
    plt.xlabel("Percent of Base Pairs")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/1-data_processed/" + home_dir + "_percent_bp_histogram.png")

    # save the csv file in the cutadapt/sickle step folder
    path_to_save_csv = cwd + "/1-data_processed/filtering_stats.csv"
    filtering_stats.to_csv(path_to_save_csv, index = False)
    print("Done with read filtering QC")

if __name__ == "__main__":
    path_to_samples=args.samples
    current_directory = args.currentDirectory
    main(path_to_samples, current_directory)