############################################
# PHLAME Snakefile (Classify)#
############################################

import sys
import os
import glob

###################### PRE-SNAKEMAKE ######################

# Global variables
CURRENT_DIR = os.getcwd()

spls = config["sample_table"]
SCRIPTS_DIR = config["myscripts_directory"]
REFGENOME_DIR = config["ref_genome_directory"]
sys.path.insert(0, SCRIPTS_DIR)

from phlame_SM_module import *

## Define couple of lists from samples.csv
## Format: Path, Sample, FileName, Classifier, Reference
[PATH_ls, SAMPLE_ls, FILENAME_ls, CLASSIFIER_ls, REF_GENOME_ls] = read_samplesCSV_classify(spls)

# Set up wildcards, write sample_info.csv for each sample
split_samplesCSV_classify(PATH_ls,SAMPLE_ls,FILENAME_ls,CLASSIFIER_ls,REF_GENOME_ls)

# Wishlist is to accept multiple classifiers and reference in the same SM
# Rn will only accept one classifier, one reference
assert len(set(REF_GENOME_ls)) == 1
assert len(set(CLASSIFIER_ls)) == 1

###################### SNAKEMAKE ######################

rule all:
	input:
		expand("4-frequencies/{sampleID}_ref_{reference}_frequencies.csv", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
		# expand("4-frequencies/{sampleID}_ref_{reference}_plot.pdf", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls))


rule make_data_links:
	# NOTE: All raw data needs to be named fastq.gz. No fq! 
	# The links will be named fq.
	input:
		sample_info_csv="data/{sampleID}/sample_info.csv",
	output:
		# Recommend using symbolic links to your likely many different input files
		fq1="data/{sampleID}/R1.fq.gz",
		fq2="data/{sampleID}/R2.fq.gz",
	run:
		# get stuff out of mini csv file
		with open(input.sample_info_csv,'r') as f:
			this_sample_info = f.readline() # only one line to read
		this_sample_info = this_sample_info.strip('\n').split(',')
		path = this_sample_info[0]
		path_ls = path.split(' ')
		sample = this_sample_info[1]
		filename = this_sample_info[2]
		filename_ls = filename.split(' ')
		# make links
		#When sample is run on multiple lanes with same barcode
		if len(path_ls)>1 or len(filename_ls)>1:
			cp_append_files(path_ls, sample, filename_ls) 
		else:
			makelink(path, sample, filename)

rule cutadapt:
	input:
		fq1 = "data/{sampleID}/R1.fq.gz",
		fq2 = "data/{sampleID}/R2.fq.gz",
	output:
		fq1o="1-cutadapt/{sampleID}_R1_trim.fq.gz",
		fq2o="1-cutadapt/{sampleID}_R2_trim.fq.gz",
	log:
		log="logs/cutadapt_{sampleID}.txt",
	conda:
		"phlame_snakemake",
	shell:
		"cutadapt -a CTGTCTCTTAT --cores=8 "
			"-o {output.fq1o} {input.fq1} 1> {log};"
		"cutadapt -a CTGTCTCTTAT --cores=8 "
			"-o {output.fq2o} {input.fq2} 1>> {log};"

rule sickle:
	input:
		fq1o = "1-cutadapt/{sampleID}_R1_trim.fq.gz",
		fq2o = "1-cutadapt/{sampleID}_R2_trim.fq.gz",
	output:
		fq1o="2-sickle/{sampleID}/filt1.fq.gz",
		fq2o="2-sickle/{sampleID}/filt2.fq.gz",
		fqSo="2-sickle/{sampleID}/filt_sgls.fq.gz",
	log:
		log="logs/sickle2050_{sampleID}.txt",
	conda:
		"phlame_snakemake",
	shell:
		"sickle pe -g -q 15 -l 50 -x -n -t sanger "
			"-f {input.fq1o} -r {input.fq2o} "
			"-o {output.fq1o} -p {output.fq2o} "
			"-s {output.fqSo} 1> {log}"

rule refGenome_index: 
	input:
		fasta=expand(REFGENOME_DIR + "/{reference}/genome.fasta",reference=set(REF_GENOME_ls))
	params:
		"data/references/{reference}/genome_bowtie2",
	output:
		bowtie2idx="data/references/{reference}/genome_bowtie2.1.bt2"
	conda:
		"phlame_snakemake",
	shell:
		"bowtie2-build -q {input.fasta} {params} "

rule bowtie2:
	input:
		fq1="2-sickle/{sampleID}/filt1.fq.gz",
		fq2="2-sickle/{sampleID}/filt2.fq.gz",
		bowtie2idx="data/references/{reference}/genome_bowtie2.1.bt2"
	params:
		refGenome="data/references/{reference}/genome_bowtie2", # just a prefix
	output:
		samA="3-bowtie2/{sampleID}_ref_{reference}_aligned.sam",
	log:
		log="logs/bowtie2_{sampleID}_ref_{reference}.txt",
	conda:
		"phlame_snakemake",
	shell:
		# 8 threads coded into json
		"bowtie2 --threads 8 -X 2000 --no-mixed --dovetail "
			"-1 {input.fq1} -2 {input.fq2} "
			"-x {params.refGenome} "
			"-S {output.samA} 2> {log} "

rule sam2bam:
    input:
        samA="3-bowtie2/{sampleID}_ref_{reference}_aligned.sam",
    params:
        # fqU1="3-bowtie2/{sampleID}_ref_{reference}_unaligned.1.fastq",
        # fqU2="3-bowtie2/{sampleID}_ref_{reference}_unaligned.2.fastq",
        bamDup="3-bowtie2/{sampleID}_ref_{reference}_aligned_dups.bam",
        bamDupMate="3-bowtie2/{sampleID}_ref_{reference}_aligned_dups.mates.bam",
        bamDupMateSort="3-bowtie2/{sampleID}_ref_{reference}_aligned_dups.sorted.mates.bam",
        DupStats="3-bowtie2/{sampleID}_ref_{reference}_markdup_stats.txt",
    output:
        bamA="3-bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam",
    conda:
        "phlame_snakemake",
    shell:
        # 8 threads coded into json
        " samtools view -bS {input.samA} | samtools sort -n - -o {params.bamDup} ;"
        " samtools fixmate -m {params.bamDup} {params.bamDupMate} ;"
        " samtools sort -o {params.bamDupMateSort} {params.bamDupMate} ;"
        " samtools markdup -r -s -f {params.DupStats} -d 100 -m s {params.bamDupMateSort} {output.bamA} ;"
        " samtools index {output.bamA} ;"
        # " bgzip -f {params.fqU1}; bgzip -f {params.fqU2} ;"
        " rm {input.samA} ;"
        " rm {params.bamDup} {params.bamDupMate} {params.bamDupMateSort} ;"

rule classify:
	input:
		bam = "3-bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam",
	params:
		cfr = CLASSIFIER_ls[0],
		refGenome=REFGENOME_DIR[0] + "/{reference}/genome.fasta"
	conda:
		"phlame_snakemake",
	output:
		frequencies="4-frequencies/{sampleID}_ref_{reference}_frequencies.csv",
		data="4-frequencies/{sampleID}_ref_{reference}_fitinfo.data",
	shell:
		"mkdir -p 4-frequencies ;"
		"phlame classify "
			"-i {input.bam} "
			"-c {params.cfr} "
			"-r {params.refGenome} "
			"-m bayesian "
			"-o {output.frequencies} "
			"-p {output.data} "
			"--max_pi 0.35 "
			"--min_prob 0.5 "
			"--min_snps 10 ;"

rule plot:
	input:
		frequencies = rules.classify.output.frequencies,
		data = rules.classify.output.data,
	params:
		refGenome="data/references/{reference}/genome.fasta"
	conda:
		"phlame_snakemake",
	output:
		plot="4-frequencies/{sampleID}_ref_{reference}_plot.pdf",
	shell:
		"mkdir -p 4-frequencies ;"
		"phlame plot "
			"-f {input.bam} "
			"-d {params.cfr} "
			"-p {params.refGenome}"
			"-o {output.plot} "
			"--max_pi 0.3 "
			"--min_prob 0.5 "