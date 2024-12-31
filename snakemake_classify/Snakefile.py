############################################
# PHLAME Snakefile (Classify / Part 2)#
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
		# # Only data links # #
		expand("data/{sampleID}/R1.fq.gz",sampleID=SAMPLE_ls),
		expand("data/{sampleID}/R2.fq.gz",sampleID=SAMPLE_ls),
		# # Through all steps # #
		expand("3-bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
		expand("5-counts/{sampleID}_ref_{reference}.counts.pickle.gz", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
		expand("6-frequencies/{sampleID}_ref_{reference}_frequencies.csv", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
		# # Including cleanup # #
		# "logs/cleanUp_done.txt",
		# # With QC # #
		"3-bowtie2/alignment_stats.csv",

rule get_positions:
	input:
		cfrs = CLASSIFIER_ls[0],
		#cfs can also be specified 
	params:
		refGenome_file = (REFGENOME_DIR + "/" + REF_GENOME_ls[0]),
	output:
		all_positions="data/positions/allpositions.txt",
		chr_positions="data/positions/chrpositions.txt",
	run:
		from phlame_SM_module import get_positions
		get_positions(input.cfrs, 
					  output.all_positions, 
					  output.chr_positions, 
					  params.refGenome_file)

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
		"envs/cutadapt.yaml"
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
		"envs/sickle-trim.yaml"
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
		"envs/bowtie2.yaml"
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
		"envs/bowtie2.yaml"
	shell:
		# 8 threads coded into json
		"bowtie2 --threads 8 -X 2000 --no-mixed --dovetail "
			"-1 {input.fq1} -2 {input.fq2} "
			"-x {params.refGenome} "
			"-S {output.samA} 2> {log} "

rule bowtie2qc:
	input:
		bowtie2_logs = expand("logs/bowtie2_{sampleID}_ref_{reference}.txt", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
	output:
		alignment_stats = "3-bowtie2/alignment_stats.csv",
	conda:
		"envs/bowtie2qc.yaml",
	shell:
		"python3 {CURRENT_DIR}/scripts/bowtie2qc.py -s {spls} -d {CURRENT_DIR}"

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
            "envs/samtools115.yaml"
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

rule mpileup:
	input:
		bamA="3-bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam",
		ref=rules.refGenome_index.input.fasta,
		pos2grab="data/positions/chrpositions.txt",
	output:
		pileup="4-vcf/{sampleID}_ref_{reference}_aligned.sorted.pileup",
	conda:
		"envs/samtools15_bcftools12.yaml"
	shell:
		" samtools faidx {input.ref} ; "
		" samtools mpileup -q30 -x -s -O -d3000 "
			"-l {input.pos2grab} "
			"-f {input.ref} {input.bamA} > {output.pileup} ;" 

rule counts:
	input:
		pileup = "4-vcf/{sampleID}_ref_{reference}_aligned.sorted.pileup",
	params:
		refGenomeDir=expand(REFGENOME_DIR + "/{reference}/",reference=set(REF_GENOME_ls)),
		cfrs = CLASSIFIER_ls[0],
	output:
		phlame_cts = "5-counts/{sampleID}_ref_{reference}.counts.pickle.gz",
	shell:
		"python scripts/phlame_counts.py "
			"-i {input.pileup} "
			"-r {params.refGenomeDir} "
			"-c {params.cfrs} "
			"-o {output.phlame_cts}; "

rule classify:
	input:
		counts = rules.counts.output.phlame_cts,
	params:
		cfr = "Gardnerella_classifiers/Gvaginalis_HKY85.classifier",
		# level="Cacnes_classifiers/Cacnes_ALL_phylogroup_IDs.txt",
	conda:
		"envs/phlame.yaml"
	output:
		frequencies="6-frequencies/{sampleID}_ref_{reference}_frequencies.csv",
		data="6-frequencies/{sampleID}_ref_{reference}_fitinfo.data",
	shell:
		"mkdir -p 6-frequencies ;"
		"phlame_.py classify "
			"-i {input.counts} "
			"-c {params.cfr} "
			# "-l {params.level} "
			"-o {output.frequencies} "
			"-p {output.data} "
			"--max_pi 0.3 "
			"--min_prob 0.5 "
			"--min_snps 10 ;"

# rule cleanUp:
#     input:
#         candidate_mutation_table = "7-candidate_mutation_table/candidate_mutation_table.pickle.gz",
#     params:
#         temp_folder = "6-case_temp/",
#         cutad="1-cutadapt_temp/",
#         sickle="2-sickle2050_temp/",
#     output:
#         "logs/cleanUp_done.txt",
#     shell:
#         " rm -rf {params.temp_folder} {params.cutad} {params.sickle}; touch {output} ;"
