####################################################
# PHLAME Snakefile (MakeDB)
####################################################


###############
# PRE-SNAKEMAKE 
###############

import sys
import os

##########################################################################################
# Global variables: In theory do not need to be changed

CURRENT_DIRECTORY = os.getcwd()
REF_GENOME_DIRECTORY = config["ref_genome_directory"]
SCRIPTS_DIRECTORY = config["myscripts_directory"]
sys.path.insert(0, SCRIPTS_DIRECTORY)
spls = config["sample_table"]

from snakemake_functions import *
# from itertools import compress

[PATH_ls, SAMPLE_ls, FILENAME_ls, REF_GENOME_ls, OUTGROUP_ls] = read_samples_CSV_classifier(spls)
# Write sample_info.csv for each sample
split_samplesCSV_classifier(PATH_ls, SAMPLE_ls, FILENAME_ls, REF_GENOME_ls, OUTGROUP_ls)

# Require the same reference genome for all samples
assert len(set(REF_GENOME_ls))==1

##########################################################################################

rule all:
	input:
		# # Only data links # #
		expand("data/{sampleID}/R1.fq.gz",sampleID=SAMPLE_ls),
		expand("data/{sampleID}/R2.fq.gz",sampleID=SAMPLE_ls),
		# # Through alignment steps # #
		expand("1-Mapping/bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
		# # Candidate mutation table # #
		expand("2-Case/CMT_ref_{reference}.pickle.gz",  reference=set(REF_GENOME_ls)),
		

rule make_data_links:
	# NOTE: All raw data needs to be named fastq.gz. No fq! The links will be named fq though.
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
		paths = path.split(' ')
		sample = this_sample_info[1]
		filename = this_sample_info[2]
		filenames = filename.split(' ')
		# make links
		if len(paths)>1 or len(filenames)>1:
			cp_append_files(paths, sample, filenames) #When sample is run on multiple lanes with same barcode
		else:
			makelink(path, sample, filename)

rule cutadapt:
	input:
		# Recommend using symbolic links to your likely many different input files
		fq1 = rules.make_data_links.output.fq1,
		fq2 = rules.make_data_links.output.fq2,
	output:
		fq1o="tmp/{sampleID}/R1_trim.fq.gz",
		fq2o="tmp/{sampleID}/R2_trim.fq.gz",
	log:
		log="logs/cutadapt_{sampleID}.txt",
	conda:
		"phlame_snakemake",
	benchmark:
		"benchmarks/rule_cutadapt_{sampleID}.benchmark",
	shell:
		"cutadapt -a CTGTCTCTTAT "
			"-o {output.fq1o} "
			"{input.fq1} 1> {log};"
		"cutadapt -a CTGTCTCTTAT "
			"-o {output.fq2o} "
			"{input.fq2} 1>> {log};"

rule sickle2050:
	input:
		fq1o = rules.cutadapt.output.fq1o,
		fq2o = rules.cutadapt.output.fq2o,
	output:
		fq1o="tmp/{sampleID}/R1_filt.fq.gz",
		fq2o="tmp/{sampleID}/R2_filt.fq.gz",
		fqSo="tmp/{sampleID}/filt_sgls.fq.gz",
	log:
		log="logs/sickle2050_{sampleID}.txt",
	conda:
		"phlame_snakemake",
	benchmark:
		"benchmarks/rule_sickle2050_{sampleID}.benchmark",
	shell:
		"sickle pe -g -t sanger "
			"-f {input.fq1o} -r {input.fq2o} "
			"-o {output.fq1o} -p {output.fq2o} -s {output.fqSo} "
			"-q 20 -l 50 -x -n 1> {log}"

rule refGenome_index: 
	input:
		fasta=REF_GENOME_DIRECTORY+"/{reference}/genome.fasta",
	params:
		refGenome=REF_GENOME_DIRECTORY+"/{reference}/genome_bowtie2",
	output:
		bowtie2idx=REF_GENOME_DIRECTORY+"/{reference}/genome_bowtie2.1.bt2",
	conda:
		"phlame_snakemake",
	shell:
		"bowtie2-build -q {input.fasta} {params.refGenome} ;"

rule bowtie2:
	input:
		fq1=rules.sickle2050.output.fq1o,
		fq2=rules.sickle2050.output.fq2o,
		bowtie2idx=rules.refGenome_index.output.bowtie2idx, # put here, so rule bowtie2 only executed after rule refGenome_index done
	params:
		refGenome=REF_GENOME_DIRECTORY+"/{reference}/genome_bowtie2",
		# fqU="1-Mapping/bowtie2/{sampleID}_ref_{reference}_unaligned.fastq", # just a prefix. 
	output:
		samA="1-Mapping/bowtie2/{sampleID}_ref_{reference}_aligned.sam",
	log:
		log="logs/bowtie2_{sampleID}_ref_{reference}.txt",
	benchmark:
		"benchmarks/rule_bowtie2_{sampleID}_{reference}.benchmark",
	conda:
		"phlame_snakemake",
	shell:
		# 8 threads coded into json
		# --un-conc {params.fqU}
		"bowtie2 --threads 16 -X 2000 --no-mixed --dovetail "
			"-x {params.refGenome} "
			"-1 {input.fq1} -2 {input.fq2} "
			"-S {output.samA} 2> {log} ;"

rule sam2bam:
	input:
		samA=rules.bowtie2.output.samA,
	output:
		bamA="1-Mapping/bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam",
	benchmark:
		"benchmarks/rule_sam2bam_{sampleID}_{reference}.benchmark",
	conda:
		"phlame_snakemake",
	shell:
		# 8 threads coded into json
		" samtools view -bS {input.samA} | samtools sort - -o {output.bamA} ;"
		" samtools index {output.bamA} ;"
		" rm {input.samA} ;"

# Indexes reference genome for samtools
rule samtools_idx:
    input:
        fasta = REF_GENOME_DIRECTORY+"/{reference}/genome.fasta",
    output:
        fasta_idx = REF_GENOME_DIRECTORY+"/{reference}/genome.fasta.fai",
    conda:
        "phlame_snakemake"
    shell:
        " samtools faidx {input.fasta} ; "

rule mpileup2vcf:
	input:
		bamA=rules.sam2bam.output.bamA,
		ref=REF_GENOME_DIRECTORY+"/{reference}/genome.fasta",
		fasta_idx = ancient(rules.samtools_idx.output.fasta_idx),
	output:
		pileup="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.pileup",
		vcf="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.vcf.gz",
		vcf_variants="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.vcf.gz",
	params:
		vcf_tmp="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.vcf.tmp",
	benchmark:
		"benchmarks/rule_mpileup2vcf_{sampleID}_{reference}.benchmark",
	conda:
		"phlame_snakemake",
	shadow: 
		"minimal", # avoids leaving leftover temp files esp if job aborted
	shell:
		" samtools mpileup -q30 -x -s -O -d3000 -f {input.ref} {input.bamA} > {output.pileup} "
		" bcftools mpileup -q30 -d3000 -f {input.ref} {input.bamA} > {params.vcf_tmp} "
		" bcftools call -c -Oz -o {output.vcf} {params.vcf_tmp} --ploidy 1 "
		" bcftools view -Oz -v snps -q .75 {output.vcf} > {output.vcf_variants} "
		" tabix -p vcf {output.vcf_variants} "
		" rm {params.vcf_tmp} "

rule counts:
	input:
		pileup="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.pileup",
		vcf="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.vcf.gz",
		vcf_variants="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.vcf.gz",
	params:
		refGenome=REF_GENOME_DIRECTORY+"/{reference}/genome.fasta",
	output:
		counts="1-Mapping/counts/{sampleID}_ref_{reference}_aligned.counts"
	conda:
		"phlame_snakemake"
	shell:
		"phlame counts -p {input.pileup} -v {input.vcf} -w {input.vcf_variants} -r {params.refGenome} -o {output.counts}"

rule candidate_mutation_table_prep:
	input:
		counts=expand("1-Mapping/counts/{sampleID}_ref_{reference}_aligned.counts", sampleID=SAMPLE_ls, reference=set(REF_GENOME_ls)),
	output:
		counts_files="2-Case/counts_files.txt",
		sample_names="2-Case/sample_names.txt",
	run:
		with open(output.counts_files) as f:
			for c in input.counts:
				f.write(c+'\n')
		with open(output.sample_names) as f:
			for s in SAMPLE_ls:
				f.write(s+'\n')
			
rule candidate_mutation_table:
	input:
		counts_files="2-Case/counts_files.txt",
		sample_names="2-Case/sample_names.txt",
		ref=REF_GENOME_DIRECTORY+"/{reference}/genome.fasta",
	output:
		cmt="2-Case/CMT_ref_{reference}.pickle.gz",
	conda:
		"phlame_snakemake"
	shell:
		"phlame cmt -i counts_files.txt -s sample_names.txt -r {input.ref} -o {output.cmt}"
		