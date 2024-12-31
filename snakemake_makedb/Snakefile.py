####################################################
# Phlame Snakefile (Make Classifier / Part 1)#
####################################################


###############
# PRE-SNAKEMAKE 
###############

import sys
import os

# Global variables: In theory do not need to be changed

CURRENT_DIRECTORY = os.getcwd()
REF_GENOME_DIRECTORY = config["ref_genome_directory"]
SCRIPTS_DIRECTORY = config["myscripts_directory"]
sys.path.insert(0, SCRIPTS_DIRECTORY)
spls = config["sample_table"]

from ss_caller_module import *
from itertools import compress

spls = config["sample_table"]

[PATH_ls, SAMPLE_ls, FILENAME_ls, REF_Genome_ls, OUTGROUP_ls] = read_samples_CSV_classifier(spls)
# Write sample_info.csv for each sample
split_samplesCSV_classifier(PATH_ls, SAMPLE_ls, FILENAME_ls, REF_Genome_ls, OUTGROUP_ls)

#require the same reference genome for all samples
# assert len(set(REF_Genome_ls))

###########
# SNAKEMAKE
###########

rule all:
	input:
		# # Only data links # #
		expand("data/{sampleID}/R1.fq.gz",sampleID=SAMPLE_ls),
		expand("data/{sampleID}/R2.fq.gz",sampleID=SAMPLE_ls),
		# # Through mapping steps # #
		expand("1-Mapping/quals/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.quals", zip, sampleID=SAMPLE_ls, reference=REF_Genome_ls),
		expand("1-Mapping/diversity/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.diversity.gz", zip, sampleID=SAMPLE_ls, reference=REF_Genome_ls),
		# # CMT # #
		"2-Case/candidate_mutation_table/candidate_mutation_table.pickle.gz",
		# # Tree # #
		# "8-tree/parsimony.tre",
		# # With QC # #
		"1-Mapping/bowtie2/alignment_stats.csv",
		# # Including cleanup # #
		# "logs/cleanUp_done.txt",


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
		"envs/cutadapt.yaml",
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
		"envs/sickle-trim.yaml",
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
		"envs/bowtie2.yaml",
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
		"envs/bowtie2.yaml",
	shell:
		# 8 threads coded into json
		# --un-conc {params.fqU}
		"bowtie2 --threads 16 -X 2000 --no-mixed --dovetail "
			"-x {params.refGenome} "
			"-1 {input.fq1} -2 {input.fq2} "
			"-S {output.samA} 2> {log} ;"

rule bowtie2qc:
	input:
		bowtie2_logs = expand("logs/bowtie2_{sampleID}_ref_{reference}.txt", sampleID=SAMPLE_ls, reference=set(REF_Genome_ls)),
	output:
		alignment_stats = "1-Mapping/bowtie2/alignment_stats.csv",
	conda:
		"envs/bowtie2qc.yaml",
	shell:
		"python3 {SCRIPTS_DIRECTORY}/bowtie2qc.py -s {spls} -d {CURRENT_DIRECTORY} ;"

rule sam2bam:
	input:
		samA=rules.bowtie2.output.samA,
	output:
		bamA="1-Mapping/bowtie2/{sampleID}_ref_{reference}_aligned.sorted.bam",
	benchmark:
		"benchmarks/rule_sam2bam_{sampleID}_{reference}.benchmark",
	conda:
		"envs/samtools15_bcftools12.yaml",
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
        "envs/samtools15_bcftools12.yaml"
    shell:
        " samtools faidx {input.fasta} ; "

rule mpileup2vcf:
	input:
		bamA=rules.sam2bam.output.bamA,
		ref=REF_GENOME_DIRECTORY+"/{reference}/genome.fasta",
		fasta_idx = ancient(rules.samtools_idx.output.fasta_idx),
	output:
		pileup="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.pileup",
		variants="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.vcf.gz",
		vcf_strain="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.vcf.gz",
	params:
		vcf_raw="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.gz",
	benchmark:
		"benchmarks/rule_mpileup2vcf_{sampleID}_{reference}.benchmark",
	conda:
		"envs/samtools15_bcftools12.yaml",
	shadow: 
		"minimal", # avoids leaving leftover temp files esp if job aborted
	shell:
		" samtools mpileup -q30 -x -s -O -d3000 -f {input.ref} {input.bamA} > {output.pileup} ;" 
		" samtools mpileup -q30 -t SP -d3000 -vf {input.ref} {input.bamA} > {params.vcf_raw} ;"
		" bcftools call -c -Oz -o {output.vcf_strain} {params.vcf_raw} ;"
		" bcftools view -Oz -v snps -q .75 {output.vcf_strain} > {output.variants} ;"
		" tabix -p vcf {output.variants} ;"
		" rm {params.vcf_raw} ;"

rule vcf2quals:
	input:
		vcf_strain="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.vcf.gz",
	params:
		refGenomeDir = REF_GENOME_DIRECTORY+"/{reference}/", 
	output:
		file_quals = "1-Mapping/quals/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.quals",
	conda:
		"envs/py_for_snakemake.yaml",
	benchmark:
		"benchmarks/rule_vcf2quals_{sampleID}_{reference}.benchmark",
	shell:
		"mkdir -p 1-Mapping/quals ;"
		"python {SCRIPTS_DIRECTORY}/ss_vcf2quals_snakemake.py -i {input.vcf_strain} -r {params.refGenomeDir} -o {output.file_quals} ;"
		

rule pileup2diversity_matrix:
	input:
		pileup="1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.pileup",
	params:
		refGenomeDir=REF_GENOME_DIRECTORY+"/{reference}/",
	output:
		file_diversity = "1-Mapping/diversity/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.diversity.gz",
		file_coverage = "1-Mapping/diversity/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.coverage.gz",
	conda:
		"envs/py_for_snakemake.yaml",
	benchmark:
		"benchmarks/rule_pileup2diversity_{sampleID}_{reference}.benchmark",
	shell:
		"mkdir -p 1-Mapping/diversity ;"
		"python {SCRIPTS_DIRECTORY}/ss_pileup2diversity.py -i {input.pileup} -r {params.refGenomeDir} -o {output.file_diversity} -c {output.file_coverage} ;"


rule include_outgroup:
	input:
		#just here to make sure happens after mapping step 
		vcf=expand("1-Mapping/vcf/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.vcf.gz",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls),
		qual=expand("1-Mapping/quals/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.quals",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls),
		div=expand("1-Mapping/diversity/{sampleID}_ref_{reference}_aligned.sorted.strain.variant.diversity.gz",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls),
	output:
		vcf_links = expand("2-Case/temp/vcf/{sampleID}_ref_{reference}_outgroup{outgroup}.vcf.gz",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls, outgroup=OUTGROUP_ls),
		qual_mat_links = expand("2-Case/temp/qual/{sampleID}_ref_{reference}_outgroup{outgroup}.quals",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls, outgroup=OUTGROUP_ls),
		div_mat_links = expand("2-Case/temp/diversity/{sampleID}_ref_{reference}_outgroup{outgroup}.diversity.gz",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls, outgroup=OUTGROUP_ls),
	log:
		"logs/build_links.log"
	run:
		import subprocess
		#subprocess.run( "rm -fr 2-Case/temp/ " ,shell=True) # clean it up prior run
		subprocess.run( "mkdir -p 2-Case/temp/diversity/ 2-Case/temp/qual/ 2-Case/temp/diversity/ " ,shell=True)
		for i in range(len(SAMPLE_ls)):
			subprocess.run( "ln -fs -T " + CURRENT_DIRECTORY + "/1-Mapping/diversity/" + SAMPLE_ls[i] + "_ref_" + REF_Genome_ls[i] + "*diversity.gz 2-Case/temp/diversity/" + SAMPLE_ls[i] + "_ref_" + REF_Genome_ls[i] + "_outgroup" + OUTGROUP_ls[i] + ".diversity.gz" ,shell=True)
			subprocess.run( "ln -fs -T " + CURRENT_DIRECTORY + "/1-Mapping/quals/" + SAMPLE_ls[i] + "_ref_" + REF_Genome_ls[i] + "*quals 2-Case/temp/qual/" + SAMPLE_ls[i] + "_ref_" + REF_Genome_ls[i] + "_outgroup" + OUTGROUP_ls[i] + ".quals" ,shell=True)
			subprocess.run( "ln -fs -T " + CURRENT_DIRECTORY + "/1-Mapping/vcf/" + SAMPLE_ls[i] + "_ref_" + REF_Genome_ls[i] + "*variant.vcf.gz 2-Case/temp/vcf/" + SAMPLE_ls[i] + "_ref_" + REF_Genome_ls[i] + "_outgroup" + OUTGROUP_ls[i] + ".vcf.gz " ,shell=True)

rule variants2positions:
	input:
		variants = "2-Case/temp/vcf/{sampleID}_ref_{reference}_outgroup{outgroup}.vcf.gz",
	params:
		refGenomeDir = REF_GENOME_DIRECTORY + "/{reference}/",
		maxFQ=-30,
		outgroup_tag = "{outgroup}", # boolean (0==ingroup or 1==outgroup)
	output:
		positions = "2-Case/temp/positions/{sampleID}_ref_{reference}_outgroup{outgroup}_positions.pickle",
	conda:
		"envs/py_for_snakemake.yaml",
	benchmark:
		"benchmarks/rule_variants2positions_{sampleID}_{reference}_{outgroup}.benchmark",
	shell:
		"mkdir -p 2-Case/temp/positions/ ;"
		"python scripts/ss_variants2positions.py -i {input.variants} -o {output.positions} -r {params.refGenomeDir} -q {params.maxFQ} -b {params.outgroup_tag} ;"    

rule combine_positions_prep:
	input:
		mat_positions = expand("2-Case/temp/positions/{sampleID}_ref_{reference}_outgroup{outgroup}_positions.pickle",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls, outgroup=OUTGROUP_ls)
	params:	
		outgroup_bool = expand( "{outgroup}" , outgroup=OUTGROUP_ls ),
	output:
		string_input_pos = "2-Case/temp/positions/string_file_other_p_to_consider.txt",
		string_outgroup_bool = "2-Case/temp/positions/string_outgroup_bool.txt",
	run:
		with open( output.string_input_pos ,"w") as f: 
			print(*input.mat_positions, sep="\n", file=f)
		with open( output.string_outgroup_bool ,"w") as f:
			print(*params.outgroup_bool, sep="\n", file=f)

rule combine_positions:
	input:
		string_input_pos = "2-Case/temp/positions/string_file_other_p_to_consider.txt",
		string_outgroup_bool = "2-Case/temp/positions/string_outgroup_bool.txt",
	params:
		# file_other_p_to_consider = "add_positions/other_positions.mat",
		refGenomeDir = expand(REF_GENOME_DIRECTORY + "/{reference}/",reference=set(REF_Genome_ls)), # expands to single reference genome!
	output:
		allpositions = "2-Case/temp/allpositions.pickle",
	benchmark:
		"benchmarks/rule_combine_positions.benchmark",
	conda:
		"envs/py_for_snakemake.yaml",
	shell:
		"python scripts/ss_combine_positions.py "
			"-i {input.string_input_pos} "
			"-r {params.refGenomeDir} "
			"-b {input.string_outgroup_bool} "
			"-o {output.allpositions} ;"

# build input for candidate_mutation_table
rule candidate_mutation_table_prep:
	input:
		diversity=expand("2-Case/temp/diversity/{sampleID}_ref_{reference}_outgroup{outgroup}.diversity.gz",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls, outgroup=OUTGROUP_ls),
		quals=expand("2-Case/temp/qual/{sampleID}_ref_{reference}_outgroup{outgroup}.quals",zip,sampleID=SAMPLE_ls, reference=REF_Genome_ls, outgroup=OUTGROUP_ls),
	params:
		sampleID_names=expand( "{sampleID}" , sampleID=SAMPLE_ls ),
	output:
		string_diversity="2-Case/temp/string_diversity_mat.txt",
		string_quals="2-Case/temp/string_qual_mat.txt",
		string_sampleID_names="2-Case/temp/string_sampleID_names.txt",
	benchmark:
		"benchmarks/rule_candidate_mutation_table_prep.benchmark",
	run:
		with open( output.string_diversity ,"w") as f: 
			print(*input.diversity, sep="\n", file=f)
		with open( output.string_quals ,"w") as f:
			print(*input.quals, sep="\n", file=f)
		with open( output.string_sampleID_names ,"w") as f: 
			print(*params.sampleID_names, sep="\n", file=f)

rule candidate_mutation_table:
	input:
		positions = rules.combine_positions.output.allpositions,
		string_diversity = rules.candidate_mutation_table_prep.output.string_diversity,
		string_quals = rules.candidate_mutation_table_prep.output.string_quals,
		string_sampleID_names = rules.candidate_mutation_table_prep.output.string_sampleID_names,
		string_outgroup_bool = rules.combine_positions_prep.output.string_outgroup_bool,
	output:
		cmt="2-Case/candidate_mutation_table/candidate_mutation_table.pickle.gz",
	conda:
		"envs/py_for_snakemake.yaml",
	benchmark:
		"benchmarks/rule_candidate_mutation_table.benchmark",
	shell:
		# -c/-n optional flag to build cov/norm matrix in folder of cmt. check -h for help.
		"python3 scripts/build_candidate_mutation_table.py "
			"-p {input.positions} "
			"-s {input.string_sampleID_names} "
			"-g {input.string_outgroup_bool} "
			"-q {input.string_quals} "
			"-d {input.string_diversity} "
			"-o {output.cmt} ;"
		

################
# Post-case step
################

rule cmt2tree:
	input:
		cmt="2-Case/candidate_mutation_table/candidate_mutation_table.pickle.gz",
	params:
		ref=REF_GENOME_DIRECTORY+"/{REF_Genome_ls[0]}/genome.fasta",
		align="1-Mapping/bowtie2/alignment_stats.csv",
		dnapars_exe="/scratch/mit_lieberman/projects/evan/tools/phylip-3.697/exe/dnapars",
		phylip_prefix="8-tree/good_positions_for_tree",
		rename_prefix="8-tree/phylip2names.txt",
		tree_prefix="8-tree/parsimony",
	output:
		tree="8-tree/parsimony.tre",
	conda:
		"envs/py_for_snakemake.yaml",
	benchmark:
		"benchmarks/cmt2phylip.benchmark",
	shell:
		"ln -s {params.dnapars_exe} ./dnapars ;"
		"mkdir -p 8-tree ;"
		"python scripts/cmt2tree.py -i {input.cmt} -p {params.phylip_prefix} -o {params.tree_prefix} -n {params.rename_prefix} -r {params.ref} -a {params.align};"
		# optional arguments
		# --min_cov --min_maf --min_strand_cov --min_qual --min_presence_core --min_median_cov_samples --filter_indels (bool)


# rule define_levels:


# rule get_clade_specific_snps:

# rule cleanUp:
#     input:
#         candidate_mutation_table = "2-candidate_mutation_table/candidate_mutation_table.pickle.gz",
#     params:
#         temp_folder = "1-temp_pos",
#     output:
#         "logs/DONE_cleanUp"
#     shell:
#         " rm -rf {params.temp_folder} ; touch {output} "


