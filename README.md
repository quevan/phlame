# PHLAME: Novelty-aware intraspecies profiling from metagenomes

PHLAME is a complete pipeline for the creation of intraspecies reference databases and the metagenomic detection of intraspecies clades, their relative frequency, and their estimated Divergence from the reference phylogeny.

The accepted raw inputs to PHLAME are:
* [1] A species-specific assembled reference genome in .fasta format
* [2] A collection of whole genome sequences of the same species in .fastq or aligned .bam/.pileup format
* [3] Metagenomic sequencing data in either .fastq or aligned .bam format.

Link to preprint is [here](https://www.biorxiv.org/content/10.1101/2025.02.07.636498v1).

## Installation
```
$ pip install phlame
```

## Dependencies
* python >=3.8, <3.13
* numpy - (tested with v1.20.3)
* matplotlib - (tested with v3.4.2)
* pandas - (tested with v1.2.5)
* biopython - (tested with v1.79)
* scipy - (tested with v1.6.2)
* statsmodels - (tested with v0.13.1)
* [ete3](https://etetoolkit.org/download/) - (tested with v3.1.2)
* [samtools](https://github.com/samtools/samtools) (>=v1.15)
* [bcftools](https://github.com/samtools/bcftools) (>=v1.2) 

### Optional

* [RaXML](https://cme.h-its.org/exelixis/web/software/raxml/) - (tested with v8.2.13)
* Additionally, starting with raw sequencing read data will require an aligner (like [bowtie2](https://bowtie-bio.sourceforge.net/bowtie2/index.shtml)).

## Overview

PHLAME constructs phylogenetic abundance profiles of individual species from metagenomic data. Unlike similar methods, PHLAME is *novelty-aware*, meaning that PHLAME will identify the phylogenetic resolution at which novel strains in a sample no longer share ancestry with the reference phylogeny. This functionality is made possible through PHLAME's Divergence (DVb) metric, which estimates the point on individual branches of a phylogeny for which novel strains in a sample are inferred to diverge from known references.

![alt text](example/profile.png)

## Tutorial

This tutorial uses the small set of files found in `example/` and is made to be run inside the `example/` directory.

### 1. Building a database

PHLAME uses a compressed object called a candidate mutation table to store allele information from many independent samples. To create one, we first need to use samtools/bcftools to extract pileups from aligned .bam files. See `snakemake_makedb` for an example of how to take raw sequencing reads to aligned .bam files.  
```
$ samtools mpileup -q30 -x -s -O -d3000 -f Pacnes_C1.fasta Cacnes_isolate_aligned.sorted.bam > Cacnes_isolate_aligned.pileup
$ bcftools mpileup -q30 -t SP -d3000 -f Pacnes_C1.fasta Cacnes_isolate_aligned.sorted.bam > Cacnes_isolate_aligned.vcf.tmp
$ bcftools call -c -Oz -o Cacnes_isolate_aligned.sorted.strain.vcf.gz Cacnes_isolate_aligned.vcf.tmp --ploidy 1
$ bcftools view -Oz -v snps -q .75 Cacnes_isolate_aligned.sorted.strain.vcf.gz > Cacnes_isolate_aligned.sorted.strain.variant.vcf.gz
$ tabix -p vcf Cacnes_isolate_aligned.sorted.strain.variant.vcf.gz
$ rm Cacnes_isolate_aligned.vcf.tmp
```

Pileup files can be quite large. We extract data from pileup files into a compressed format using the `counts` function in PHLAME.
```
phlame counts -p Cacnes_isolate_aligned.pileup -v Cacnes_isolate_aligned.sorted.strain.vcf.gz -w Cacnes_isolate_aligned.sorted.strain.variant.vcf.gz -r Pacnes_C1.fasta -o Cacnes_isolate.counts
```

Data from many counts files is aggregated into a candidate mutation table. For this, several counts files are already made in `example/counts/`
```
phlame cmt -i counts_files.txt -s sample_names.txt -r Pacnes_C1.fasta -o Cacnes_CMT.pickle.gz
```

From a candidate mutation table, we can create a phylogeny and a PHLAME database using the commands `phlame tree` and `phlame makedb`, respectively.

Using the integrated tree-building step requires RaXML installed.
```
conda install raxml
```

You can run the integrated `tree` step as follows:
```
phlame tree -i Cacnes_CMT.pickle.gz -p Cacnes.phylip -r Cacnes_phylip2names.txt -o Cacnes.tre
```

Alternatively, you can use `tree` to create a PHYLIP formatted file, which plugs into many different phylogenetic inference algorithms.
```
phlame tree -i Cacnes_CMT.pickle.gz -p Cacnes.phylip -r Cacnes_phylip2names.txt
```

Now that we have both our candidate mutation table and our tree, we can run the `makedb` step, which will detect candidate clades in our phylogeny as well as clade-specific mutations for each clade.

It is important to visualize your tree (for example, using [FigTree](https://github.com/rambaut/figtree/releases)) before moving on to the database creation step. Looking at our phylogeny will give us important information, including whether the species has obvious population structure in the first place. Our rooted phylogeny in `example/` looks like this:

![alt text](example/tree.png)

At a quick glance, it looks like there are 3 distinct clades in our phylogeny, separated by a minimum branch length of ~623 mutations. By default, PHLAME will rescale branch lengths into absolute numbers of mutations when the correlation between the two is sufficiently high (0.75). A key parameter to give to `makedb` is `--min_branchlen`, which defines the minimum branch length for a branch of the phylogeny to be considered a clade. The two outputs of the `makedb` step are the compressed database and a text file giving the identifies of each clade. The phylogeny should be rooted in some way before inputting into the `makedb` step. You can specify `--midpoint` to default midpoint root the phylogeny.
```
phlame makedb -i Cacnes_CMT.pickle.gz -t rescaled_Cacnes.tree -o Cacnes_db.classifier -p Cacnes_cladeIDs.txt --min_branchlen 500 --min_leaves 2 --midpoint
```

After running makedb successfully, PHLAME will report the number of clade-specific mutations found for each clade. The identities of each clade can be found in the `cladeIDs.txt` file
```
Reading in files...
Number of core positions: 16280/17491
Getting unanimous alleles...
Getting unique alleles...
Classifier results:
Clade C.1: 3959 csSNPs found
Clade C.1.1: 844 csSNPs found
Clade C.2: 3956 csSNPs found
Clade C.2.1: 1338 csSNPs found
Clade C.2.2: 3526 csSNPs found
```
### 2. Classifying metagenome samples

To classify a metagenomic sample, you will first have to align your metagenomic reads to the same species-specific reference genome used to build your classifier. PHLAME takes as input the aligned, indexed .bam file. See `snakemake_classify` for an example of how to take raw sequencing reads to aligned .bam files. 

There are several options and parameters that can be set when running `phlame classify`. Two important ones are `-m`, which specifies whether PHLAME will run a maximum likelihood or Bayesian algorithm. The Bayesian algorithm takes longer to run but offers more information (see 3. Visualizing classification results). The parameter`--max-pi` defines the divergence limit past which strains in a sample will be considered too distant to be a member of the same clade (the default for this threshold is `0.35`). 

```
phlame classify -i skin_mg_aligned.sorted.bam -c Cacnes_db.classifier -r Pacnes_C1.fasta -m mle -o skin_mg_frequencies.csv -p skin_mg_fitinfo.data --max_pi 0.35
```

After running, the classify step will output a frequencies file, as well as a compressed data file. The output of a frequencies file will look like this:
```
,Relative abundance,DVb,Probability score
C.1,0.0,0.6874,0.0
C.1.1,0.0,-1.0,-1.0
C.2,0.3690262106911293,0.3425,1.0
C.2.1,0.0,0.5683,0.0
C.2.2,0.0,0.6815,0.0
```

The 3 fields that PHLAME will return are: [1] the estimated relative abundance of the clade in the sample, [2] DVb, which represents the estimated divergence of the sample from the MRCA of that clade, and [3] a Probability Score, which represents the overall probability that the sample supports a clade that is within your `--max_pi` threshold. Note that Probability Score only has information in the Bayesian implementation of PHLAME, and will either be 1 or 0 in the MLE version. You may notice that the total relative abundances across any set of non-overlapping clades does not add up to 1. This is intended and suggests that the sample may harbor intraspecies diversity that is novel with regards to any of the clades in the reference set.

### 3. Visualizing classification results

The compressed data file has lots of useful information that can be used to help visualize detection decisions. You can view the output of a data file with the command `phlame plot`; this is generally much more useful when running the bayesian version of the classify step, as you will be able to visualize full posteriors over each parameter. For this, a pre-made data file has been included in `example`.

```
phlame plot -f skin_mg_frequencies.csv -d skin_mg_fitinfo_bayesian.data -o skin_mg_frequencies_plot.pdf
```

![alt text](example/plot.png)

Each clade will have four relevant plots. From left to right, they are: [1] A histogram of the actual number of reads supporting each clade-specific allele (red), as well as all alleles at the same positions (gray). [2] The posterior probability over the pi parameter (equivalent to DVb) in green, as well as the 95% highest posterior density interval (gray bar). [3] The posterior probability over the lambda (rate) parameter and [4] The posterior probability density opver the relative abundance of the clade in the same. In this particular example, The posterior densities all have fairly high spreads because the sequencing depth is low. Visualizing the posterior densities helps us make detection decisions. For example, while clade C.2 is has enough density below our threshold to be detected, very little density is actually centered around pi values of 0. If we wanted to limit our detections to only strains that we think are for sure within the mRCA of C.2, we might reject this detection (for example, via the --hpd threshold in `phlame classify`)
