# PHLAME: Novelty-aware intraspecies profiling from metagenomes

PHLAME is a complete pipeline for the creation of intraspecies reference databases and the metagenomic detection of intraspecies clades, their relative frequency, and their estimated divergence from the reference phylogeny.

The accepted raw inputs to PHLAME are:
* [1] A species-specific assembled reference genome in .fasta format
* [2] A collection of whole genome sequences of the same species in .fastq or aligned .bam/.pileup format
* [3] Metagenomic sequencing data in either .fastq or aligned .bam format.

## Installation
```
$ pip install phlame
```

## Dependencies
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

## Tutorial

This tutorial uses the small set of files in `examples/`. 

### 1. Building a database

PHLAME uses a compressed object called a candidate mutation table to store allele information from many independent samples. To create one, we first need to use samtools/bcftools to extract pileups from aligned .bam files
```
$ samtools mpileup -q30 -x -s -O -d3000 -f Pacnes_C1.fasta skin_isolate_aligned.sorted.bam > skin_isolate_aligned.pileup
$ samtools mpileup -q30 -t SP -d3000 -vf Pacnes_C1.fasta skin_isolate_aligned.sorted.bam > skin_isolate_aligned.vcf.tmp
$ bcftools call -c -Oz -o skin_isolate_aligned.sorted.strain.vcf.gz skin_isolate_aligned.vcf.tmp
$ bcftools view -Oz -v snps -q .75 skin_isolate_aligned.sorted.strain.vcf.gz > skin_isolate_aligned.sorted.strain.variant.vcf.gz
$ tabix -p vcf skin_isolate_aligned.sorted.strain.variant.vcf.gz
$ rm skin_isolate_aligned.vcf.tmp
```

Pileup files can be quite large. We extract data from pileup files into a compressed format using the `counts` function in PHLAME.
```
phlame counts -p skin_isolate_aligned.pileup -v skin_isolate_aligned.sorted.strain.vcf.gz -w skin_isolate_aligned.sorted.strain.variant.vcf.gz -r Pacnes_C1.fasta -o skin_isolate.counts
```

Data from many counts files is aggregated into a candidate mutation table. For this, several counts files are already made in `examples/counts/`
```
phlame cmt -i examples/counts/*.counts -s 'A039,A441,A443,B089,F109,F189,L363' -r Pacnes_C1.fasta -o Cacnes_CMT.pickle.gz
```

From a candidate mutation table, we can create a phylogeny and a PHLAME database using the commands `phlame tree` and `phlame makedb`, respectively.

```
phlame tree -i Cacnes_CMT.pickle.gz -p Cacnes.phylip -o Cacnes.tree
```

Using the integrated tree-building step requires RaXML installed. Alternatively, you can use `tree` to create a PHYLIP formatted file, which plugs into many different phylogenetic inference algorithms
```
phlame tree -i Cacnes_CMT.pickle.gz -p Cacnes.phylip -r Cacnes_phylip2names.txt
```

Now that we have both our candidate mutation table and our tree, we can run the `makedb` step, which will detect candidate clades in our phylogeny as well as clade-specific mutations for each clade.

A key parameter to specify is `--min_branchlen`, which defines the minimum branch length for a branch of the phylogeny to be considered a clade. It is important to visualize your tree (for example, using [FigTree](https://github.com/rambaut/figtree/releases)) to determine a good value. The two outputs of the `makedb` step are the compressed database and a text file giving the identifies of each clade.
```
phlame makedb -i Cacnes_CMT.pickle.gz -t Cacnes.tree -o Cacnes_db.classifier -p Cacnes_cladeIDs.txt --min_branchlen 100
```

### 2. Classifying metagenome samples

To classify a metagenomic sample, you will first have to align your metagenomic reads to the same species-specific reference genome used to build your classifier. PHLAME takes as input the aligned .bam file. The classify step outputs a frequencies file, as well as a compressed data file. 

There are several options and parameters that can be set when running `phlame classify`. Two important ones are `-m`, which specifies whether PHLAME will run a maximum likelihood or Bayesian algorithm. The Bayesian algorithm takes longer to run but offers more information (see 3. Visualizing classification results). The parameter`--max-pi` defines the divergence limit past which strains in a sample will be considered too distant to be a member of the same clade (the default for this threshold is `0.35`).

```
phlame classify -i skin_mg_aligned.sorted.bam -c Cacnes_db.classifier -r Pacnes_C1.fasta -o skin_mg_frequencies.csv -p skin_mg_fitinfo.data --max_pi 0.35
```

The output of a frequencies file will look like this:
```
,Relative abundance,Estimated divergence,Confidence score
C.1,0.0,[0.42201403 0.71522194],0.004111111111111111
C.1.1,0.0,[0.7602548 0.8559106],0.0
C.2,0.9992863197270372,[1.08247258e-08 1.40671647e-04],1.0
C.2.1,0.05124272640768824,[5.98496803e-05 7.75948650e-04],1.0
C.2.1.1,0.0,[0.34709593 0.60451219],0.03766666666666667
C.2.1.1.1,0.0,[0.40496092 0.86246598],0.026555555555555554
C.2.1.1.1.1,0.0,[0.38075201 0.85498158],0.03422222222222222
C.2.1.1.2,0.0,[0.54419195 0.78240866],0.0
```

The 3 fields that PHLAME will return are: [1] the estimated relative abundance of the clade in the sample, [2] DVb, which represents the estimated divergence of the sample from the MRCA of that clade, and [3] a Probability Score, which represents the overall probability that the sample supports a clade that is within your `--max_pi` threshold. Note that Probability Score only has informative information in the Bayesian implementation of PHLAME, and will either be 1 or 0 in the MLE version.

### 3. Visualizing classification results

The compressed data file has lots of useful information that will add context to detection decisions. You can view the output of a data file with the command `phlame plot`.

```
phlame plot -f skin_mg_frequencies.csv -d skin_mg_fitinfo.data -o skin_mg_frequencies_plot.pdf
```

The output plot will look something like this. 

![Alt text](examples/example_plot.pdf)

Going from left to right, the plots show 






