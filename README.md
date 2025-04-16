# PHLAME: Novelty-aware intraspecies profiling from metagenomes

PHLAME is a complete pipeline for the creation of intraspecies reference databases and the metagenomic detection of intraspecies clades, their relative frequency, and their estimated Divergence from the reference phylogeny.

The accepted raw inputs to PHLAME are:
* [1] A species-specific assembled reference genome in .fasta format
* [2] A collection of whole genome sequences of the same species in .fastq or .fasta format
* [3] Metagenomic sequencing data in either .fastq or aligned .bam format.

Link to preprint is [here](https://www.biorxiv.org/content/10.1101/2025.02.07.636498v1).

The data used in the manuscript is available [here](https://zenodo.org/records/15226099).

## Installation

You can install PHLAME using either pip or conda:

```
pip install phlame
```
```
conda install -c bioconda phlame
```

PHLAME is in active development. Please make sure you have the latest version installed before running.

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

PHLAME uses intraspecies reference databases to profile strain-level diversity of individual species from metagenomic data. PHLAME is *novelty-aware* and will quantify the proportion of novel diversity in a sample that is not be explained by existing reference genomes. This is made possible through PHLAME's Divergence (DVb) metric, which estimates the point on individual branches of a phylogeny for which novel strains in a sample are inferred to diverge from known references.

![alt text](docs/profile.png)

## Tutorial

This tutorial uses the small set of files found in `example/` and is made to be run inside the `example/` directory.

[PHLAME manual](docs/manual.md)

[0. Conceptual introduction to PHLAME](docs/conceptual_intro.md)

[1. Building a database](docs/building_database_tutorial.md)\
    *   [Collecting genomes your species of interest](docs/building_database_tutorial.md#1-collecting-genomes-for-your-species-of-interest)\
    *   [Sequence data to candidate mutation table](docs/building_database_tutorial.md#2-sequence-data-to-candidate-mutation-table)\
    *   [Creating a phylogeny](docs/building_database_tutorial.md#3-creating-a-phylogeny)\
    *   [Making a PHLAME database](docs/building_database_tutorial.md#4-making-a-phlame-database)

[2. Classifying metagenome samples](docs/classifying_samples_tutorial.md)\
    *   [Aligning metagenomic reads](docs/classifying_samples_tutorial.md#1-aligning-metagenomic-reads)\
    *   [Running phlame classify](docs/classifying_samples_tutorial.md#2-running-phlame-classify)

[3. Intepreting PHLAME results](docs/interpreting_results_tutorial.md)\
    *   [Visualizing classification results](docs/interpreting_results_tutorial.md#1-visualizing-classification-results)\
    *   [Analyzing results at specific phylogenetic levels](docs/interpreting_results_tutorial.md#2-analyzing-results-at-specific-phylogenetic-levels)


