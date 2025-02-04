# PHLAME: Novelty-aware intraspecies profiling in strain-rich metagenomes

PHLAME is a complete pipeline for the creation of intraspecies reference databases and the metagenomic detection of intraspecies clades, their relative frequency, and their estimated divergence from the reference phylogeny.

The accepted raw inputs to PHLAME are:
* [1] A species-specific assembled reference genome in .fasta format
* [2] A collection of whole genome sequences of the same species in .fastq or aligned .bam/.pileup format
* [3] metagenomic sequencing data in either .fastq or aligned .bam/.pileup format.

## Installation
```
$ pip install drep
```



## Dependencies
* numpy - (tested with v1.20.3)
* matplotlib - (tested with v3.4.2)
* pandas - (tested with v1.2.5)
* biopython - (tested with v1.79)
* scipy - (tested with v1.6.2)
* ete3 - (tested with v3.1.2)
* statsmodels - (tested with v0.13.1)

### Optional

* [RaXML](https://cme.h-its.org/exelixis/web/software/raxml/) - (tested with v8.2.13)
