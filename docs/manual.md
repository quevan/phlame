# PHLAME Manual


## Overview

You can view the core commands of PHLAME with `phlame -h`

```
            ...::: PHLAME v1.0 :::...
       Evan Qu, Lieberman Lab, MIT. 2025

Usage: phlame [-h] {classify,makedb,tree} ...


Choose one of the operations below for more detailed help.
Example: phlame classify -h

Main operations:
    classify ->  Estimate metagenomic clade-level frequencies using a PHLAME database.
    makedb   ->  Create a PHLAME database.
    tree     ->  Create a phylogenetic tree for a PHLAME database.

Auxiliary operations:
    plot -> Generate informative plots from classify output.
    cmt -> Convert counts files into a candidate mutation table.
    counts -> Convert aligned pileup files into compressed counts matrix format.

```

## PHLAME classify

```
usage: phlame classify [-h] -i INPUT -c CLASSIFIER -r REF [-l LEVEL] -o OUTPUT [-p OUTPUTDATA] -m {bayesian,mle} [--max_pi MAX_PI] [--min_snps MIN_SNPS] [--min_prob MIN_PROB] [--min_hpd MIN_HPD] [--seed SEED] [--verbose]

options:
  -h, --help           show this help message and exit
  -i INPUT             Path to input bam file (required).
  -c CLASSIFIER        Path to classifer file (required).
  -r REF               Path to reference genome (required, fasta format).
  -l LEVEL             Level specification.
  -o OUTPUT            Path to output frequencies file (required).
  -p OUTPUTDATA        Path to output data file (required).
  -m {bayesian,mle}    Inference algorithm to use. Defaults to 'mle'
  --max_pi MAX_PI      Maximum Divergence to count a lineage as present. Default = 0.3.
  --min_snps MIN_SNPS  Minimum number of present mutations to count a lineage as present. Default = 10.
  --min_prob MIN_PROB  Bayesian only: Minimum probability score to count a lineage as present. Default = 0.5.
  --min_hpd MIN_HPD    Bayesian only: Minimum value the highest posterior density interval over divergence must cover to count a lineage as present. Default = 0.15.
  --seed SEED          Set random seed for reproducibility.
  --verbose            Print progress messages.
```

## PHLAME makedb

```
usage: phlame makedb [-h] -i INPUT -t INTREE -o OUTDB -p OUTCLADES [-y OUTTREE] [-c INCLADES] [--outgroup OUTGROUP] [--max_outgroup MAX_OUTGROUP] [--min_snps MIN_SNPS] [--maxn MAXN] [--core CORE] [--min_branchlen MIN_BRANCHLEN]
                     [--min_leaves MIN_LEAVES] [--min_support MIN_SUPPORT] [--minAF MINAF] [--min_strand_cov MIN_STRAND_COV] [--qual QUAL] [--max_frac_ambiguous MAX_FRAC_AMBIGUOUS] [--midpoint]

options:
  -h, --help            show this help message and exit
  -i INPUT              Path to input candidate mutation table (required).
  -t INTREE             Path to input phylogeny (required, newick format).
  -o OUTDB              Path to output classifier (required).
  -p OUTCLADES          Path to output clades file (required).
  -y OUTTREE            Path to output tree labelled with clade names.
  -c INCLADES           Path to optional file (newline-separated) listing clades to be manually included in the classifier.
  --outgroup OUTGROUP   Specify a comma separated list of samples to be considered outgroups.
  --max_outgroup MAX_OUTGROUP
                        Maximum percent of outgroup samples a position can be found in (non-N) to be considered. Default=0.1
  --min_snps MIN_SNPS   Minimum number of SNPs to include a candidate clade. Default = 10.
  --maxn MAXN           Maximum percentage of Ns for a position to be considered. Default = 0.1.
  --core CORE           Minimum percent of samples a position must be found in (non-N) to be considered. Default = 0.9.
  --min_branchlen MIN_BRANCHLEN
                        Minimum branch length leading up to a clade. Default = 100.
  --min_leaves MIN_LEAVES
                        Minimum number of samples in a clade. Default = 3.
  --min_support MIN_SUPPORT
                        Minimum bootstrap support for a clade. Default = 0.75.
  --minAF MINAF         Minimum allele frequency to make a base call. Default = 0.75.
  --min_strand_cov MIN_STRAND_COV
                        Minimum per-strand coverage across a position to make a base call. Default = 2.
  --qual QUAL           Minimum MAPQ score to make a base call. Default = -30.
  --max_frac_ambiguous MAX_FRAC_AMBIGUOUS
                        Maximum fraction of ambiguous (N) calls to include a sample. Default = 0.5.
  --midpoint            Root the input tree at the midpoint. Default = False.
```

## PHLAME tree

```
usage: phlame tree [-h] [-i CMT] [-o OUTTREE] [-p OUTPHYLIP] [-r RENAMING] [-q INPHYLIP] [--rescale] [--min_cov MIN_COV] [--minAF MINAF] [--min_strand_cov MIN_STRAND_COV] [--qual QUAL] [--core CORE]
                   [--min_cov_position MIN_COV_POSITION] [--max_frac_ambiguous MAX_FRAC_AMBIGUOUS] [--copynum COPYNUM] [--remov_recomb] [--outgroup OUTGROUP]

options:
  -h, --help            show this help message and exit
  -i CMT                Path to input candidate mutation table.
  -o OUTTREE            Path to output tree.
  -p OUTPHYLIP          Path to output phylip file.
  -r RENAMING           Path to phylip renaming file.
  -q INPHYLIP           Path to existing phylip file (this will just run RaXML).
  --rescale             Rescale tree branch lengths into numbers of SNVs.
  --min_cov MIN_COV     Minimum coverage across a position to make a base call.
  --minAF MINAF         Minimum allele frequency to make a base call.
  --min_strand_cov MIN_STRAND_COV
                        Minimum per-strand coverage across a position to make a base call.
  --qual QUAL           Minimum MAPQ score to make a base call.
  --core CORE           Minimum percent of samples a position must be found in (non-N) to be considered.
  --min_cov_position MIN_COV_POSITION
                        Minimum median coverage of position across samples to include.
  --max_frac_ambiguous MAX_FRAC_AMBIGUOUS
                        Maximum fraction of ambiguous (N) calls to include a sample.
  --copynum COPYNUM     Maximum average copy number to include a position.
  --remov_recomb        Remove recombination events from the tree.
  --outgroup OUTGROUP   Specify a comma separated list of samples to be considered outgroups; i.e. not included in tree building.

```

## PHLAME plot

```
usage: phlame plot [-h] -f INFREQ -d INDATA -o OUTPUT [--max_pi MAX_PI] [--min_prob MIN_PROB]

options:
  -h, --help           show this help message and exit
  -f INFREQ            Path to input frequencies file (required).
  -d INDATA            Path to input data file (required).
  -o OUTPUT            Path to pdf of informative output plots (required).
  --max_pi MAX_PI      Maximum pi value to count a lineage as present. Should be the same as used in classify step. Default = 0.3.
  --min_prob MIN_PROB  Minimum probability score to count a lineage as present. Should be the same as used in classify step. Default = 0.5.
```


## PHLAME counts

```
usage: phlame counts [-h] -p PILEUP -v VCF -w VARIANT_VCF -r REF -o OUTPUT_COUNTS

options:
  -h, --help        show this help message and exit
  -p PILEUP         Path to input pileup file (required).
  -v VCF            Path to input VCF file (required).
  -w VARIANT_VCF    Path to input variant VCF file.
  -r REF            Path to reference genome.
  -o OUTPUT_COUNTS  Path to output counts file.

```


## PHLAME cmt

```
usage: phlame cmt [-h] -i COUNTS_FILES -s SAMPLE_NAMES -r REF -o OUT_CMT

options:
  -h, --help       show this help message and exit
  -i COUNTS_FILES  Path to file (newline-separated) listing counts files, in same order as sample names (required).
  -s SAMPLE_NAMES  Path to file (newline-separated) listing sample names (required).
  -r REF           Path to reference genome (required, fasta format).
  -o OUT_CMT       Path to output compressedCMT file (required).
```