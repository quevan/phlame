# Classifying metagenome samples

This tutorial is made to be on run on a small set of files inside the `examples/` directory. In addition, key outputs of this tutorial are available in `examples/output` for you to check your work with.

## 1. Aligning metagenomic reads

To classify a metagenomic sample, you will first have to align your metagenomic reads to the same species-specific reference genome used to build your classifier. PHLAME takes as input the aligned, indexed .bam file. 

See `snakemake_classify` for an snakemake that takes you from metagenomic sequence data to aligned .bam files, see `snakemake_classify`.

We use bowtie2 in this tutorial to align short reads, but any aligner that produces compatible .bam files could work in theory.

```
$ bowtie2 -X 2000 --no-mixed --dovetail -x Pacnes_C1.fasta -1 5X_skinmg_A_1.fastq.gz -2 5X_skinmg_A_2.fastq.gz -S skinmg_A.sam
```

We use samtools to convert the .sam file to a .bam file (a compressed file format for alignment data)
```
$ samtools view -bS skinmg_A.sam | samtools sort - -o skinmg_A.bam
$ samtools index skinmg_A.bam
$ rm skinmg_A.sam
```

## 2. Running phlame classify

You can run the PHLAME classify step once you have an aligned .bam file of your metagenome sample

There are several options and parameters that can be set when running `phlame classify`. Two important ones are:
*   `-m`, which specifies whether PHLAME will run a maximum likelihood or Bayesian algorithm. The Bayesian algorithm takes longer to run but offers more information (see 3. Visualizing classification results). 
*   `--max-pi`, which defines the divergence limit past which strains in a sample will be considered too distant to be a member of the same clade (the default for this threshold is `0.35`).

To see the full list of options, check the [manual](manual.md) or the help option (`phlame makedb -h`). 


We can run `phlame classify` using default parameters as follows:
```
$ phlame classify -i skin_mg_A.sam -c Cacnes_db.classifier -r Pacnes_C1.fasta -m bayesian -o skin_mg_frequencies.csv -p skin_mg_fitinfo.data
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

The 3 fields that PHLAME will return are: 
*   Relative abundance: the estimated relative abundance of the clade in the sample
*   DVb, which represents the estimated divergence of the sample from the MRCA of that clade. Go [here](conceptual_intro.md) for a conceptual introduction to divergence.
*   Probability Score, which represents the overall probability that the sample supports a clade that is within your `--max_pi` threshold. The Probability Score only has information in the Bayesian implementation of PHLAME, and will either be 1 or 0 in the MLE version. 

You may notice that the total relative abundances across any set of non-overlapping clades does not add up to 1. This is intended and suggests that the sample may harbor intraspecies diversity that is novel with regards to any of the clades in the reference set.
