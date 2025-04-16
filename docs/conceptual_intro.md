# Conceptual introduction to PHLAME

## Reference databases and Novel diversity

Reference databases are a useful tool to characteize intraspecies diversity in a metagenomic sample, particularly [for sample types](https://www.sciencedirect.com/science/article/pii/S0022202X21023514) that make it difficult to recover genomes directly from metagenomes. A challenge common to all taxonomic reference databases is that they cannot resolve novel diversity not found in the reference database. This issue is particularly xxx at when xxx intraspecies diversity, as there are only a small number of instances where the *exact* same strains in a reference database are expected to also be a sample. 

Because of the small number of genetic differences between strains of the same species, novel strains are often reported as combinations of known reference genomes, although these reference genomes are likely not truly in the sample

## Divergence (DV~b~)

Novel strains in a sample are expected to share evolutionary history with the genomes in a reference database up until a certain point. We refer to the degree of shared evolutionary history as Divergence (DV~b~), which is calculated with respect to each branch (b) of the phylogeny. Formally, Divergence is defined as the ratio of shared to unshared branch lengths, and varies from 0 to 1.

![alt text](docs/divergence.png)

We calculate Divergence in metagenomic samples using specific marker SNVs that we assume accumulate in a clock-like fashion along a specific branch (we call these clade-specific SNVs). A novel strain should share clade-specific SNVs up until it diverges with all our known strains. Any clade-specific SNVs past that point should be missing from the sample.

![alt text](docs/divergence_reads.png)

## Posterior inference of Divergence

At low sequencing depths, it is hard to distringuish the clade-specific SNVs that are truly missing from a sample from those missed by chance. PHLAME uses a Bayesian model to quantify the uncertainty of DV~b~ estimates in a sample. Uncertainty is shown through a posterior probability distribution, where higher values indicate more confidence that DV~b~ is a certain value.

![alt text](docs/posteriors.png)


At higher sequencing depths, the proportion of clade-specific SNVs that are truly missing is more obvious, and this is reflected in a probability distribution that places more probability around a specific value (more confident). At lower sequencing depths, the reverse is true and this is reflected as a more spread-out probability distribution.



