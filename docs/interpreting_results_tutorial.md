#  Intepreting PHLAME results

## 1. Visualizing classification results

The compressed data file returned by `phlame classify` has useful information that can be used to help visualize classification decisions. You can view the output of a data file with the command `phlame plot`; this is  much more useful when running `-m bayesian`, as you will be able to visualize full posterior distributions over each parameter. For this, a pre-made data file has been included in `example`.

```
phlame plot -f skin_mg_frequencies.csv -d skin_mg_fitinfo_bayesian.data -o skin_mg_frequencies_plot.pdf
```

![alt text](example/plot.png)

Each clade will have four relevant plots. From left to right, they are: 
*   [1] A histogram of the actual number of reads supporting each clade-specific allele (red), as well as all alleles at the same positions (gray). 
*   [2] The posterior probability over the pi parameter (equivalent to DVb), as well as the 95% highest posterior density interval (gray bar). 
*   [3] The posterior probability over lambda , which represents the average read depth across 
*   [4] The posterior probability density over the relative abundance of the clade in the sample.

In this particular example, The posterior densities all have fairly high spreads because the sequencing depth is low. Visualizing the posterior densities helps us make detection decisions. For example, while clade C.2 is has enough density below our threshold to be detected, very little density is actually centered around pi values of 0. If we wanted to limit our detections to only strains that we think are for sure within the mRCA of C.2, we might reject this detection (for example, via the --hpd threshold in `phlame classify`)


## 2. Analyzing results at specific phylogenetic levels

By default, PHLAME reports results for every clade in the phylogeny simultaneously. In order to generate downstream analyses like taxonomic bar plots and ordination plots, you need to select a set of non-overlapping clades (a level) to analyze results at.

![alt text](docs/profile.png)

If you already know what phylogenetic level you are interested in, you can ask PHLAME to classify only at those clades using the `-l` parameter in `phlame classify`. You can specify clades as either a string list of clade names or a clade_IDs file.

```
$ phlame classify -i skin_mg_A.sam -c Cacnes_db.classifier -r Pacnes_C1.fasta -m bayesian -o skin_mg_frequencies.csv -p skin_mg_fitinfo.data -l 'C.1,C.2'
```
or
```
$ phlame classify -i skin_mg_A.sam -c Cacnes_db.classifier -r Pacnes_C1.fasta -m bayesian -o skin_mg_frequencies.csv -p skin_mg_fitinfo.data -l Cacnes_cladeIDs.txt
```

Note that total inferred frequencies at a specific phylogenetic level may sum to less than 1, reflecting the presence of uncharacterized or novel clades in the sample. This effect is more pronounced at more-finely resolved phylogenetic clades, as you are less likely to also have that clade in an unrelated sample. In rare cases, total frequencies may sum to a value above 1. To address this, we recommend normalizing cases that sum above 1 down to 1.

One useful way to analyze PHLAME results is using a coverage versus percent called plot. In this plot, the total inferred frequency of a sample at a specific phylogenetic level is plotted against the mean coverage of that sample. As coverage decreases, PHLAME becomes less confident about individual calls and therefore assigns less of the sample. On the other hand, samples with high coverage that still have a lower percent assigned likely harbor uncharacterized or novel clades.

![alt text](docs/coverage.png)