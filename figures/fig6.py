"""
author: evanqu
date: 2024/05/02 04:08
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ete3
import scipy
import seaborn as sns

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures')

import fig_helper_functions as helper

#%% Set plotting parameters

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

fmt={'fontsize':15,
     'fontname':'Helvetica'}

%matplotlib auto

#%% Functions

def read_sample_frequencies_multi(sample_names,
                                  phlame_out_dir_ls,
                                  reference_genome_ls,
                                  reference_genome_names_ls):
    '''
    Read and consolidate frequency files from multiple phlame output directories
    '''
    
    jt_frequencies = pd.DataFrame()

    for i, (phlame_out_dir, reference_genome) in enumerate(zip(phlame_out_dir_ls,
                                                               reference_genome_ls)):

        ref_frequencies = pd.DataFrame()

        for sample_name in sample_names:
        
            frequencies_file = f'{phlame_out_dir}/{sample_name}_ref_{reference_genome}_frequencies.csv'
            
            sample_freqs = helper.Frequencies(frequencies_file)
            sample_freqs.freqs = sample_freqs.freqs['Relative abundance']
            sample_freqs.freqs.name = sample_name

            # Rename clades to include prefix of what ref genome
            sample_freqs.freqs.index = [f'{reference_genome_names_ls[i]}.'+clade for clade in sample_freqs.freqs.index]

            ref_frequencies = pd.concat((ref_frequencies,sample_freqs.freqs), axis=1)

        jt_frequencies = pd.concat((jt_frequencies,ref_frequencies), axis=0)
    
    return jt_frequencies.T

def read_alignment_stats_multi(alignment_stats_ls,
                               reference_genome_names_ls):
    '''
    Read and consolidate multiple alignment stats files.
    '''

    alignment_stats_df = pd.DataFrame()

    for alignment_stats_file, refgenome_name in zip(alignment_stats_ls, reference_genome_names_ls):

        alignment_stats = pd.read_csv(alignment_stats_file, index_col=1)['Number Aligned Once']
        
        alignment_stats.name = refgenome_name

        alignment_stats_df = pd.concat((alignment_stats_df,alignment_stats), axis=1)

    return alignment_stats_df


def parse_coverage_multi(sample_names,
                         counts_dir,
                         reference_genome_ls,
                         reference_genome_names_ls):

    cov_cat = pd.DataFrame()
    
    for refgenome,refgenome_name in zip(reference_genome_ls,reference_genome_names_ls):

        cov_df = helper.parse_coverage(sample_names, counts_dir, refgenome)

        cov_df.columns = [refgenome_name]

        cov_cat = pd.concat((cov_cat, cov_df), axis=1)

    return cov_cat

def read_bracken_frequencies(bracken_dir, sample_names):

    bracken_frequencies_df = pd.DataFrame()
    bracken_nreads_df = pd.DataFrame()
    
    for sample in sample_names:
        
        bracken_file = f'{bracken_dir}/{sample}.bracken'
        
        bracken_info = pd.read_csv(bracken_file, sep='\t', index_col=0, header=0)
        
        bracken_nreads = bracken_info['new_est_reads']
        bracken_freqs = bracken_info['fraction_total_reads']

        bracken_freqs.name = sample
        bracken_nreads.name = sample
        
        bracken_frequencies_df = pd.concat((bracken_frequencies_df, bracken_freqs), axis=1)
        bracken_nreads_df = pd.concat((bracken_nreads_df, bracken_nreads), axis=1)
        
    return bracken_frequencies_df.fillna(0).T, bracken_nreads_df.fillna(0).T

def calc_MAP(chain, bins=False):

    # if bins == False:
    #     bins = 100

    counts, bins = np.histogram(chain, bins=bins)
    max_idx = np.argmax(counts)
    
    return bins[max_idx]


def get_hpd(chain, interval_size=0.95):
    """
    Returns highest probability density region for a given interval
    """
    # Get sorted list
    d = np.sort(np.copy(chain))

    # Number of total samples taken
    n = len(chain)
    
    # Get interval size that should be included in HPD
    interval = np.floor(interval_size * n).astype(int)
    
    # Get width (in units of param) of all intervals 
    int_width = d[interval:] - d[:n-interval]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+interval]])

def get_hpds(phlame_out_dir, sample_names, reference_genome_ls, reference_genome_names_ls,
             cidx_translate):

    pi_maps = np.full((len(sample_names), len(phlame_frequencies.columns)),-1, dtype=np.float)
    pi_hpd_lower = np.full((len(sample_names), len(phlame_frequencies.columns)),-1, dtype=np.float)
    pi_hpd_upper = np.full((len(sample_names), len(phlame_frequencies.columns)),-1, dtype=np.float)

    for sidx, sample in enumerate(sample_names):

        for ridx, refgenome in enumerate(reference_genome_ls):

            path_to_sample_data = f"{phlame_out_dir}/{sample}_ref_{refgenome}_fitinfo.data"

            data = helper.FrequenciesData(path_to_sample_data)

            for cidx, chain in enumerate(data.chain):

                if len(chain) > 0:
                    pi_chain = chain['pi']
                else:
                    continue

                pi_map = calc_MAP(pi_chain, bins = np.arange(0,1.01,0.01))
                hpd_lower, hpd_upper = get_hpd(pi_chain, interval_size=0.95)

                pi_maps[sidx,cidx_translate[ridx]+cidx] = pi_map
                pi_hpd_lower[sidx,cidx_translate[ridx]+cidx] = hpd_lower
                pi_hpd_upper[sidx,cidx_translate[ridx]+cidx] = hpd_upper
    
    return pi_maps, pi_hpd_lower, pi_hpd_upper

def dereplicate_by_subject(md_sid, md_cov, md_cst):
    '''
    Dereplicate by subject.
    Pick the highest depth sample for each subject, prioritizing samples with either the iners or Gardnerella dominated CST (III and IV)
    '''
    
    derep_bool = pd.Series(False, index = coverage.index)

    for subj in np.unique(md_sid):
        
        subj_bool = md_sid == subj
        this_subj_cst = md_cst.loc[subj_bool]
        this_subj_cst_bool = ((this_subj_cst.str.startswith('III')) | (this_subj_cst.str.startswith('IV')))
        # this_subj_cst_bool = ((this_subj_cst.str.startswith('III')))

        if np.sum(this_subj_cst_bool) > 0:
            this_subj_coverage = md_cov.loc[subj_bool][this_subj_cst_bool]
            # print(this_subj_coverage)

            max_cov_sample = this_subj_coverage.index[np.where(this_subj_coverage == max(this_subj_coverage[this_subj_cst_bool]))]

        else:
            this_subj_coverage = md_cov.loc[subj_bool]

            max_cov_sample = this_subj_coverage.index[np.where(this_subj_coverage == max(this_subj_coverage))]

        derep_bool.loc[max_cov_sample] = True

    return derep_bool


#%% Figure 6A: Gardnerella trees
# =============================================================================

gvaginalis_tree = ete3.Tree("fig6/trees/Gvaginalis_HKY85_calls_2", format=0)
gpiotii_tree = ete3.Tree("fig6/trees/Gpiotii_HKY85_calls_2", format=0)
gleopoldii_tree = ete3.Tree("fig6/trees/Gleopoldii_HKY85_calls_2", format=0)
ggreenwoodii_tree = ete3.Tree("fig6/trees/Ggreenwoodii_HKY85_calls_2", format=0)    

# =============================================================================
# Add metadata to nodes
# =============================================================================

clade2color_dct = {'GS1':'gainsboro',
                  'GS2':'k',
                  'GS3':'gainsboro',
                  'GS4':'k',
                  'GS5':'gainsboro',
                  'GS6':'k',
                  'GS7':'gainsboro',
                  'GS8':'k',
                  '-1':'k'}

clade_IDs = pd.read_csv('fig6/Gardnerella_clade_IDs.txt', sep="\t", index_col=0, header=None)

# Color each leaf by phylogroup identity
for leaf in gvaginalis_tree.iter_leaves():
    clade=clade_IDs.loc[leaf.name].values
    if len(clade) > 1:
        clade = clade[0][0]
    else:
        clade = clade[0]
    slstshape=clade2color_dct[clade]
    # Add SLST features
    leaf.add_face(ete3.RectFace(1,2,fgcolor=slstshape,bgcolor=slstshape),
                  column=0,position = "aligned")

for leaf in gpiotii_tree.iter_leaves():
    clade=clade_IDs.loc[leaf.name].values
    if len(clade) > 1:
        clade = clade[1][0]
    else:
        clade = clade[0]

    slstshape=clade2color_dct[clade]
    # Add SLST features
    leaf.add_face(ete3.RectFace(1,2,fgcolor=slstshape,bgcolor=slstshape),
                  column=0,position = "aligned")
    
for leaf in gleopoldii_tree.iter_leaves():
    clade=clade_IDs.loc[leaf.name].values[0]
    slstshape=clade2color_dct[clade]
    # Add SLST features
    leaf.add_face(ete3.RectFace(1,2,fgcolor=slstshape,bgcolor=slstshape),
                  column=0,position = "aligned")


for leaf in ggreenwoodii_tree.iter_leaves():
    clade=clade_IDs.loc[leaf.name].values[0]
    slstshape=clade2color_dct[clade]
    # Add SLST features
    leaf.add_face(ete3.RectFace(1,2,fgcolor=slstshape,bgcolor=slstshape),
                  column=0,position = "aligned")
    
general_ts = ete3.TreeStyle()
general_ts.show_leaf_name=False
general_ts.scale = 0.005
general_ts.optimal_scale_level = 'full'
# Remove blue dot on every node
nstyle = ete3.NodeStyle()
nstyle['shape']=''; nstyle['size']=0
nstyle['vt_line_width']=0.025
nstyle['hz_line_width']=0.025



for node in gvaginalis_tree.traverse():
   node.set_style(nstyle)
for node in gpiotii_tree.traverse():
   node.set_style(nstyle)
for node in gleopoldii_tree.traverse():
    node.set_style(nstyle)
for node in ggreenwoodii_tree.traverse():
    node.set_style(nstyle)
general_ts.legend_position=3


# gvaginalis_tree.show(tree_style=general_ts)
# gpiotii_tree.show(tree_style=general_ts)
gleopoldii_tree.show(tree_style=general_ts)

gvaginalis_tree.render("fig6/fig6a1.pdf", tree_style=general_ts, w=20, units="in", dpi=300)
gpiotii_tree.render("fig6/fig6a2.pdf", tree_style=general_ts, w=20, units="in", dpi=300)
gleopoldii_tree.render("fig6/fig6a3.pdf", tree_style=general_ts, w=20, units="in", dpi=300)
ggreenwoodii_tree.render("fig6/fig6a4.pdf", tree_style=general_ts, w=20, units="in", dpi=300)

#%% Load in Gardnerella PHLAME results
# =============================================================================

phlame_out_dir = 'fig6/6-frequencies'
counts_dir='fig6/5-counts'
bracken_dir = 'fig6/bracken'

metadata_file = 'fig6/vaginal_mg_metadata_standardized.csv'
samples_csv_file = 'fig6/samples_all.csv'

Gvag_refgenome = 'Gvaginalis_FDAARGOS_568'
Gpio_refgenome = 'Gpiotii_ASM339758'
Gleo_refgenome = 'Gleopoldii_6420B'
Ggre_refgenome = 'Ggreenwoodii_ASM26363'


#%% Read in sample frequencies 
# =============================================================================
samples_csv = pd.read_csv(samples_csv_file)
sample_names = samples_csv['Sample']

phlame_frequencies = read_sample_frequencies_multi(sample_names,
                                                   [phlame_out_dir,phlame_out_dir,phlame_out_dir,phlame_out_dir],
                                                   [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
                                                   ['Gvag','Gpio','Gleo','Ggre'])
clade_names_all = phlame_frequencies.columns

phlame_frequencies[phlame_frequencies < 0.01] = 0

##### Read in coverage info #####
coverage = parse_coverage_multi(sample_names,
                                counts_dir,
                                [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
                                ['Gvag','Gpio','Gleo','Ggre'])
    
bracken_frequencies, bracken_nreads = read_bracken_frequencies(bracken_dir,
                                                               sample_names)

##### Read in metadata #####
md = pd.read_csv( metadata_file, sep=',', index_col=0)
md_cst = md.loc[sample_names]['CST']
md_age = md.loc[sample_names]['Age_cat']
md_ph = md.loc[sample_names]['pH_cat']
md_nugent = md.loc[sample_names]['Nugent_cat']
md_bv = md.loc[sample_names]['Amsel-BV']
md_symbv = md.loc[sample_names]['Sym-Amsel-BV']
md_pj = md.loc[sample_names]['Source Study']
md_sid = md.loc[sample_names]['SID']
md_cov = md.loc[sample_names]['Coverage']

#%% Define levels
# =============================================================================

gvag_bool = [col.startswith('Gvag') for col in phlame_frequencies.columns]
gpio_bool = [col.startswith('Gpio') for col in phlame_frequencies.columns]
gleo_bool = [col.startswith('Gleo') for col in phlame_frequencies.columns]
ggre_bool = [col.startswith('Ggre') for col in phlame_frequencies.columns]

# # Normalize frequencies that sum above 1 to 1
norm_to_1 = phlame_frequencies.iloc[:, 0:2].loc[(phlame_frequencies.iloc[:, 0:2].sum(axis=1) > 1)]
phlame_frequencies.iloc[:, 0:2].loc[(phlame_frequencies.iloc[:, 0:2].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = phlame_frequencies.iloc[:, 2:4].loc[(phlame_frequencies.iloc[:, 2:4].sum(axis=1) > 1)]
phlame_frequencies.iloc[:, 2:4].loc[(phlame_frequencies.iloc[:, 2:4].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = phlame_frequencies.iloc[:, 4:6].loc[(phlame_frequencies.iloc[:, 4:6].sum(axis=1) > 1)]
phlame_frequencies.iloc[:, 4:6].loc[(phlame_frequencies.iloc[:, 4:6].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = phlame_frequencies.iloc[:, 6:8].loc[(phlame_frequencies.iloc[:, 6:8].sum(axis=1) > 1)]
phlame_frequencies.iloc[:, 6:8].loc[(phlame_frequencies.iloc[:, 6:8].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)

# Normalize frequencies across different reference genomes 
# by their mean coverage across informative positions
phlame_frequencies_norm = phlame_frequencies.copy()

coverage_norm = coverage.div(coverage.sum(1),0)
for name, bool_ in zip(['Gvag','Gpio','Gleo','Ggre'],
                       [gvag_bool,gpio_bool,gleo_bool,ggre_bool]):
    phlame_frequencies_norm.loc[:,bool_] = phlame_frequencies_norm.loc[:,bool_].mul(coverage_norm[name], axis=0)


#%% QC plot -> Percent called vs coverage
# =============================================================================

perc_called = np.sum(phlame_frequencies_norm,1)
coverage_sum = coverage.sum(1)

fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, .3]})
fig.set_size_inches(7,4)

axs[0].scatter(coverage_sum, perc_called,
                color='k', alpha=.4, s=5)
axs[0].axhline(y=1, color='k', alpha=.3)
# axs[0].axvline(x=1, color='k', alpha=.3)    
axs[0].set_xscale('log')
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].set_xlabel('Mean Gardnerella coverage \n(across 4 reference genomes, unique regions)', **fmt); axs[0].set_ylabel('Percent of sample called', **fmt)
# axs[0].legend()

axs[1].hist(perc_called[(coverage_sum > 1).values], bins=np.arange(0,1.01,0.02), color='k', alpha=0.9, orientation="horizontal")
axs[1].set_title('Mean coverage \n>1X', **fmt)
axs[1].set_yticks([])
axs[1].set_xlabel('# of Samples', **fmt)
axs[1].tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout()

print(f"% of samples above 1X coverage where >80% of the sample was classifiable: {np.sum(perc_called[(coverage_sum > 1).values] > 0.8)/np.sum(coverage_sum > 1):.2f}")
print(f"% of samples above 5X coverage where >80% of the sample was classifiable: {np.sum(perc_called[(coverage_sum > 5).values] > 0.8)/np.sum(coverage_sum > 5):.2f}")

# fig.savefig('supplemental/figS11.pdf', dpi=300, format='pdf')


#%% KS test C. acnes versus Gardnerella percent called

from scipy.stats import ks_2samp

Cacnes_perc_called = perc_called_phylo[(coverage[0] > 1).values] # From fig5.py

Gardnerella_perc_called = perc_called[(coverage_sum > 1).values]

_, pval = ks_2samp(Cacnes_perc_called, Gardnerella_perc_called)


#%% Implement some sample filters
# =============================================================================

# Coverage filter
coverage_bool = coverage.sum(1) > 3

# Percent called filter
perc_called_bool = perc_called > 0.8

derep_bool = dereplicate_by_subject(md_sid, md_cov, md_cst)

bool_include = coverage_bool & derep_bool

#%% Figure 6B: Does the proportion of a particular Gardnerella clade correlate with the proportion of L. iners?
# ==============================================================================================================

fig, axs = plt.subplots(1,2)
fig.set_size_inches(8,4)
nclades = len(phlame_frequencies_norm.columns)

rs = np.zeros(nclades); pvals = np.zeros(nclades)
nhits = 0

for i, clade in enumerate(phlame_frequencies_norm.columns):

    # Get just the samples that have the clade at non-zero frequency
    # clade_bool = phlame_frequencies_norm[clade] > 0.01
    clade_bool = [True]*len(phlame_frequencies_norm)
    # ref_cov_bool = coverage[clade.split('.')[0]] > 5
    frequencies_clade = phlame_frequencies_norm[clade][clade_bool & bool_include]

    r, pval = scipy.stats.spearmanr(frequencies_clade, 
                                    bracken_frequencies['Lactobacillus iners'][clade_bool & bool_include])

    pval = pval*nclades # Bonferroni correction
    rs[i] = r
    pvals[i] = pval

    if pval < 0.01:
        axs[nhits].scatter(bracken_frequencies['Lactobacillus iners'][clade_bool & bool_include], frequencies_clade,
                     color='k', alpha=.6, s=7)
        axs[nhits].set_title(f"p={pval:.2E}, $r^2$={r:.2f}")
        axs[nhits].set_ylabel(f"GS6 frequency", **fmt)
        axs[nhits].set_xlabel('L. iners frequency', **fmt)
        axs[nhits].tick_params(axis='both', which='major', labelsize=12)
        # axs[nhits].set_xlim(10e-5,1)
        axs[nhits].set_xscale('log')
        axs[nhits].set_yscale('log')
        nhits += 1

        pval_good = pval
        r_good = r

fig.tight_layout()
# fig.savefig('fig6/fig6b2.pdf', dpi=300, format='pdf')

###### Plot p values ######
fig2, axs2 = plt.subplots()
fig2.set_size_inches(3,2)
axs2.scatter(np.arange(0,nclades,1), np.log10(np.sort(pvals[pvals.nonzero()])),
             color='k')
axs2.axhline(np.log10(0.01/nclades), color='k')
# axs2.plot(np.arange(0,6), np.log10(np.arange(1,7)*(0.05/6)), color='k', alpha=.9)
# axs2.set_yscale('log')
axs2.set_ylabel('Log10(p-value)', **fmt)
axs2.set_xlabel('Rank', **fmt)
axs2.tick_params(axis='both', which='major', labelsize=12)
axs2.set_xticks([])

fig2.tight_layout()
# fig2.savefig('fig6/fig6b1.pdf', dpi=300, format='pdf')

#%% Figure 6B Supplemental: Is there any association between G. leopoldii reads and L. iners frequency?
# ==============================================================================================================

fig, axs = plt.subplots()
fig.set_size_inches(5.5,5)
r, pval = scipy.stats.spearmanr(coverage['Gleo'][bool_include], 
                                bracken_frequencies['Lactobacillus iners'][bool_include])

axs.scatter(bracken_frequencies['Lactobacillus iners'][bool_include], coverage['Gleo'][bool_include],
            color='k', alpha=.6, s=8)
axs.set_title(f"p={pval:.2f}, $r^2$={r:.2f}")
axs.set_xlabel('L. iners frequency', **fmt)
axs.set_ylabel('Coverage across G. leopoldii reference', **fmt)
axs.xaxis.set_tick_params(labelsize=12)
axs.yaxis.set_tick_params(labelsize=12)
axs.set_yscale('log')
axs.set_xscale('log')

fig.tight_layout()
# fig.savefig('fig6/figs61.pdf', dpi=300, format='pdf')

#%% Figure 6B Supplemental: Split up the association by common confounders (Study)

# Definitely study, nugent, ph
fig, axs = plt.subplots(4)
fig.set_size_inches(5,14)

for i, study in enumerate(np.unique(md_pj)):
    
    study_bool = md_pj == study
    frequencies_clade = phlame_frequencies_norm['Gleo.GS6'][bool_include & study_bool]
    frequencies_liners = bracken_frequencies['Lactobacillus iners'][bool_include & study_bool].values

    r, pval = scipy.stats.spearmanr(frequencies_clade, 
                                    frequencies_liners)
    
    liners_sorted_idx = np.flip(np.argsort(-frequencies_liners))

    nsamples = np.sum(study_bool & bool_include)

    axs[i].scatter(frequencies_liners[liners_sorted_idx], frequencies_clade.iloc[liners_sorted_idx],
                   color='k', alpha=.6, s=7)
    # axs[i].bar(np.arange(0,nsamples,1),frequencies_clade.iloc[liners_sorted_idx],
    #            label = 'Gleo.GS6', alpha=.85, color='k')
    # axs[i].bar(np.arange(0,nsamples,1),-frequencies_liners[liners_sorted_idx],
    #            label = 'L. iners', alpha=.2, color='k')
    
    axs[i].set_title(f"{study}: p={pval:.3f}, $r^2$={r:.2f}")
    axs[3].set_xlabel('L. iners frequency', **fmt)
    axs[i].set_ylabel('GS6 frequency', **fmt)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')

    axs[i].tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout()
# fig.savefig('supplemental/figS12a.pdf', dpi=300, format='pdf')

#%% Figure 6B Supplemental: Split up the association by common confounders (Nugent score)
# ==============================================================================================================

fig, axs = plt.subplots(4)
fig.set_size_inches(6,10)

for i, nugent in enumerate(np.unique(md_nugent)):

    nugent_bool = md_nugent == nugent
    frequencies_clade = phlame_frequencies_norm['Gleo.GS6'][bool_include & nugent_bool]
    frequencies_liners = bracken_frequencies['Lactobacillus iners'][bool_include & nugent_bool].values

    r, pval = scipy.stats.spearmanr(frequencies_clade,
                                    frequencies_liners)
    
    liners_sorted_idx = np.flip(np.argsort(-frequencies_liners))

    nsamples = np.sum(nugent_bool & bool_include)
    axs[i].bar(np.arange(0,nsamples,1),frequencies_clade.iloc[liners_sorted_idx],
               label = 'Gleo.GS6', alpha=.85, color='k')
    axs[i].bar(np.arange(0,nsamples,1),-frequencies_liners[liners_sorted_idx],
               label = 'L. iners', alpha=.2, color='k')
    
    axs[i].set_title(f"Nugent = {nugent}: p={pval:.3f},  $r^2$={r:.2f}")
    axs[i].set_xticks([])
    axs[i].tick_params(axis='both', which='major', labelsize=12)
    axs[i].axhline(0, color='k', alpha=.5)
    axs[i].set_ylim(-1,1)

# fig.savefig('supplemental/figS12b.pdf', dpi=300, format='pdf')

#%% Figure 6C: Search for novel clades driving true effects
# =============================================================================

hpd_threshold = 0.2

cidx_translate = {0:0,
                  1:2,
                  2:4,
                  3:6}

# Actual metagenomes

pi_maps, pi_hpd_lower, pi_hpd_upper = get_hpds(phlame_out_dir, sample_names,
                                               [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
                                               ['Gvag','Gpio','Gleo','Ggre'],
                                               cidx_translate)

hpd_mask = (pi_hpd_lower != -1) & (pi_hpd_upper != -1) & (pi_hpd_upper - pi_hpd_lower < 0.2)

# Read in simulated data

benchmark_path_to_samples_csv = 'fig6/simulations/benchmark_samples.csv'
benchmark_phlame_out_dir = 'fig6/simulations/6-frequencies_benchmark'

benchmark_sample_names = pd.read_csv(benchmark_path_to_samples_csv)['Sample']


pi_maps_benchmark, pi_hpd_lower_benchmark, pi_hpd_upper_benchmark = get_hpds(benchmark_phlame_out_dir, benchmark_sample_names,
                                                                            [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
                                                                            ['Gvag','Gpio','Gleo','Ggre'],
                                                                            cidx_translate)

hpd_mask_benchmark = (pi_hpd_lower_benchmark != -1) & (pi_hpd_upper_benchmark != -1) & (pi_hpd_upper_benchmark - pi_hpd_lower_benchmark < 0.2)


#%% Figure 6C: Plot distribution of high confidence pis for every clade
# =============================================================================

from scipy.stats import ks_2samp

idx_to_look = np.where(np.in1d(phlame_frequencies.columns, 
                               ['Gvag.GS1','Gvag.GS2','Gpio.GS3','Gpio.GS4','Gleo.GS5','Gleo.GS6','Ggre.GS7','Ggre.GS8']))[0]

names = ['GS1','GS2','GS3','GS4','GS5','GS6','GS8','GS9/GS10']
fig, axs = plt.subplots(8)
fig.set_size_inches(5,20)

for idx_, clade in enumerate(idx_to_look):
    axs[idx_].hist(pi_maps[hpd_mask[:,clade],clade], bins=np.arange(0,1.01,0.01),
                color='grey', edgecolor='k', label='Metagenomes')
    axs[idx_].set_ylabel('# of Samples', **fmt)
    axs[idx_].tick_params(axis='both', which='major', labelsize=12)
    axs[idx_].set_title(f"{names[clade]}", **fmt)

    pos_counts, _ = np.histogram(pi_maps[hpd_mask[:,clade],clade], bins=np.arange(0,1.01,0.01))

    counts, bins, patches = axs[idx_].hist(pi_maps_benchmark[hpd_mask_benchmark[:,clade],clade],
                                      bins=np.arange(0,1.01,0.01), color='b', edgecolor='b', alpha=0.5,
                                      label='Simulations')

    neg_counts, _ = np.histogram(pi_maps_benchmark[hpd_mask_benchmark[:,clade],clade],
                                 bins=np.arange(0,1.01,0.01))
    
    for patch in patches:
        patch.set_height(-patch.get_height())  # Invert the height of the bars

    axs[idx_].set_ylim(max(neg_counts)*-1.4,max(pos_counts)*1.1)

    # KS test between synthetic and real metagenomes
    _, pval = ks_2samp(pi_maps[hpd_mask[:,clade],clade], pi_maps_benchmark[hpd_mask_benchmark[:,clade],clade])

    axs[idx_].text(0.7, 0.9, '$p_{adj}$'+f'={pval*len(idx_to_look):.2E}', transform=axs[idx_].transAxes, fontsize=12)

    print(pval*len(idx_to_look))

    # axs[idx_].legend(fontsize=12)


axs[7].set_xlabel('MAP \u03C0 (High confidence)', **fmt)

fig.tight_layout()

# fig.savefig('supplemental/figS16A.pdf', dpi=300, format='pdf')

#%% Print out the number of benchmark samples that have a high confidence pi above 0.35

total_calls_MGs = 0
calls_count_MGs = 0

total_calls_sims = 0
calls_count_sims = 0

for idx_, clade in enumerate(idx_to_look):
    print(f"{names[idx_]}: {np.sum(pi_maps_benchmark[hpd_mask_benchmark[:,clade],clade] > 0.35)}")
    total_calls_sims += np.sum(hpd_mask_benchmark[:,clade])
    calls_count_sims += np.sum(pi_maps_benchmark[hpd_mask_benchmark[:,clade],clade] > 0.35)

    total_calls_MGs += np.sum(hpd_mask[:,clade])
    calls_count_MGs += np.sum(pi_maps[hpd_mask[:,clade],clade] > 0.35)

print(f"Total calls > 0.35: {calls_count_MGs}/{total_calls_MGs} or {calls_count_MGs/total_calls_MGs:.4f}")
print(f"Total calls > 0.35: {calls_count_sims}/{total_calls_sims} or {calls_count_sims/total_calls_sims:.4f}")

# Fishers exact test
oddsratio, pvalue = scipy.stats.fisher_exact([[calls_count_MGs, total_calls_MGs-calls_count_MGs],
                                              [calls_count_sims, total_calls_sims-calls_count_sims]])

#%% Figure 6C Supplemental: Look at MAP pi breakdown by study
# =============================================================================

fig, axs = plt.subplots(8)
fig.set_size_inches(5,20)

pjs = ['UMB-HMP', 'VIRGO', 'VMRC']

for i, (clade, name) in enumerate(zip(['Gvag.GS1','Gvag.GS2','Gpio.GS3','Gpio.GS4','Gleo.GS5','Gleo.GS6','Ggre.GS7','Ggre.GS8'],
                                      ['GS1','GS2','GS3','GS4','GS5','GS6','GS8','GS9/GS10'])):

    idx_to_look = np.where(np.in1d(phlame_frequencies.columns, [clade]))[0][0]

    df_melt = pd.DataFrame()

    for c, category in enumerate(np.unique(md_pj)):
            
            category_bool = md_pj == category
    
            if np.sum(category_bool) > 0:
    
                pis_ = pi_maps[:,idx_to_look][hpd_mask[:,idx_to_look] & category_bool]
    

                df_melt = pd.concat([df_melt, pd.DataFrame({'MAP pi':pis_,
                                                            'Category':[category]*len(pis_)})])


    # p value kruskal wallis
    _, pval = scipy.stats.kruskal(*[df_melt['MAP pi'][df_melt['Category'] == category].values for category in pjs])

    sns.stripplot(data=df_melt, y='Category', x="MAP pi", ax=axs[i], alpha=0.5, color='k')

    # Add median lines
    sns.boxplot(data=df_melt, y='Category', x="MAP pi", ax=axs[i],
                meanline=True, showmeans=True, meanprops={'color':'k','ls':'-','lw':2},
                medianprops={'visible':False}, whiskerprops={'visible':False},
                showfliers=False, showcaps=False, showbox=False)
    
    if pval*8 < 0.05:
        axs[i].set_title('$p_{adj}$'+f' = {pval*8:.4f}', fontsize=12)
    axs[i].set_xlabel(f'MAP $\pi$ (High confidence)' , **fmt)
    axs[i].set_xlim(-0.05,1.05)
    axs[i].tick_params(axis='both', which='major', labelsize=12)
    axs[i].set_ylabel(name, **fmt)


fig.tight_layout()
# fig.savefig('supplemental/figS17A.pdf', dpi=300, format='pdf')

#%% Look at MAP pi breakdown for GS4 specifically

import itertools
            
# pairwise rank sum test
# get pairwise combinations of categories
combinations = list(itertools.combinations(np.unique(md_pj)[1:],2))
for comb in combinations:
    _, pval = scipy.stats.ranksums(df_melt['MAP pi'][df_melt['Category'] == comb[0]].values,
                                   df_melt['MAP pi'][df_melt['Category'] == comb[1]].values)
    print(f"{comb[0]} vs {comb[1]}: {pval:2E}")


#%% Supplemental: ID specific samples with high abundances of very diverged things
# =============================================================================

# Get matrix of per-clade frequency call regardless of detected or not

inferred_highpi_frequencies = np.full((len(sample_names), 
                               len(phlame_frequencies.columns)),
                               -1, dtype=np.float)

lambdas = np.full((len(sample_names),
                   len(phlame_frequencies.columns)),
                   -1, dtype=np.float)

for sidx, sample in enumerate(sample_names):

    for ridx, refgenome in enumerate([Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome]):

        path_to_sample_data = f"{phlame_out_dir}/{sample}_ref_{refgenome}_fitinfo.data"

        data = helper.FrequenciesData(path_to_sample_data)

        for cidx, MAPest in enumerate(data.counts_MAP):

            if len(MAPest) == 0:
                lambda_est = 0
            else:
                lambda_est = MAPest['a']/MAPest['b']
            
            total_lambdaest = data.total_MLE[cidx][0]

            inferred_highpi_frequencies[sidx,cidx_translate[ridx]+cidx] = min(1,lambda_est/total_lambdaest)


inferred_highpi_frequencies_norm = pd.DataFrame(inferred_highpi_frequencies.copy(),
                                                index=sample_names,
                                                columns=phlame_frequencies.columns)


# # Normalize frequencies that sum above 1 to 1
norm_to_1 = inferred_highpi_frequencies_norm.iloc[:, 0:2].loc[(inferred_highpi_frequencies_norm.iloc[:, 0:2].sum(axis=1) > 1)]
inferred_highpi_frequencies_norm.iloc[:, 0:2].loc[(inferred_highpi_frequencies_norm.iloc[:, 0:2].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = inferred_highpi_frequencies_norm.iloc[:, 2:4].loc[(inferred_highpi_frequencies_norm.iloc[:, 2:4].sum(axis=1) > 1)]
inferred_highpi_frequencies_norm.iloc[:, 2:4].loc[(inferred_highpi_frequencies_norm.iloc[:, 2:4].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = inferred_highpi_frequencies_norm.iloc[:, 4:6].loc[(inferred_highpi_frequencies_norm.iloc[:, 4:6].sum(axis=1) > 1)]
inferred_highpi_frequencies_norm.iloc[:, 4:6].loc[(inferred_highpi_frequencies_norm.iloc[:, 4:6].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = inferred_highpi_frequencies_norm.iloc[:, 6:8].loc[(inferred_highpi_frequencies_norm.iloc[:, 6:8].sum(axis=1) > 1)]
inferred_highpi_frequencies_norm.iloc[:, 6:8].loc[(inferred_highpi_frequencies_norm.iloc[:, 6:8].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)


# Normalize frequencies across different reference genomes 
# by their mean coverage across informative positions

coverage_norm = coverage.div(coverage.sum(1),0)
for name, bool_ in zip(['Gvag','Gpio','Gleo','Ggre'],
                       [gvag_bool,gpio_bool,gleo_bool,ggre_bool]):
    inferred_highpi_frequencies_norm.loc[:,bool_] = inferred_highpi_frequencies_norm.loc[:,bool_].mul(coverage_norm[name], axis=0)

plt.hist(np.sum(inferred_highpi_frequencies_norm,1), bins=np.arange(0,2,.04), color='white', edgecolor='k')


# Get samples that:
# Have a high confidence pi above 0.5
# Have a high estimated frequency above 0.25

target_novel_bool = (pi_maps > 0.5) & (inferred_highpi_frequencies_norm.to_numpy() > 0.2) & (hpd_mask)

# of samples with at least 1 novel clade
print(f"Samples with at least 1 novel clade: {np.sum(np.sum(target_novel_bool,1) > 0)}/{len(sample_names)}")

# Output table for each clade of sample names, estimated frequency, and pi
with pd.ExcelWriter('supplemental/Supplemental_Table_S8.xlsx') as writer:
    
    for cladeidx, clade in enumerate(phlame_frequencies.columns):

        sample_names_clade = sample_names[target_novel_bool[:,cladeidx]]

        estimated_frequency_clade = inferred_highpi_frequencies_norm.to_numpy()[target_novel_bool[:,cladeidx],cladeidx]
        pi_clade = pi_maps[target_novel_bool[:,cladeidx],cladeidx]

        df = pd.DataFrame({'Sample':sample_names_clade.values,
                            'Relative abundance of the Gardnerella population':estimated_frequency_clade,
                            'Estimated pi':pi_clade,
                            'Study':md_pj.loc[sample_names_clade].values})

        df.to_excel(writer, sheet_name=f'Clade {clade}', index=False)

#%% Supplemental: What is the average number of Gardnerella clades that individuals have?
# =============================================================================

fig, axs = plt.subplots()
fig.set_size_inches(4,3)
nclades = np.count_nonzero(phlame_frequencies_norm[bool_include],1)
axs.hist(nclades, bins=np.arange(0,max(nclades)+2,1),
            color='white', edgecolor='k')
axs.set_xlabel('# of taxa', **fmt)
axs.set_ylabel('# of samples', **fmt)
axs.set_xticks(np.arange(0.5,max(nclades)+1.5,1))
axs.set_xticklabels([0,1,2,3,4,5,6,7,8])
# axs.set_title('All')
fig.tight_layout()

print(f'Average number of clades: {np.mean(nclades):.2f} +/- {np.std(nclades):.2f}')

# fig.savefig('supplemental/figs11b.pdf', dpi=300, format='pdf')


#%% Supplemental 2: 50 random stacked barplots from each region

import scipy.cluster.hierarchy as sch

slst2shape_dct = {'Gvag.GS1':'#1f77b4',
                  'Gvag.GS2':'#ff7f0e',
                  'Gpio.GS3':'#2ca02c',
                  'Gpio.GS4':'#e377c2',
                  'Gleo.GS5':'#9467bd',
                  'Gleo.GS6':'#17becf',
                  'Ggre.GS7':'#bcbd22',
                  'Ggre.GS8':'#d62728',
                  '-1':'#000000'}

colors_ = [slst2shape_dct[slst] for slst in phlame_frequencies_norm.columns]

import random

np.random.seed(123)

fig, axs = plt.subplots()
fig.set_size_inches(10,3)

random_sample_names = np.random.choice(list(sample_names.values[bool_include]), 50)

random_samples = phlame_frequencies_norm.loc[bool_include].loc[random_sample_names]

Z = sch.linkage(random_samples, method='ward')

from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, t=3, criterion='maxclust')
random_samples['Cluster'] = clusters

random_samples_sorted = random_samples.sort_values(by='Cluster')

# Sort by phylogroup A
# random_samples = random_samples.loc[random_samples['Gvag.GS1'].sort_values().index]

colors_ = [slst2shape_dct[slst] for slst in random_samples_sorted.iloc[:,:8].columns]

random_samples_sorted.iloc[:,:8].plot(kind='bar', stacked=True, ax=axs, edgecolor='k', linewidth=0.5, color=colors_) 
axs.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)

axs.set_xticks([])
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_ylabel('R. abundance', **fmt)
axs.set_xlabel('Subjects', **fmt)

axs.legend(labels=[slst.split('.')[1] for slst in random_samples_sorted.iloc[:,:8].columns],
              bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)
fig.tight_layout()

# fig.savefig('supplemental/figS11c.pdf', dpi=300, format='pdf')

#%% Benchmarking performance of Gardnerella classifier (simulated data)

# os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_02_Vaginal_gardnerella/benchmarking')

# samples_csv_file = 'benchmark_samples.csv'
# samples_csv = pd.read_csv(samples_csv_file)
# sample_names = samples_csv['Sample']

# counts_dir = '5-counts_benchmark'
# phlame_out_dir = '6-frequencies_benchmark'

# Gvag_refgenome = 'Gvaginalis_FDAARGOS_568'
# Gpio_refgenome = 'Gpiotii_ASM339758'
# Gleo_refgenome = 'Gleopoldii_6420B'
# Ggre_refgenome = 'Ggreenwoodii_ASM26363'

# phlame_frequencies = read_sample_frequencies_multi(sample_names[:-1],
#                                                    [phlame_out_dir,phlame_out_dir,phlame_out_dir,phlame_out_dir],
#                                                    [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
#                                                    ['Gvag','Gpio','Gleo','Ggre'])
# clade_names_all = phlame_frequencies.columns

# phlame_frequencies[phlame_frequencies < 0.01] = 0

# gvag_bool = [col.startswith('Gvag') for col in phlame_frequencies.columns]
# gpio_bool = [col.startswith('Gpio') for col in phlame_frequencies.columns]
# gleo_bool = [col.startswith('Gleo') for col in phlame_frequencies.columns]
# ggre_bool = [col.startswith('Ggre') for col in phlame_frequencies.columns]

# # # Normalize frequencies that sum above 1 to 1
# norm_to_1 = phlame_frequencies.iloc[:, 0:3].loc[(phlame_frequencies.iloc[:, 0:3].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 0:3].loc[(phlame_frequencies.iloc[:, 0:3].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
# norm_to_1 = phlame_frequencies.iloc[:, 2:4].loc[(phlame_frequencies.iloc[:, 2:4].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 2:4].loc[(phlame_frequencies.iloc[:, 2:4].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
# norm_to_1 = phlame_frequencies.iloc[:, 4:6].loc[(phlame_frequencies.iloc[:, 4:6].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 4:6].loc[(phlame_frequencies.iloc[:, 4:6].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
# norm_to_1 = phlame_frequencies.iloc[:, 6:8].loc[(phlame_frequencies.iloc[:, 6:8].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 6:8].loc[(phlame_frequencies.iloc[:, 6:8].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)


# coverage = parse_coverage_multi(sample_names[:-1],
#                                 counts_dir,
#                                 [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
#                                 ['Gvag','Gpio','Gleo','Ggre'])

# # Normalize frequencies across different reference genomes
# phlame_frequencies_norm = phlame_frequencies.copy()
# coverage_norm = coverage.div(coverage.sum(1),0)
# for name, bool_ in zip(['Gvag','Gpio','Gleo','Ggre'],
#                        [gvag_bool,gpio_bool,gleo_bool,ggre_bool]):
#     phlame_frequencies_norm.loc[:,bool_] = phlame_frequencies_norm.loc[:,bool_].mul(coverage_norm[name], axis=0)

# #%% Read in true frequencies

# true_abundances_file = 'true_community_abundances.csv'
# true_abundances = pd.read_csv(true_abundances_file, header=0, index_col=0).T

# isolate2lineage_file = 'Gardnerella_iso2_phylo.txt'
# iso2lineage_dct = {}
# with open(isolate2lineage_file,'r') as f:
#     for line in f:
#         lineinfo = line.rstrip('\n').split('\t')
#         iso2lineage_dct[lineinfo[0]] = lineinfo[1]


# true_abundances_phylo = helper.merge_frequencies_by_lineage(true_abundances,
#                                                              iso2lineage_dct)

# true_abundances_phylo.sort_index(inplace=True)

# true_abundances_phylo = true_abundances_phylo.iloc[:,:-1]
# # Normalize to 1
# true_abundances_phylo = true_abundances_phylo.div(true_abundances_phylo.sum(1),0)
# true_abundances_phylo['GS8'] = 0

# true_abundances_phylo = true_abundances_phylo.reindex(sorted(true_abundances_phylo.columns), axis=1)

# assert (true_abundances_phylo.index == phlame_frequencies_norm.index).all()

# #%% Plot true vs. estimated abundances

# fig, axs = plt.subplots()

# colors = ['red','lime','tab:orange','teal','dimgrey','purple','blue','black']
# for idx_, column in enumerate(true_abundances_phylo.columns):
#     if  column == 'GS8':
#         continue
#     axs.scatter(true_abundances_phylo[column], phlame_frequencies_norm.iloc[:,idx_],
#                 label=column, alpha=.5, color=colors[idx_])
#     axs.set_xlabel('True abundance')
#     axs.set_ylabel('Estimated abundance')

# axs.plot([0,1],[0,1], color='k', linestyle='--')
# axs.legend()


# #%% Laura samples

# os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/for_people/for_laura/leg_gardnerella')

# samples_csv_file = 'samples_vaginalis.csv'
# samples_csv = pd.read_csv(samples_csv_file)
# sample_names = samples_csv['Sample']

# counts_dir = '5-counts'
# phlame_out_dir = '6-frequencies'

# Gvag_refgenome = 'Gvaginalis_FDAARGOS_568'
# Gpio_refgenome = 'Gpiotii_ASM339758'
# Gleo_refgenome = 'Gleopoldii_6420B'
# Ggre_refgenome = 'Ggreenwoodii_ASM26363'

# phlame_frequencies = read_sample_frequencies_multi(sample_names,
#                                                    [phlame_out_dir,phlame_out_dir,phlame_out_dir,phlame_out_dir],
#                                                    [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
#                                                    ['Gvag','Gpio','Gleo','Ggre'])
# clade_names_all = phlame_frequencies.columns

# phlame_frequencies[phlame_frequencies < 0.01] = 0

# gvag_bool = [col.startswith('Gvag') for col in phlame_frequencies.columns]
# gpio_bool = [col.startswith('Gpio') for col in phlame_frequencies.columns]
# gleo_bool = [col.startswith('Gleo') for col in phlame_frequencies.columns]
# ggre_bool = [col.startswith('Ggre') for col in phlame_frequencies.columns]

# # # Normalize frequencies that sum above 1 to 1
# norm_to_1 = phlame_frequencies.iloc[:, 0:3].loc[(phlame_frequencies.iloc[:, 0:3].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 0:3].loc[(phlame_frequencies.iloc[:, 0:3].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
# norm_to_1 = phlame_frequencies.iloc[:, 2:4].loc[(phlame_frequencies.iloc[:, 2:4].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 2:4].loc[(phlame_frequencies.iloc[:, 2:4].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
# norm_to_1 = phlame_frequencies.iloc[:, 4:6].loc[(phlame_frequencies.iloc[:, 4:6].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 4:6].loc[(phlame_frequencies.iloc[:, 4:6].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
# norm_to_1 = phlame_frequencies.iloc[:, 6:8].loc[(phlame_frequencies.iloc[:, 6:8].sum(axis=1) > 1)]
# phlame_frequencies.iloc[:, 6:8].loc[(phlame_frequencies.iloc[:, 6:8].sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)


# coverage = parse_coverage_multi(sample_names,
#                                 counts_dir,
#                                 [Gvag_refgenome,Gpio_refgenome,Gleo_refgenome,Ggre_refgenome],
#                                 ['Gvag','Gpio','Gleo','Ggre'])

# # Normalize frequencies across different reference genomes
# phlame_frequencies_norm = phlame_frequencies.copy()
# coverage_norm = coverage.div(coverage.sum(1),0)
# for name, bool_ in zip(['Gvag','Gpio','Gleo','Ggre'],
#                        [gvag_bool,gpio_bool,gleo_bool,ggre_bool]):
#     phlame_frequencies_norm.loc[:,bool_] = phlame_frequencies_norm.loc[:,bool_].mul(coverage_norm[name], axis=0)

# phlame_frequencies_norm.fillna(0, inplace=True)
# #%% 

# fig, axs = plt.subplots()
# fig.set_size_inches(5,5)

# perc_called = phlame_frequencies_norm.sum(1)

# axs.scatter(coverage.sum(1), perc_called, color='k', alpha=.6)
# axs.set_xlabel('Sum non-overlapping Gardnerella coverage \n(across 4 reference genomes)', **fmt)
# axs.set_ylabel('Percent called', **fmt)
# axs.xaxis.set_tick_params(labelsize=12)
# axs.yaxis.set_tick_params(labelsize=12)

# axs.set_ylim(0,1)
# axs.set_xscale('log')

# axs.axvline(1, color='k', alpha=.5)

# fig.tight_layout()


# phlame_frequencies_norm.to_csv('phlame_frequencies_gardnerella.csv')

#%% Supplemental: Overdispersion of all the vaginal samples (from counts)