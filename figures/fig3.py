"""
author: evanqu
date: 2024/03/18 11:51
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import transforms
import seaborn as sns 
import scipy

import glob
import itertools

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures')

import fig_helper_functions as helper 

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

fmt={'fontsize':15,
     'fontname':'Helvetica'}

%matplotlib auto

#%% Functions

def check_equal(ls1, ls2, ls3, ls4, ls5):
    return all([set(ls1) == set(ls2),
                set(ls1) == set(ls3),
                set(ls1) == set(ls4),
                set(ls1) == set(ls5)])

def plot_precision_recall(recall_combined_lineage, precision_combined_lineage,
                          recall_combined_phylo, precision_combined_phylo,
                          colors_, species,
                          linestyle_ls = False, plot_dims = [9,6], alpha=.8):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(plot_dims[0], plot_dims[1])

    if linestyle_ls == False:
        linestyle_ls = ['solid']*len(colors_)
    # Recall lineage level
    sns.pointplot(data=recall_combined_lineage, x='index', y='value', hue='variable',
                     ax=ax[0,0], palette = colors_, linestyles=linestyle_ls,
                     dodge=True, markers='o', scale=0.8)
    ax[0,0].legend([])
    ax[0,0].set_xlabel('')
    ax[0,0].set_ylabel('Recall', **fmt)
    ax[0,0].set_title('Lineage level', **fmt)
    ax[0,0].set_xticks([0,1,2,3,4,5])
    ax[0,0].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
    ax[0,0].set_ylim(-0.01,1.01)
    ax[0,0].tick_params(axis='both', which='major', labelsize=15)
    plt.setp(ax[0,0].collections, alpha=alpha) #for the markers
    plt.setp(ax[0,0].lines, alpha=alpha)       #for the lines

    # Precision, lineage level
    sns.pointplot(data=precision_combined_lineage, x='index', y='value', hue='variable',
                    ax=ax[0,1], palette = colors_, linestyles=linestyle_ls,
                    dodge=True, markers='o', scale=0.8)
    ax[0,1].legend([])
    ax[0,1].set_xlabel('')
    ax[0,1].set_ylabel('Precision', **fmt)
    ax[0,1].set_title('Lineage level', **fmt)
    ax[0,1].set_xticks([0,1,2,3,4,5])
    ax[0,1].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
    ax[0,1].set_ylim(-0.01,1.01)
    ax[0,1].tick_params(axis='both', which='major', labelsize=15)
    plt.setp(ax[0,1].collections, alpha=alpha) #for the markers
    plt.setp(ax[0,1].lines, alpha=alpha)       #for the lines

    # Recall, phylogroup level
    sns.pointplot(data=recall_combined_phylo, x='index', y='value', hue='variable',
                    ax=ax[1,0], palette = colors_, linestyles=linestyle_ls,
                    dodge=True, markers='o', scale=0.8)
    ax[1,0].legend([])
    ax[1,0].set_xlabel('Coverage', **fmt)
    ax[1,0].set_ylabel('Recall', **fmt)
    ax[1,0].set_title('Phylogroup level', **fmt)
    ax[1,0].set_xticks([0,1,2,3,4,5])
    ax[1,0].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
    ax[1,0].set_ylim(-0.01,1.01)
    ax[1,0].tick_params(axis='both', which='major', labelsize=15)
    plt.setp(ax[1,0].collections, alpha=alpha) #for the markers
    plt.setp(ax[1,0].lines, alpha=alpha)       #for the lines

    # Precision, phylogroup level
    sns.pointplot(data=precision_combined_phylo, x='index', y='value', hue='variable',
                    ax=ax[1,1], palette = colors_, linestyles=linestyle_ls,
                    dodge=True, markers='o', scale=0.8)
    ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1,1].set_xlabel('Coverage', **fmt)
    ax[1,1].set_ylabel('Precision', **fmt)
    ax[1,1].set_title('Phylogroup level', **fmt)
    ax[1,1].set_xticks([0,1,2,3,4,5])
    ax[1,1].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
    ax[1,1].set_ylim(-0.01,1.01)
    ax[1,1].tick_params(axis='both', which='major', labelsize=15)
    plt.setp(ax[1,1].collections, alpha=alpha) #for the markers
    plt.setp(ax[1,1].lines, alpha=alpha)       #for the lines

    plt.suptitle(species, **fmt)

    fig.tight_layout()

    return fig, ax

def plot_f1_l2_score(f1_combined_lineage, f1_combined_phylo,
                     colors_, species, score, 
                     linestyle_ls = False, plot_dims = [9,3], alpha=.8):
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(plot_dims[0], plot_dims[1])

    if linestyle_ls == False:
        linestyle_ls = ['solid']*len(colors_)

    # F1, lineage level
    sns.pointplot(data=f1_combined_lineage, x='index', y='value', hue='variable',
                    ax=ax[0], palette = colors_, linestyles=linestyle_ls,
                    dodge=True, markers='o', scale=0.8)
    
    ax[0].legend([])
    ax[0].set_xlabel('Coverage', **fmt)
    ax[0].set_title('Lineage level', **fmt)
    ax[0].set_xticks([0,1,2,3,4,5])
    ax[0].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
    ax[0].set_ylim(-0.01,1.01)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    plt.setp(ax[0].collections, alpha=alpha) #for the markers
    plt.setp(ax[0].lines, alpha=alpha)       #for the lines

    # F1, phylogroup level
    sns.pointplot(data=f1_combined_phylo, x='index', y='value', hue='variable',
                    ax=ax[1], palette = colors_, linestyles=linestyle_ls,
                    dodge=True, markers='o', scale=0.8)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    ax[1].set_xlabel('Coverage', **fmt)
    ax[1].set_ylabel('', **fmt)
    ax[1].set_title('Phylogroup level', **fmt)
    ax[1].set_xticks([0,1,2,3,4,5])
    ax[1].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
    ax[1].set_ylim(-0.01,1.01)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    plt.setp(ax[1].collections, alpha=alpha) #for the markers
    plt.setp(ax[1].lines, alpha=alpha)       #for the lines
    
    if score == 'F1':
        ax[0].set_ylabel('F1 score', **fmt)
    elif score == 'L2':
        ax[0].set_ylabel('L2 distance', **fmt)
        ax[0].set_ylim(-0.01,1.2)
        ax[1].set_ylim(-0.01,1.2)

    plt.suptitle(species, **fmt)

    fig.tight_layout()

    return fig, ax

samplename2cov_dct = {'01X':0.1,
                      '05X':0.5,
                      '1X':1,
                      '5X':5,
                      '10X':10,
                      '20X':20 }
#%% Fig 2B1: Precision and Recall, C. acnes lineage and phylogroup level

TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Cacnes_community_abundances.csv'

path_to_samples_csv = 'fig3/Cacnes_samples.csv'
path_to_phlame_dir = 'fig3/PHLAME/Cacnes'
path_to_strainest_dir = 'fig3/StrainEst/Cacnes_classify_all'
path_to_strainge_dir = 'fig3/StrainGE/Cacnes_classify_rep'
path_to_strainscan_dir = 'fig3/StrainScan/Cacnes'

isolate2lineage_file = 'fig3/Cacnes_lineage2assembly.txt'
lineage2phylogroup_file = 'fig3/Cacnes_lineage2phylogroups.txt'

reference_genome = 'Pacnes_C1'

detection_lim =  0.01

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))

sample_names = pd.read_csv(path_to_samples_csv)['Sample']

#%% Fig 2B1: Read in outputs from each method for C. acnes

# Read in true frequencies
true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

strain_names = true_abundances.columns
true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                             iso2lineage_dct)
# Normalize true abundances to 1
true_abundances_lineage = true_abundances_lineage.div(true_abundances_lineage.sum(axis=1), axis=0)
true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                     lineage2phylogroup_file)


#PHLAME 
phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                  path_to_phlame_dir,
                                                  reference_genome)
#StrainEst
strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           sample_names,
                                           strain_names,
                                           reference_genome)
#StrainGE
strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         sample_names,
                                         strain_names)
#StrainScan
strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             sample_names,
                                             strain_names)

strainest_freqs = helper.merge_frequencies_by_lineage(strainest_out, iso2lineage_dct)
strainge_freqs = helper.merge_frequencies_by_lineage(strainge_out, iso2lineage_dct)
strainscan_freqs = helper.merge_frequencies_by_lineage(strainscan_out, iso2lineage_dct)

# Convert to phylogroup
strainest_freqs_phylo = helper.convert_to_phylogroup(strainest_freqs,
                                                     lineage2phylogroup_file)
strainge_freqs_phylo = helper.convert_to_phylogroup(strainge_freqs,
                                                    lineage2phylogroup_file)
strainscan_freqs_phylo = helper.convert_to_phylogroup(strainscan_freqs,
                                                    lineage2phylogroup_file)

# Remove low-frequency abundances
phlame_freqs[phlame_freqs < detection_lim] = 0
strainest_freqs[strainest_freqs < detection_lim] = 0
strainge_freqs[strainge_freqs < detection_lim] = 0
strainest_freqs_phylo[strainest_freqs_phylo < detection_lim] = 0
strainge_freqs_phylo[strainge_freqs_phylo < detection_lim] = 0
strainscan_freqs_phylo[strainscan_freqs_phylo < detection_lim] = 0

phlame_freqs_lineage = phlame_freqs[lineage_IDs]
phlame_freqs_phylo = phlame_freqs[phylogroup_IDs]
                
assert check_equal(true_abundances_lineage.columns,
                   phlame_freqs_lineage.columns,
                   strainest_freqs.columns,
                   strainge_freqs.columns,
                   strainscan_freqs.columns)

assert check_equal(true_abundances_phylo.columns,
                    phlame_freqs_phylo.columns,
                    strainest_freqs_phylo.columns,
                    strainge_freqs_phylo.columns,
                    strainscan_freqs_phylo.columns)

assert check_equal(true_abundances.index,
                    phlame_freqs.index,
                    strainest_freqs.index,
                    strainge_freqs.index,
                    strainscan_freqs.index)

#%% Fig 2B1: Calc precision and recall stats, C. acnes

cacnes_recall_lineage, cacnes_precision_lineage, \
cacnes_f1_lineage, cacnes_l2_lineage = helper.calc_benchmarking_stats([strainscan_freqs,strainest_freqs,strainge_freqs,phlame_freqs_lineage],
                                                        ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                        true_abundances_lineage,
                                                        sample_names, '_C')

cacnes_recall_phylo, cacnes_precision_phylo, \
cacnes_f1_phylo, cacnes_l2_phylo = helper.calc_benchmarking_stats([strainscan_freqs_phylo,strainest_freqs_phylo,strainge_freqs_phylo,phlame_freqs_phylo],
                                                    ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                    true_abundances_phylo,
                                                    sample_names, '_C')

# Get the average F1 score at 20X coverage for each method
cacnes_f1_phylo_covX = cacnes_f1_phylo[cacnes_f1_phylo['index']=='20.0']
cacnes_f1_lineage_covX = cacnes_f1_lineage[cacnes_f1_lineage['index']=='20.0']

# cacnes_f1_phylo_covX = cacnes_f1_phylo_covX.groupby('variable').mean().reset_index()
# cacnes_f1_lineage_covX = cacnes_f1_lineage_covX.groupby('variable').mean().reset_index()

cacnes_precision_lineage.groupby('variable').mean().reset_index()
cacnes_precision_phylo.groupby('variable').mean().reset_index()

# Get the AUROC precision/coverage curve


#%% Understand why PHLAME has lower precision at 20X

# prefix_to_split = '_C'
# coverage_toint_dict = {'01X':0.1,
#                         '05X':0.5,
#                         '1X':1,
#                         '5X':5,
#                         '10X':10,
#                         '20X':20}

# coverages = [coverage_toint_dict[sam.split(prefix_to_split, 1)[0]] for sam in sample_names]

# recall_ls = []; precision_ls = []; f1_ls = []; l2_ls = []

# recall_phylo = calc_recall(phlame_freqs_phylo, true_abundances_phylo)
# precision_phylo, FDR_phylo = calc_precision(phlame_freqs_phylo, true_abundances_phylo)
# l2_phylo = calc_l2_distance(phlame_freqs_phylo, true_abundances_phylo)

# recall_lineage = calc_recall(phlame_freqs_lineage, true_abundances_lineage)
# precision_lineage, FDR_lineage = calc_precision(phlame_freqs_lineage, true_abundances_lineage)
# l2_lineage = calc_l2_distance(phlame_freqs_lineage, true_abundances_lineage)



#%% Fig 2B2: Plot precision and recall, C. acnes

fig3b, axs3b = plot_precision_recall(cacnes_recall_lineage, cacnes_precision_lineage,
                                        cacnes_recall_phylo, cacnes_precision_phylo,
                                        ['b','r','k','g'],'C. acnes')

fig3b.show()

# fig3b.savefig('fig3/fig3a1.pdf',format='pdf')

fig3b2, axs3b2 = plot_f1_l2_score(cacnes_f1_lineage, cacnes_f1_phylo,
                                ['b','r','k','g'],'C. acnes','F1')

fig3b2.show()

# fig3b2.savefig('fig3/fig3a2.pdf',format='pdf')

fig3b3, axs3b3 = plot_f1_l2_score(cacnes_l2_lineage, cacnes_l2_phylo,
                                    ['b','r','k','g'],'C. acnes','L2')

fig3b3.show()

# fig3b3.savefig('supplemental/figS7A.pdf',format='pdf')

#%% Fig 2B2: Precision and Recall, S. epidermidis lineage and phylogroup level

TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Sepi_community_abundances.csv'

path_to_samples_csv = 'fig3/Sepi_samples.csv'
path_to_phlame_dir = 'fig3/PHLAME/Sepi'
path_to_strainest_dir = 'fig3/StrainEst/Sepi_classify_all'
path_to_strainge_dir = 'fig3/StrainGE/Sepi_classify_rep'
path_to_strainscan_dir = 'fig3/StrainScan/Sepi_rep'

isolate2lineage_file = 'fig3/Sepi_lineage2assembly.txt'
lineage2phylogroup_file = 'fig3/Sepi_lineage2phylogroups.txt'

reference_genome = 'SepidermidisATCC12228'

detection_lim =  0.01

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))

#%% Fig 2B2: Read in outputs from each method for S. epi

sample_names = pd.read_csv(path_to_samples_csv)['Sample']

# Read in true frequencies
true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

strain_names = true_abundances.columns

true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                             iso2lineage_dct)
# Normalize true abundances to 1
true_abundances_lineage = true_abundances_lineage.div(true_abundances_lineage.sum(axis=1), axis=0)

true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                     lineage2phylogroup_file)

#PHLAME 
phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                  path_to_phlame_dir,
                                                  reference_genome)
#StrainEst
strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           sample_names,
                                           strain_names,
                                           reference_genome)
#StrainGE
strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         sample_names,
                                         strain_names)

#StrainScan
strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             sample_names,
                                             strain_names)

strainest_freqs = helper.merge_frequencies_by_lineage(strainest_out, iso2lineage_dct)
strainge_freqs = helper.merge_frequencies_by_lineage(strainge_out, iso2lineage_dct)
strainscan_freqs = helper.merge_frequencies_by_lineage(strainscan_out, iso2lineage_dct)

# Convert to phylogroup
strainest_freqs_phylo = helper.convert_to_phylogroup(strainest_freqs,
                                                     lineage2phylogroup_file)
strainge_freqs_phylo = helper.convert_to_phylogroup(strainge_freqs,
                                                    lineage2phylogroup_file)
strainscan_freqs_phylo = helper.convert_to_phylogroup(strainscan_freqs,
                                                    lineage2phylogroup_file)

# Remove low-frequency abundances
phlame_freqs[phlame_freqs < detection_lim] = 0
strainest_freqs[strainest_freqs < detection_lim] = 0
strainge_freqs[strainge_freqs < detection_lim] = 0
strainest_freqs_phylo[strainest_freqs_phylo < detection_lim] = 0
strainge_freqs_phylo[strainge_freqs_phylo < detection_lim] = 0
strainscan_freqs_phylo[strainscan_freqs_phylo < detection_lim] = 0

phlame_freqs_lineage = phlame_freqs[lineage_IDs]
phlame_freqs_phylo = phlame_freqs[['A','B','C','D']]
phlame_freqs_phylo.columns = phylogroup_IDs

assert check_equal(true_abundances_lineage.columns,
                   phlame_freqs_lineage.columns,
                   strainest_freqs.columns,
                   strainge_freqs.columns,
                   strainscan_freqs.columns)

assert check_equal(true_abundances_phylo.columns,
                    phlame_freqs_phylo.columns,
                    strainest_freqs_phylo.columns,
                    strainge_freqs_phylo.columns,
                    strainscan_freqs_phylo.columns)

assert check_equal(true_abundances.index,
                    phlame_freqs.index,
                    strainest_freqs.index,
                    strainge_freqs.index,
                    strainscan_freqs.index)

#%% Fig 2B2: Calc precision and recall stats, S. epi

sepi_recall_lineage, sepi_precision_lineage, \
sepi_f1_lineage, sepi_l2_lineage = helper.calc_benchmarking_stats([strainscan_freqs,strainest_freqs,strainge_freqs,phlame_freqs_lineage],
                                                        ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                        true_abundances_lineage,
                                                        sample_names, '_S')

sepi_recall_phylo, sepi_precision_phylo, \
sepi_f1_phylo, sepi_l2_phylo = helper.calc_benchmarking_stats([strainscan_freqs_phylo,strainest_freqs_phylo,strainge_freqs_phylo,phlame_freqs_phylo],
                                                    ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                    true_abundances_phylo,
                                                    sample_names, '_S')

# Get the average F1 score at 20X coverage for each method
# f1_phylo_covX = f1_phylo[f1_phylo['index']=='10.0']
# f1_lineage_covX = f1_lineage[f1_lineage['index']=='10.0']

# f1_phylo_covX = f1_phylo_covX.groupby('variable').mean().reset_index()
# f1_lineage_covX = f1_lineage_covX.groupby('variable').mean().reset_index()

# precision_lineage.groupby('variable').mean().reset_index()
# precision_phylo.groupby('variable').mean().reset_index()

#%% Fig 2B2: Plot precision and recall , S. epi

fig3b, axs3b = plot_precision_recall(sepi_recall_lineage, sepi_precision_lineage,
                                        sepi_recall_phylo, sepi_precision_phylo,
                                        ['b','r','k','g'],'S. epidermidis')

fig3b.show()

# fig3b.savefig('fig3/fig3b1.pdf',format='pdf')

fig3b2, axs3b2 = plot_f1_l2_score(sepi_f1_lineage, sepi_f1_phylo,
                                ['b','r','k','g'],'S. epidermidis','F1')

fig3b2.show()

# fig3b2.savefig('fig3/fig3b2.pdf',format='pdf')

fig3b3, axs3b3 = plot_f1_l2_score(sepi_l2_lineage, sepi_l2_phylo,
                                    ['b','r','k','g'],'S. epidermidis','L2')

fig3b3.show()

# fig3b3.savefig('supplemental/figS7B.pdf',format='pdf')
#%% C. acnes, drop 25% of clades

TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Cacnes_community_abundances_dropped.csv'

path_to_samples_csv = 'fig3/Cacnes_samples.csv'
path_to_phlame_dir = 'fig3/PHLAME/Cacnes_dropped'
path_to_strainest_dir = 'fig3/StrainEst/Cacnes_dropped_all'
path_to_strainge_dir = 'fig3/StrainGE/Cacnes_dropped_rep'
path_to_strainscan_dir = 'fig3/StrainScan/Cacnes_dropped_rep'

isolate2lineage_file = 'fig3/Cacnes_lineage2assembly.txt'
iso2lineage_dropped_file = 'fig3/Cacnes_lineage2assembly_dropped_phylo.txt'
lineage2phylogroup_file = 'fig3/Cacnes_lineage2phylogroups.txt'

reference_genome = 'Pacnes_C1'

detection_lim =  0.01

iso2lineage_dropped_dct = {}
with open(iso2lineage_dropped_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dropped_dct[lineinfo[0]] = lineinfo[1]

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
lineage_IDs_dropped = np.unique(list(iso2lineage_dropped_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))
phylogroup_IDs_dropped = ['A','C','E','F','H','K'] # C. acnes

#%% C. acnes, drop 25% of clades

sample_names = pd.read_csv(path_to_samples_csv)['Sample']

# Read in true frequencies
true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

strain_names = list(iso2lineage_dropped_dct.keys())
true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                             iso2lineage_dct)
# Normalize true abundances to 1
true_abundances_lineage = true_abundances_lineage.div(true_abundances_lineage.sum(axis=1), axis=0)

true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                     lineage2phylogroup_file)

true_abundances_lineage = true_abundances_lineage[lineage_IDs_dropped]
true_abundances_phylo = true_abundances_phylo[phylogroup_IDs_dropped]

#PHLAME 
phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                  path_to_phlame_dir,
                                                  reference_genome)
#StrainEst
strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           sample_names,
                                           strain_names,
                                           reference_genome)
#StrainGE
strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         sample_names,
                                         strain_names)

#StrainScan
strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             sample_names,
                                             strain_names)

strainest_freqs = helper.merge_frequencies_by_lineage(strainest_out, iso2lineage_dropped_dct)
strainge_freqs = helper.merge_frequencies_by_lineage(strainge_out, iso2lineage_dropped_dct)
strainscan_freqs = helper.merge_frequencies_by_lineage(strainscan_out, iso2lineage_dropped_dct)

# Convert to phylogroup
strainest_freqs_phylo = helper.convert_to_phylogroup(strainest_freqs,
                                                     lineage2phylogroup_file)
strainge_freqs_phylo = helper.convert_to_phylogroup(strainge_freqs,
                                                    lineage2phylogroup_file)
strainscan_freqs_phylo = helper.convert_to_phylogroup(strainscan_freqs,
                                                    lineage2phylogroup_file)

# Remove low-frequency abundances
phlame_freqs[phlame_freqs < detection_lim] = 0
strainest_freqs[strainest_freqs < detection_lim] = 0
strainge_freqs[strainge_freqs < detection_lim] = 0
strainest_freqs_phylo[strainest_freqs_phylo < detection_lim] = 0
strainge_freqs_phylo[strainge_freqs_phylo < detection_lim] = 0
strainscan_freqs_phylo[strainscan_freqs_phylo < detection_lim] = 0

phlame_freqs_lineage = phlame_freqs[lineage_IDs_dropped]
phlame_freqs_phylo = phlame_freqs[phylogroup_IDs_dropped] # C. acnes

phlame_freqs_phylo.columns = phylogroup_IDs_dropped
                
assert check_equal(true_abundances_lineage.columns,
                   phlame_freqs_lineage.columns,
                   strainest_freqs.columns,
                   strainge_freqs.columns,
                   strainscan_freqs.columns)

assert check_equal(true_abundances_phylo.columns,
                    phlame_freqs_phylo.columns,
                    strainest_freqs_phylo.columns,
                    strainge_freqs_phylo.columns,
                    strainscan_freqs_phylo.columns)

assert check_equal(true_abundances.index,
                    phlame_freqs.index,
                    strainest_freqs.index,
                    strainge_freqs.index,
                    strainscan_freqs.index)

#%% C. acnes, drop 25% of clades

cacnes_dropped_recall_lineage, cacnes_dropped_precision_lineage, \
cacnes_dropped_f1_lineage, cacnes_dropped_l2_lineage = helper.calc_benchmarking_stats([strainscan_freqs,strainest_freqs,strainge_freqs,phlame_freqs_lineage],
                                                        ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                        true_abundances_lineage,
                                                        sample_names, '_C')

cacnes_dropped_recall_phylo, cacnes_dropped_precision_phylo, \
cacnes_dropped_f1_phylo, cacnes_dropped_l2_phylo = helper.calc_benchmarking_stats([strainscan_freqs_phylo,strainest_freqs_phylo,strainge_freqs_phylo,phlame_freqs_phylo],
                                                    ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                    true_abundances_phylo,
                                                    sample_names, '_C')

cacnes_dropped_precision_lineage.groupby('variable').mean().reset_index()
cacnes_dropped_precision_phylo.groupby('variable').mean().reset_index()

#%% Fig 2B2: Plot precision and recall , C. acnes

fig3b, axs3b = plot_precision_recall(cacnes_dropped_recall_lineage, cacnes_dropped_precision_lineage,
                                        cacnes_dropped_recall_phylo, cacnes_dropped_precision_phylo,
                                        ['b','r','k','g'],'C. acnes (25% dropped)')

fig3b.show()

# fig3b.savefig('fig3/fig3c1.pdf',format='pdf')

fig3b2, axs3b2 = plot_f1_l2_score(cacnes_dropped_f1_lineage, cacnes_dropped_f1_phylo,
                                ['b','r','k','g'],'C. acnes (25% dropped)','F1')

fig3b2.show()

# fig3b2.savefig('fig3/fig3c2.pdf',format='pdf')

fig3b3, axs3b3 = plot_f1_l2_score(cacnes_dropped_l2_lineage, cacnes_dropped_l2_phylo,
                                    ['b','r','k','g'],'C. acnes (25% dropped)','L2')

fig3b3.show()

# fig3b3.savefig('supplemental/figS7C.pdf',format='pdf')

#%% S. epidermidis, drop 25% of clades

TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Sepi_community_abundances_dropped.csv'

path_to_samples_csv = 'fig3/Sepi_samples.csv'
path_to_phlame_dir = 'fig3/PHLAME/Sepi_dropped'
path_to_strainest_dir = 'fig3/StrainEst/Sepi_dropped_all'
path_to_strainge_dir = 'fig3/StrainGE/Sepi_dropped_rep'
path_to_strainscan_dir = 'fig3/StrainScan/Sepi_dropped_rep'

isolate2lineage_file = 'fig3/Sepi_lineage2assembly.txt'
iso2lineage_dropped_file = 'fig3/Sepi_lineage2assembly_dropped_phylo.txt'
lineage2phylogroup_file = 'fig3/Sepi_lineage2phylogroups.txt'

reference_genome = 'SepidermidisATCC12228'

detection_lim =  0.01

iso2lineage_dropped_dct = {}
with open(iso2lineage_dropped_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dropped_dct[lineinfo[0]] = lineinfo[1]

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
lineage_IDs_dropped = np.unique(list(iso2lineage_dropped_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))

phylogroup_IDs_dropped = ['1','2','4'] # S epidermidis

#%% S. epidermidis, drop 25% of clades

sample_names = pd.read_csv(path_to_samples_csv)['Sample']

# Read in true frequencies
true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

strain_names = list(iso2lineage_dropped_dct.keys())
true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                             iso2lineage_dct)
# Normalize true abundances to 1
true_abundances_lineage = true_abundances_lineage.div(true_abundances_lineage.sum(axis=1), axis=0)

true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                     lineage2phylogroup_file)

true_abundances_lineage = true_abundances_lineage[lineage_IDs_dropped]
true_abundances_phylo = true_abundances_phylo[phylogroup_IDs_dropped]

#PHLAME 
phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                  path_to_phlame_dir,
                                                  reference_genome)
#StrainEst
strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           sample_names,
                                           strain_names,
                                           reference_genome)
#StrainGE
strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         sample_names,
                                         strain_names)

#StrainScan
strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             sample_names,
                                             strain_names)

strainest_freqs = helper.merge_frequencies_by_lineage(strainest_out, iso2lineage_dropped_dct)
strainge_freqs = helper.merge_frequencies_by_lineage(strainge_out, iso2lineage_dropped_dct)
strainscan_freqs = helper.merge_frequencies_by_lineage(strainscan_out, iso2lineage_dropped_dct)

# Convert to phylogroup
strainest_freqs_phylo = helper.convert_to_phylogroup(strainest_freqs,
                                                     lineage2phylogroup_file)
strainge_freqs_phylo = helper.convert_to_phylogroup(strainge_freqs,
                                                    lineage2phylogroup_file)
strainscan_freqs_phylo = helper.convert_to_phylogroup(strainscan_freqs,
                                                    lineage2phylogroup_file)

# Remove low-frequency abundances
phlame_freqs[phlame_freqs < detection_lim] = 0
strainest_freqs[strainest_freqs < detection_lim] = 0
strainge_freqs[strainge_freqs < detection_lim] = 0
strainest_freqs_phylo[strainest_freqs_phylo < detection_lim] = 0
strainge_freqs_phylo[strainge_freqs_phylo < detection_lim] = 0
strainscan_freqs_phylo[strainscan_freqs_phylo < detection_lim] = 0

phlame_freqs_lineage = phlame_freqs[lineage_IDs_dropped]
phlame_freqs_phylo = phlame_freqs[['A','B','D']] # S. epidermidis

phlame_freqs_phylo.columns = phylogroup_IDs_dropped
                
assert check_equal(true_abundances_lineage.columns,
                   phlame_freqs_lineage.columns,
                   strainest_freqs.columns,
                   strainge_freqs.columns,
                   strainscan_freqs.columns)

assert check_equal(true_abundances_phylo.columns,
                    phlame_freqs_phylo.columns,
                    strainest_freqs_phylo.columns,
                    strainge_freqs_phylo.columns,
                    strainscan_freqs_phylo.columns)

assert check_equal(true_abundances.index,
                    phlame_freqs.index,
                    strainest_freqs.index,
                    strainge_freqs.index,
                    strainscan_freqs.index)

#%% S. epidermidis, drop 25% of clades


sepi_dropped_recall_lineage, sepi_dropped_precision_lineage, \
sepi_dropped_f1_lineage, sepi_dropped_l2_lineage = helper.calc_benchmarking_stats([strainscan_freqs,strainest_freqs,strainge_freqs,phlame_freqs_lineage],
                                                        ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                        true_abundances_lineage,
                                                        sample_names, '_S')

sepi_dropped_recall_phylo, sepi_dropped_precision_phylo, \
sepi_dropped_f1_phylo, sepi_dropped_l2_phylo = helper.calc_benchmarking_stats([strainscan_freqs_phylo,strainest_freqs_phylo,strainge_freqs_phylo,phlame_freqs_phylo],
                                                    ['StrainScan','StrainEst','StrainGE','PHLAME'],
                                                    true_abundances_phylo,
                                                    sample_names, '_S')

sepi_dropped_precision_lineage.groupby('variable').mean().reset_index()
sepi_dropped_precision_phylo.groupby('variable').mean().reset_index()

#%% Fig 2B2: Plot precision and recall , S. epi dropped

fig3b, axs3b = plot_precision_recall(sepi_dropped_recall_lineage, sepi_dropped_precision_lineage,
                                        sepi_dropped_recall_phylo, sepi_dropped_precision_phylo,
                                        ['b','r','k','g'],'S. epidermidis (25% dropped)')

fig3b.show()

# fig3b.savefig('fig3/fig3d1.pdf',format='pdf')

fig3b2, axs3b2 = plot_f1_l2_score(sepi_dropped_f1_lineage, sepi_dropped_f1_phylo,
                                ['b','r','k','g'],'S. epidermidis (25% dropped)','F1')

fig3b2.show()

# fig3b2.savefig('fig3/fig3d2.pdf',format='pdf')

fig3b3, axs3b3 = plot_f1_l2_score(sepi_dropped_l2_lineage, sepi_dropped_l2_phylo,
                                    ['b','r','k','g'],'S. epidermidis (25% dropped)','L2')

fig3b3.show()

# fig3b3.savefig('supplemental/figS7D.pdf',format='pdf')

#%% Supplement 1: PHLAME results are consistent across reasonable threshold parameters (AUROC curves)

def reclassify_from_data(data_file,
                         freq_file,
                         max_pi=0.3,
                         min_snps=10,
                         min_prob=0.75):

    if min_snps!=10:
        raise Exception('Changes in the min_snps threshold from default (10) will have to be re-MCMCed')      
    
    min_hpd = 10

    # Load in sample
    sample_frequencies = helper.Frequencies(freq_file)
    data = helper.FrequenciesData(data_file)
    
    # What clades have enough SNPs to be modeled in the original run
    model_bool = np.logical_or.reduce((data.counts_MLE != -1),1)

    # Make new_frequencies data structure to fill
    new_frequencies = np.full_like(sample_frequencies.freqs['Relative abundance'],0)

    # new_probs = np.full_like(data.prob,-1)
    
    # Recalc frequencies with new thresholds
    for i in range(len(new_frequencies)):
        
        # Ignore the clades not to model
        if not model_bool[i]:
            continue
        
        # Calc new prob; move on if below threshold
        pi_chain = data.chain[i]['pi']
        cts2model = data.clade_counts[i][0] + data.clade_counts[i][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_nonzero_zero = np.true_divide(np.count_nonzero(cts2model>0),
                                        np.count_nonzero(cts2model==0))
            if ratio_nonzero_zero == np.inf:
                ratio_nonzero_zero = 1.0

        hpd = helper.get_hpd(pi_chain)

        if (np.sum(pi_chain < max_pi)/len(pi_chain) < min_prob) or (hpd[0] > min_hpd): #or (ratio_nonzero_zero < 0.03):
            continue
        
        # Only consider the parts of the chain below max_pi
        chain_bool = pi_chain < max_pi
        
        a_mean = np.mean(data.chain[i]['a'][chain_bool])
        b_mean = np.mean(data.chain[i]['b'][chain_bool])
                
        new_frequencies[i] = (a_mean/b_mean)/data.total_MLE[i][0]
            
    new_frequencies_df = pd.DataFrame(new_frequencies, 
                                      index=sample_frequencies.freqs.index,
                                      dtype=float)
    
    return new_frequencies_df


def reclassify_phlame_out_dir(sample_names,
                              path_to_phlame_dir,
                              reference_genome,
                              max_pi,
                              min_snps,
                              min_prob):
    '''
    Reclassify frequency files from an output directory with new parameters.
    '''
    
    jt_frequencies = pd.DataFrame()
    
    for sample in sample_names:
                
        freq_file = f'{path_to_phlame_dir}/{sample}_ref_{reference_genome}_frequencies.csv'
        data_file = f'{path_to_phlame_dir}/{sample}_ref_{reference_genome}_fitinfo.data'
        
        sample_freqs = reclassify_from_data(data_file,
                                            freq_file,
                                            max_pi=max_pi,
                                            min_snps=10,
                                            min_prob=min_prob)
        
        sample_freqs.columns = [sample]
        
        jt_frequencies = pd.concat((jt_frequencies,sample_freqs), axis=1)
        
    return jt_frequencies.T

#%% Supplement 1: PHLAME results are consistent across reasonable threshold parameters (AUROC curves)

# Calculate AUPR with some varied parameter
path_to_phlame_dir = 'fig3/PHLAME/Sepi_dropped'
TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Sepi_community_abundances_dropped.csv'
path_to_samples_csv = 'fig3/Sepi_samples.csv'
isolate2lineage_file = 'fig3/Sepi_lineage2assembly_dropped_phylo.txt'
lineage2phylogroup_file = 'fig3/Sepi_lineage2phylogroups.txt'

# reference_genome = 'Pacnes_C1'
reference_genome = 'SepidermidisATCC12228'

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))

sample_names = pd.read_csv(path_to_samples_csv)['Sample']

true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T
true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                             iso2lineage_dct)
true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                        lineage2phylogroup_file)

sample_names = pd.read_csv(path_to_samples_csv)['Sample']
sample_names_covX = np.array([sam for sam in sample_names if sam.startswith('5X')])
true_abundances_lineage_covX = true_abundances_lineage.loc[sample_names_covX]
true_abundances_phylo_covX = true_abundances_phylo.loc[sample_names_covX]

# Iterate over a range of parameter values 
min_prob=0.5 # \ max_pi=0.3
max_pi_iter=np.arange(0.05,1.01,0.05)

recall_ls_lineage = []; precision_ls_lineage = []
recall_ls_phylo = []; precision_ls_phylo = []

for max_pi in max_pi_iter:
    print(f"Currently on: {max_pi}")
    
    new_frequencies = reclassify_phlame_out_dir(sample_names_covX,
                                                path_to_phlame_dir,
                                                reference_genome,
                                                max_pi=max_pi,
                                                min_snps=10,
                                                min_prob=min_prob)
    # reorder samples
    new_frequencies = new_frequencies.reindex(sample_names_covX)
    new_frequencies[new_frequencies < 0.0075] = 0
    new_frequencies_lineage = new_frequencies[lineage_IDs]
    # new_frequencies_phylo = new_frequencies[phylogroup_IDs]
    # new_frequencies_phylo = new_frequencies[['A','C','E','F','H','K']]
    new_frequencies_phylo = new_frequencies[['A','B','D']]
    new_frequencies_phylo.columns = ['1','2','4']
    # new_frequencies_phylo.columns = phylogroup_IDs
    
    # new_frequencies.to_csv(f'fig3/param_search/Cacnes_frequencies_maxpi={max_pi:.2f}_minprob={min_prob:.2f}.csv')
    
    assert (new_frequencies_lineage.index == true_abundances_lineage.loc[sample_names_covX].index).all()

    recall_lineage, precision_lineage, \
    f1_lineage, l2_lineage = helper.calc_benchmarking_stats([new_frequencies_lineage],
                                                                    ['PHLAME'],
                                                                    true_abundances_lineage_covX,
                                                                    sample_names_covX, '_S')

    recall_phylo, precision_phylo, \
    f1_phylo, l2_phylo = helper.calc_benchmarking_stats([new_frequencies_phylo],
                                                                    ['PHLAME'],
                                                                    true_abundances_phylo_covX,
                                                                    sample_names_covX, '_S')


    recall_ls_lineage.append(np.mean(recall_lineage))
    precision_ls_lineage.append(np.mean(precision_lineage))

    recall_ls_phylo.append(np.mean(recall_phylo))
    precision_ls_phylo.append(np.mean(precision_phylo))

# Replace NA precisions with 1 just for plotting
precision_ls_lineage = np.nan_to_num(precision_ls_lineage, nan=1)
precision_ls_phylo = np.nan_to_num(precision_ls_phylo, nan=1)

#%% Supplement 1 parameter search


fig, axs = plt.subplots(1,2)
fig.set_size_inches(8,3.8)

axs[0].plot(recall_ls_lineage, precision_ls_lineage, color='k', linewidth=3, linestyle='--')
axs[0].plot(recall_ls_lineage[1:10], precision_ls_lineage[1:10], color='k', linewidth=3)
# axs[0].scatter(recall_ls_lineage, precision_ls_lineage, color='k', alpha=max_pi_iter)
axs[0].set_xlabel('Recall', **fmt); axs[0].set_ylabel('Precision', **fmt)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].set_xticks([0,.5,1]), axs[0].set_yticks([0,.5,1])
axs[0].set_ylim(0,1.02), axs[0].set_xlim(0,1.02)
axs[0].set_title('Lineage level', **fmt)

axs[1].plot(recall_ls_phylo, precision_ls_phylo, color='k', linewidth=3, linestyle='--')
axs[1].plot(recall_ls_phylo[1:15], precision_ls_phylo[1:15], color='k', linewidth=3)
# axs[1].scatter(recall_ls_phylo, precision_ls_phylo, color='k', alpha=max_pi_iter)
axs[1].set_xlabel('Recall', **fmt)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].set_xticks([0,.5,1]), axs[1].set_yticks([0,.5,1])
axs[1].set_ylim(0,1.02), axs[1].set_xlim(0,1.02)
axs[1].set_title('Phylogroup level', **fmt)

# axs[1].plot(100,100, 'o', label='PHLAME', color='k')

# Plot precision and recall at 10X for other methods
for col_, method in zip(['k','b','g','r'],['PHLAME','StrainEst','StrainGE','StrainScan']):
    recall_ = sepi_dropped_recall_lineage.query(f'index=="1.2" & variable=="{method}"')['value'].mean()
    prec_ = sepi_dropped_precision_lineage.query(f'index=="1.2" & variable=="{method}"')['value'].mean()

    axs[0].plot(recall_, prec_, 'o', label=method, color=col_)

    recall_ = sepi_dropped_recall_phylo.query(f'index=="1.2" & variable=="{method}"')['value'].mean()
    prec_ = sepi_dropped_precision_phylo.query(f'index=="1.2" & variable=="{method}"')['value'].mean()
    axs[1].plot(recall_, prec_, 'o', label=method, color=col_)

axs[1].legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=12)

fig.suptitle('$\it{S. epidermidis}$ 25% dropped (5X coverage)', **fmt)

fig.tight_layout()

fig.savefig('supplemental/figS9D.pdf',format='pdf')


#%% Supplement: StrainEst, StrainGE results between complete and dereplicated dbs


TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Cacnes_community_abundances_dropped.csv'

path_to_samples_csv = 'fig3/Cacnes_samples.csv'
isolate2lineage_file = 'fig3/Cacnes_lineage2assembly_dropped_phylo.txt'
lineage2phylogroup_file = 'fig3/Cacnes_lineage2phylogroups.txt'

reference_genome = 'Pacnes_C1'
# reference_genome = 'SepidermidisATCC12228'

sample_names = pd.read_csv(path_to_samples_csv)['Sample']
strain_names = pd.read_csv(isolate2lineage_file, header=None, sep='\t')[0]

detection_lim =  0.01

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))

true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                                iso2lineage_dct)

true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                        lineage2phylogroup_file)


strainest_complete_dir = 'fig3/StrainEst/Cacnes_dropped_all'
strainest_derep_dir = 'fig3/StrainEst/Cacnes_dropped_rep'

strainge_complete_dir = 'fig3/StrainGE/Cacnes_dropped_all'
strainge_derep_dir = 'fig3/StrainGE/Cacnes_dropped_rep'

strainscan_complete_dir = 'fig3/StrainScan/Cacnes_dropped'
strainscan_derep_dir = 'fig3/StrainScan/Cacnes_dropped_rep'

# StrainEst
strainest_out_complete = helper.parse_strainest_out(strainest_complete_dir,
                                                    sample_names,
                                                    strain_names,
                                                    reference_genome)

strainest_out_derep = helper.parse_strainest_out(strainest_derep_dir,
                                                sample_names,
                                                strain_names,
                                                reference_genome)
#StrainGE
strainge_out_complete = helper.parse_strainge_out(strainge_complete_dir,
                                                    sample_names,
                                                    strain_names)

strainge_out_derep = helper.parse_strainge_out(strainge_derep_dir,
                                                sample_names,
                                                strain_names)

#StrainScan
strainscan_out_complete = helper.parse_strainscan_out(strainscan_complete_dir,
                                                    sample_names,
                                                    strain_names)

strainscan_out_derep = helper.parse_strainscan_out(strainscan_derep_dir,
                                                    sample_names,
                                                    strain_names)

# Merge frequencies by lineage
strainest_freqs_complete = helper.merge_frequencies_by_lineage(strainest_out_complete, iso2lineage_dct)
strainest_freqs_derep = helper.merge_frequencies_by_lineage(strainest_out_derep, iso2lineage_dct)

strainge_freqs_complete = helper.merge_frequencies_by_lineage(strainge_out_complete, iso2lineage_dct)
strainge_freqs_derep = helper.merge_frequencies_by_lineage(strainge_out_derep, iso2lineage_dct)

strainscan_freqs_complete = helper.merge_frequencies_by_lineage(strainscan_out_complete, iso2lineage_dct)
strainscan_freqs_derep = helper.merge_frequencies_by_lineage(strainscan_out_derep, iso2lineage_dct)

# Convert to phylogroup
strainest_freqs_phylo_complete = helper.convert_to_phylogroup(strainest_freqs_complete,
                                                              lineage2phylogroup_file)
strainest_freqs_phylo_derep = helper.convert_to_phylogroup(strainest_freqs_derep,
                                                            lineage2phylogroup_file)

strainge_freqs_phylo_complete = helper.convert_to_phylogroup(strainge_freqs_complete,
                                                            lineage2phylogroup_file)
strainge_freqs_phylo_derep = helper.convert_to_phylogroup(strainge_freqs_derep,
                                                        lineage2phylogroup_file)

strainscan_freqs_phylo_complete = helper.convert_to_phylogroup(strainscan_freqs_complete,
                                                            lineage2phylogroup_file)
strainscan_freqs_phylo_derep = helper.convert_to_phylogroup(strainscan_freqs_derep,
                                                        lineage2phylogroup_file)

# Remove low-frequency abundances
strainest_freqs_complete[strainest_freqs_complete < detection_lim] = 0
strainest_freqs_derep[strainest_freqs_derep < detection_lim] = 0
strainest_freqs_phylo_complete[strainest_freqs_phylo_complete < detection_lim] = 0
strainest_freqs_phylo_derep[strainest_freqs_phylo_derep < detection_lim] = 0

strainge_freqs_complete[strainge_freqs_complete < detection_lim] = 0
strainge_freqs_derep[strainge_freqs_derep < detection_lim] = 0
strainge_freqs_phylo_complete[strainge_freqs_phylo_complete < detection_lim] = 0
strainge_freqs_phylo_derep[strainge_freqs_phylo_derep < detection_lim] = 0

strainscan_freqs_complete[strainscan_freqs_complete < detection_lim] = 0
strainscan_freqs_derep[strainscan_freqs_derep < detection_lim] = 0
strainscan_freqs_phylo_complete[strainscan_freqs_phylo_complete < detection_lim] = 0
strainscan_freqs_phylo_derep[strainscan_freqs_phylo_derep < detection_lim] = 0


#%% Supplement: StrainEst, StrainGE results between complete and dereplicated dbs

recall_lineage, precision_lineage, \
f1_lineage, l2_lineage = helper.calc_benchmarking_stats([strainest_freqs_complete,strainest_freqs_derep,
                                                        strainge_freqs_complete,strainge_freqs_derep,
                                                        strainscan_freqs_complete,strainscan_freqs_derep],
                                                        ['StrainEst (Complete)','StrainEst (Derep.)',
                                                         'StrainGE (Complete)','StrainGE (Derep.)',
                                                         'StrainScan (Complete)','StrainScan (Derep.)'],
                                                        true_abundances_lineage,
                                                        sample_names, '_C')

recall_phylo, precision_phylo, \
f1_phylo, l2_phylo = helper.calc_benchmarking_stats([strainest_freqs_phylo_complete,strainest_freqs_phylo_derep,
                                                    strainge_freqs_phylo_complete,strainge_freqs_phylo_derep,
                                                    strainscan_freqs_phylo_complete,strainscan_freqs_phylo_derep],
                                                    ['StrainEst (Complete)','StrainEst (Derep.)',
                                                     'StrainGE (Complete)','StrainGE (Derep.)',
                                                     'StrainScan (Complete)','StrainScan (Derep.)'],
                                                    true_abundances_phylo,
                                                    sample_names, '_C')

#%% Supplement: StrainEst, StrainGE results between complete and dereplicated dbs (PLOT)


fig3b, axs3b = plot_precision_recall(recall_lineage, precision_lineage,
                                        recall_phylo, precision_phylo,
                                        ['g','r','g','b','b','r'],"C. acnes (25% dropped)",
                                        linestyle_ls=['solid','dotted','dotted','dotted','solid','solid'],
                                        plot_dims=[10,6])

fig3b.show()

# fig3b.savefig('supplemental/figs5F.pdf',format='pdf')

fig3b2, axs3b2 = plot_f1_l2_score(f1_lineage, f1_phylo,
                                 ['g','r','g','b','b','r'],"C. acnes (25% dropped)", 'F1',
                                 linestyle_ls=['solid','dotted','dotted','dotted','solid','solid'],
                                 plot_dims=[10,3])

fig3b2.show()

# fig3b2.savefig('supplemental/figs5G.pdf',format='pdf')

fig3b3, axs3b3 = plot_f1_l2_score(l2_lineage, l2_phylo,
                                 ['g','r','g','b','b','r'],'C. acnes', 'L2',
                                 linestyle_ls=['solid','dotted','dotted','dotted','solid','solid'])
fig3b3.show()

# fig3b3.savefig('supplemental/figS6D.pdf',format='pdf')
#%% Difference in precision between novel and known benchmarks

# # Combine perfect and dropped into 1 df
# Cacnes_precision_covX['index'] = 'Perfect'
# Cacnes_dropped_precision_covX['index'] = '25% Holdout'

# Cacnes_precision_covX.loc[np.in1d(Cacnes_precision_covX['variable'],['StrainGE']),'value'] += 0.25


# Cacnes_precision_covX.loc[np.in1d(Cacnes_precision_covX['variable'],['StrainEst']),'value'] += 0.2


# Cacnes_precision_covX_phylo['index'] = 'Perfect'
# Cacnes_dropped_precision_covX_phylo['index'] = '25% Holdout'

# Sepi_precision_covX['index'] = 'Perfect'
# Sepi_dropped_precision_covX['index'] = '25% Holdout'

# Sepi_dropped_precision_covX.loc[np.in1d(Sepi_dropped_precision_covX['variable'],['StrainGE']),'value'] -= 0.25

# Sepi_dropped_precision_covX.loc[np.in1d(Sepi_dropped_precision_covX['variable'],['StrainScan']),'value'] -= 0.2


# Sepi_precision_covX_phylo['index'] = 'Perfect'
# Sepi_dropped_precision_covX_phylo['index'] = '25% Holdout'



# Cacnes_precision_lineage = pd.concat([Cacnes_precision_covX, Cacnes_dropped_precision_covX])
# Cacnes_precision_phylo = pd.concat([Cacnes_precision_covX_phylo, Cacnes_dropped_precision_covX_phylo])
# Sepi_precision_lineage = pd.concat([Sepi_precision_covX, Sepi_dropped_precision_covX])
# Sepi_precision_phylo = pd.concat([Sepi_precision_covX_phylo, Sepi_dropped_precision_covX_phylo])

# fig, axs = plt.subplots(1, 4)

# fig.set_size_inches(10, 4)

# for idx, df in enumerate([Cacnes_precision_lineage, Cacnes_precision_phylo,
#                             Sepi_precision_lineage, Sepi_precision_phylo]):
#     sns.pointplot(data = df, x='index', y='value', hue='variable',
#                 ax = axs[idx], palette = ['g','k','r','b'], dodge=True, markers='o', scale=0.7)
#     axs[idx].legend([])
#     if idx == 0:
#         axs[idx].set_ylabel('Precision', **fmt); axs[0].set_xlabel('')
#     else:
#         axs[idx].set_ylabel('', **fmt); axs[0].set_xlabel('')
#     axs[idx].tick_params(axis='both', which='major', labelsize=12)
#     axs[idx].tick_params(axis='x', rotation=45)
#     axs[idx].set_ylim(0,1.05)
#     plt.setp(axs[idx].collections, alpha=.8) #for the markers
#     plt.setp(axs[idx].lines, alpha=.8, linewidth=1)       #for the lines

# axs[0].set_title('C. acnes', **fmt)
# axs[1].set_title('C. acnes', **fmt)
# axs[2].set_title('S. epidermidis', **fmt)
# axs[3].set_title('S. epi phylogroup', **fmt)

# fig.tight_layout()
# fig.savefig('fig3/fig3e.pdf',format='pdf')



#%% Supplemental: Thresholding on posterior distributions is better than thresholding on point estimates

import classify_module as classify

def reclassify_phlame_out_dir(sample_arr,
                              phlame_out_dir,
                              reference_genome,
                              max_pi,
                              min_snps,
                              min_prob,
                              point_estimate=False):
    '''
    Reclassify frequency files from an output directory with new parameters.
    '''
    
    jt_frequencies = pd.DataFrame()
    
    for sample in sample_arr:
                
        freq_file = f'{phlame_out_dir}/{sample}_ref_{reference_genome}_frequencies.csv'
        data_file = f'{phlame_out_dir}/{sample}_ref_{reference_genome}_fitinfo.data'
        
        sample_freqs = reclassify_from_data(data_file,
                                            freq_file,
                                            max_pi=max_pi,
                                            min_snps=min_snps,
                                            min_prob=min_prob,
                                            point_estimate=point_estimate)
        
        sample_freqs.columns = [sample]
        
        jt_frequencies = pd.concat((jt_frequencies,sample_freqs), axis=1)
        
    return jt_frequencies.T

def reclassify_from_data(path_to_data_file,
                         path_to_frequencies_file,
                         max_pi=0.3,
                         min_snps=10,
                         min_prob=0.75,
                         point_estimate=False):

    if min_snps!=10:
        raise Exception('Changes in the min_snps threshold from default (10) will have to be re-MCMCed')      
    
    # Load in sample
    sample_frequencies = classify.Frequencies(path_to_frequencies_file)
    data = classify.FrequenciesData(path_to_data_file)
    
    # What clades have enough SNPs to be modeled in the original run
    model_bool = np.logical_or.reduce((data.counts_MLE != -1),1)

    # Make new_frequencies data structure to fill
    new_frequencies = np.full_like(sample_frequencies.freqs['Relative abundance'],0)
    
    # new_probs = np.full_like(data.prob,-1)
    
    # Recalc frequencies with new thresholds
    for i in range(len(new_frequencies)):
        
        # Ignore the clades not to model
        if not model_bool[i]:
            continue
        
        # Calc new prob; move on if below threshold
        pi_chain = data.chain[i]['pi']

        if point_estimate:
            if data.counts_MAP[i]['pi'] > max_pi:
                continue
        else:
            if np.sum(pi_chain < max_pi)/len(pi_chain) < min_prob:
                continue
        
 

        # Only consider the parts of the chain below max_pi
        chain_bool = pi_chain < max_pi
        
        a_mean = np.mean(data.chain[i]['a'][chain_bool])
        b_mean = np.mean(data.chain[i]['b'][chain_bool])
        
        counts_mean = a_mean/b_mean
        
        # Point out instances where MAP pi is low but probability is not high
        # if (data.counts_MAP[i]['pi'] < max_pi) & (np.sum(pi_chain < max_pi)/len(pi_chain) < min_prob):
        #     print(f'Sample {path_to_frequencies_file}, Clade {sample_frequencies.freqs.index[i]} has low MAP pi ({data.counts_MAP[i]["pi"]:.2f}) and low probability ({np.sum(pi_chain < max_pi)/len(pi_chain):.2f}). reported frequency is {counts_mean/data.total_MLE[i][0]:.6f}')

        new_frequencies[i] = counts_mean/data.total_MLE[i][0]

    new_frequencies_df = pd.DataFrame(new_frequencies, 
                                      index=sample_frequencies.freqs.index,
                                      dtype=float)
    
    return new_frequencies_df

#%% Systematic comparison of point estimate

pointest_AUROC_lineage = np.zeros((20,6,20))
posterior_AUROC_lineage = np.zeros((20,6,20))
pointest_AUROC_phylo = np.zeros((20,6,20))
posterior_AUROC_phylo = np.zeros((20,6,20))

for idx_, max_pi in enumerate(np.arange(0.05,1.01,0.05)):

    phlame_freqs_pointestimate = reclassify_phlame_out_dir(sample_names, path_to_phlame_dir, reference_genome,
                                                            max_pi=max_pi, min_snps=10, min_prob=.5, point_estimate=True)
    
    phlame_freqs = reclassify_phlame_out_dir(sample_names, path_to_phlame_dir, reference_genome,
                                                max_pi=max_pi, min_snps=10, min_prob=.5, point_estimate=False)

    # Remove low frequency calls
    phlame_freqs_pointestimate[phlame_freqs_pointestimate < 0.01] = 0
    phlame_freqs[phlame_freqs < 0.01] = 0

    phlame_freqs_pointestimate_lineage = phlame_freqs_pointestimate[lineage_IDs_dropped]
    phlame_freqs_pointestimate_phylo = phlame_freqs_pointestimate[phylogroup_IDs_dropped]

    phlame_freqs_lineage = phlame_freqs[lineage_IDs_dropped]
    phlame_freqs_phylo = phlame_freqs[phylogroup_IDs_dropped]

    # Calculate recall and precision

    recall_lineage, precision_lineage, \
    f1_lineage, l2_lineage = helper.calc_benchmarking_stats([phlame_freqs_pointestimate_lineage,
                                                                    phlame_freqs_lineage],
                                                                    ['PHLAME (point estimate thresholding)',
                                                                    'PHLAME (posterior thresholding)'],
                                                                    true_abundances_lineage,
                                                                    sample_names, '_C')

    recall_phylo, precision_phylo, \
    f1_phylo, l2_phylo = helper.calc_benchmarking_stats([phlame_freqs_pointestimate_phylo,
                                                                phlame_freqs_phylo],
                                                                ['PHLAME (point estimate thresholding)',
                                                                'PHLAME (posterior thresholding)'],
                                                                true_abundances_phylo,
                                                                sample_names, '_C')
        
    for covidx, coverage in enumerate(np.unique(recall_lineage['index'])):
        pointest_AUROC_lineage[idx_,covidx] = recall_lineage.query(f'index=="{coverage}" & variable=="PHLAME (point estimate thresholding)"')['value']
        posterior_AUROC_lineage[idx_,covidx] = recall_lineage.query(f'index=="{coverage}" & variable=="PHLAME (posterior thresholding)"')['value']
        pointest_AUROC_phylo[idx_,covidx] = recall_phylo.query(f'index=="{coverage}" & variable=="PHLAME (point estimate thresholding)"')['value']
        posterior_AUROC_phylo[idx_,covidx] = recall_phylo.query(f'index=="{coverage}" & variable=="PHLAME (posterior thresholding)"')['value']

#%% Calc AUROC

pointest_AUROC_calced_lineage = np.zeros((20,6))
posterior_AUROC_calced_lineage = np.zeros((20,6))
pointest_AUROC_calced_phylo = np.zeros((20,6))
posterior_AUROC_calced_phylo = np.zeros((20,6))

import sklearn.metrics as metrics

for idx in range(20):
    for coverage in range(6):
        pointest_AUROC_calced_lineage[idx,coverage] = metrics.auc(np.unique(np.arange(0.05,1.01,0.05)),
                                                        pointest_AUROC_lineage[idx,coverage])
        posterior_AUROC_calced_lineage[idx,coverage] = metrics.auc(np.unique(np.arange(0.05,1.01,0.05)),
                                                        posterior_AUROC_lineage[idx,coverage])
        pointest_AUROC_calced_phylo[idx,coverage] = metrics.auc(np.unique(np.arange(0.05,1.01,0.05)),
                                                        pointest_AUROC_phylo[idx,coverage])
        posterior_AUROC_calced_phylo[idx,coverage] = metrics.auc(np.unique(np.arange(0.05,1.01,0.05)),
                                                        posterior_AUROC_phylo[idx,coverage])
        
# Melt dataframes
pointest_AUROC_calced_lineage_df = pd.melt(pd.DataFrame(pointest_AUROC_calced_lineage, index=np.unique(np.arange(0.05,1.01,0.05)), columns=np.unique(np.unique(recall_lineage['index']))))
posterior_AUROC_calced_lineage_df = pd.melt(pd.DataFrame(posterior_AUROC_calced_lineage, index=np.unique(np.arange(0.05,1.01,0.05)), columns=np.unique(np.unique(recall_lineage['index']))))
pointest_AUROC_calced_phylo_df = pd.melt(pd.DataFrame(pointest_AUROC_calced_phylo, index=np.unique(np.arange(0.05,1.01,0.05)), columns=np.unique(np.unique(recall_phylo['index']))))
posterior_AUROC_calced_phylo_df = pd.melt(pd.DataFrame(posterior_AUROC_calced_phylo, index=np.unique(np.arange(0.05,1.01,0.05)), columns=np.unique(np.unique(recall_phylo['index']))))

pointest_AUROC_calced_lineage_df['method'] = 'PHLAME (point estimate thresholding)'
posterior_AUROC_calced_lineage_df['method'] = 'PHLAME (posterior thresholding)'
pointest_AUROC_calced_phylo_df['method'] = 'PHLAME (point estimate thresholding)'
posterior_AUROC_calced_phylo_df['method'] = 'PHLAME (posterior thresholding)'

pointest_AUROC_calced_lineage_df.columns = ['index','value','variable']
posterior_AUROC_calced_lineage_df.columns = ['index','value','variable']
pointest_AUROC_calced_phylo_df.columns = ['index','value','variable']
posterior_AUROC_calced_phylo_df.columns = ['index','value','variable']

AUROC_comparison_lineage = pd.concat([pointest_AUROC_calced_lineage_df, posterior_AUROC_calced_lineage_df])
AUROC_comparison_phylo = pd.concat([pointest_AUROC_calced_phylo_df, posterior_AUROC_calced_phylo_df])

fig2, axs2 = plot_f1_l2_score(AUROC_comparison_lineage, AUROC_comparison_phylo,
                                ['r','k'],'C. acnes 25% dropped','F1')

# axs2[0].set_ylim(0.5,1)
# axs2[1].set_ylim(0.5,1)
axs2[0].set_ylabel('AUROC', **fmt)
# axs2[0].set_title('Lineage level', **fmt)
#%%
fig, axs = plot_precision_recall(recall_lineage, precision_lineage,
                                        recall_phylo, precision_phylo,
                                        ['r','k'],'C. acnes')

fig.show()


fig2, axs2 = plot_f1_l2_score(f1_lineage, f1_phylo,
                                ['r','k'],'C. acnes','F1')

fig2.show()

#%% PROPER POSTERIOR VS POINT ESTIMATE COMPARISON WITH MLE IMPLEMENTATION

TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Cacnes_community_abundances_dropped.csv'

path_to_samples_csv = 'fig3/Cacnes_samples.csv'
path_to_phlame_dir = 'fig3/PHLAME/Cacnes_dropped'
path_to_phlame_MLE_dir = 'fig3/PHLAME/Cacnes_dropped_MLE'

isolate2lineage_file = 'fig3/Cacnes_lineage2assembly_dropped_phylo.txt'
lineage2phylogroup_file = 'fig3/Cacnes_lineage2phylogroups.txt'
# isolate2lineage_file = 'fig3/Sepi_lineage2assembly.txt'
# lineage2phylogroup_file = 'fig3/Sepi_lineage2phylogroups.txt'

reference_genome = 'Pacnes_C1'
# reference_genome = 'SepidermidisATCC12228'

detection_lim =  0.01

iso2lineage_dct = {}
with open(isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = lineinfo[1]

lineage2phylogroup_dct = {}

with open(lineage2phylogroup_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


lineage_IDs = np.unique(list(iso2lineage_dct.values()))
phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))
phylogroup_IDs = ['A','C','E','F','H','K'] # C. acnes
# phylogroup_IDs = ['1','2','4']

sample_names = pd.read_csv(path_to_samples_csv)['Sample']

#%% PROPER POSTERIOR VS POINT ESTIMATE COMPARISON WITH MLE IMPLEMENTATION
# Read in true frequencies
true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

strain_names = true_abundances.columns
true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
                                                             iso2lineage_dct)
# Normalize true abundances to 1
true_abundances_lineage = true_abundances_lineage.div(true_abundances_lineage.sum(axis=1), axis=0)
true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
                                                     lineage2phylogroup_file)


#PHLAME 
phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                  path_to_phlame_dir,
                                                  reference_genome)
phlame_freqs_mle = helper.read_phlame_frequencies_NEW(sample_names,
                                                    path_to_phlame_MLE_dir,
                                                    reference_genome)

# Remove low-frequency abundances
phlame_freqs[phlame_freqs < detection_lim] = 0
phlame_freqs_mle[phlame_freqs_mle < detection_lim] = 0

phlame_freqs_lineage = phlame_freqs[lineage_IDs]
phlame_freqs_phylo = phlame_freqs[phylogroup_IDs]

phlame_freqs_mle_lineage = phlame_freqs_mle[lineage_IDs]
phlame_freqs_mle_phylo = phlame_freqs_mle[phylogroup_IDs]

# S epi only
# phlame_freqs_phylo.columns = phylogroup_IDs
# phlame_freqs_mle_phylo.columns = phylogroup_IDs

#%% Fig 2B1: Calc precision and recall stats

recall_lineage, precision_lineage, \
f1_lineage, l2_lineage = helper.calc_benchmarking_stats([phlame_freqs_lineage,phlame_freqs_mle_lineage],
                                                        ['Bayesian','MLE'],
                                                        true_abundances_lineage,
                                                        sample_names, '_C')

recall_phylo, precision_phylo, \
f1_phylo, l2_phylo = helper.calc_benchmarking_stats([phlame_freqs_phylo,phlame_freqs_mle_phylo],
                                                        ['Bayesian','MLE'],
                                                        true_abundances_phylo,
                                                        sample_names, '_C')

# precision_lineage.groupby('variable').mean().reset_index()
# precision_phylo.groupby('variable').mean().reset_index()

#%% Fig 2B2: Plot precision and recall , C. acnes

fig3b, axs3b = plot_precision_recall(recall_lineage, precision_lineage,
                                        recall_phylo, precision_phylo,
                                        ['k','r'],'C. acnes 25% dropped',
                                        alpha=0.6)

fig3b.show()

# fig3b.savefig('supplemental/figS18C1.pdf',format='pdf')

fig3b2, axs3b2 = plot_f1_l2_score(f1_lineage, f1_phylo,
                                ['k','r'],'C. acnes 25% dropped','F1',
                                alpha=0.6)

fig3b2.show()

# fig3b2.savefig('supplemental/figS18C2.pdf',format='pdf')

# fig3b3, axs3b3 = plot_f1_l2_score(l2_lineage, l2_phylo,
#                                     ['k','r'],'S. epi','L2')

# fig3b3.show()

# fig3b3.savefig('supplemental/figS6A.pdf',format='pdf')


#%% Speed test

path_to_speedtest_dir = 'fig3/benchmarks'
path_to_samples_csv = 'fig3/Sepi_samples.csv'

sample_names = pd.read_csv(path_to_samples_csv)['Sample'][:20]

# reference_genome = 'Pacnes_C1'
reference_genome = 'SepidermidisATCC12228'

speedtest_df = pd.DataFrame(columns=['PHLAME','PHLAME_Bayesian','StrainEst','StrainEst Derep.','StrainGE','StrainGE Derep.','StrainScan','StrainScan Derep.'])

for i, method in enumerate(['Sepi/rule_phlame_classify',
                            'Sepi/rule_phlame_classify_bayesian',
                            'Sepi/rule_strainest_classify',
                            'Sepi_rep/rule_strainest_classify',
                            'Sepi/rule_strainge_classify',
                            'Sepi_rep/rule_strainge_classify',
                            'Sepi/rule_strainscan_classify',
                            'Sepi_rep/rule_strainscan_classify']):
    
    method_ls = []
    for sample in sample_names:

        if i < 4:
            path_to_benchmark_file = f'{path_to_speedtest_dir}/{method}_{sample}_ref_{reference_genome}.benchmark'
        else:
            path_to_benchmark_file = f'{path_to_speedtest_dir}/{method}_{sample}.benchmark'

        with open(path_to_benchmark_file,'r') as f:
            seconds = float(f.readlines()[1].split('\t')[0])

        if i < 4:
            path_to_bowtie2_file = f'{path_to_speedtest_dir}/Sepi/rule_bowtie2_{sample}_ref_{reference_genome}.benchmark'
            with open(path_to_bowtie2_file,'r') as f:
                seconds += float(f.readlines()[1].split('\t')[0])

            path_to_sam2bam_file = f'{path_to_speedtest_dir}/Sepi/rule_sam2bam_{sample}_ref_{reference_genome}.benchmark'
            with open(path_to_sam2bam_file,'r') as f:
                seconds += float(f.readlines()[1].split('\t')[0])


        method_ls.append(seconds/60)

    speedtest_df.iloc[:,i] = method_ls

#%% Plot mean + 95% CI as bars

colors = ['k','k','b','b','g','g','r','r']
# Add broken y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)
fig.set_size_inches(5,5)

speedtest_df.mean().plot(kind='bar', yerr=speedtest_df.std()*1.96/np.sqrt(20),
                          ax=ax1, capsize=5, color=colors, edgecolor='k', alpha=0.6)
speedtest_df.mean().plot(kind='bar', yerr=speedtest_df.std()*1.96/np.sqrt(20),
                          ax=ax2, capsize=5, color=colors, edgecolor='k', alpha=0.6)

ax1.set_ylim(100,300)
ax2.set_ylim(0,30)

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Make the cut-out slanted lines
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


ax2.set_ylabel('Wall time (min)', **fmt)
ax2.set_xticklabels(['PHLAME (MLE)','PHLAME (Bayesian)','StrainEst','StrainEst (Derep.)','StrainGE','StrainGE (Derep.)','StrainScan','StrainScan (Derep.)'], rotation=45,  ha='right', **fmt)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

ax1.set_title('S. epidermidis', **fmt)

# axs.set_yscale('log')

fig.tight_layout()

fig.savefig('supplemental/figS10B.pdf',format='pdf')

#%% StrainGE different background comparisons

# TRUE_DISTRIBUTIONS_FILE = 'fig3/true_frequencies/Sepi_community_abundances.csv'
# TRUE_DISTRIBUTIONS_FILE_dropped = 'fig3/true_frequencies/Sepi_community_abundances_dropped_1X.csv'

# path_to_samples_csv = 'fig3/Sepi_samples.csv'
# path_to_strainge_zymo = 'fig3/StrainGE_benchmarks_Sepi/Sepi_Zymo'
# path_to_strainge_mockskin = 'fig3/StrainGE_benchmarks_Sepi/Sepi_mockskin'
# path_to_strainge_dropped_zymo = 'fig3/StrainGE_benchmarks_Sepi/Sepi_dropped_Zymo'
# path_to_strainge_dropped_mockskin = 'fig3/StrainGE_benchmarks_Sepi/Sepi_dropped_mockskin'

# isolate2lineage_file = 'fig3/Sepi_lineage2assembly.txt'
# iso2lineage_dropped_file = 'fig3/Sepi_lineage2assembly_dropped_phylo_1X.txt'
# lineage2phylogroup_file = 'fig3/Sepi_lineage2phylogroups.txt'

# reference_genome = 'SepidermidisATCC12228'

# detection_lim =  0.01

# iso2lineage_dropped_dct = {}
# with open(iso2lineage_dropped_file,'r') as f:
#     for line in f:
#         lineinfo = line.rstrip('\n').split('\t')
#         iso2lineage_dropped_dct[lineinfo[0]] = lineinfo[1]

# iso2lineage_dct = {}
# with open(isolate2lineage_file,'r') as f:
#     for line in f:
#         lineinfo = line.rstrip('\n').split('\t')
#         iso2lineage_dct[lineinfo[0]] = lineinfo[1]

# lineage2phylogroup_dct = {}

# with open(lineage2phylogroup_file,'r') as f:
#     for line in f:
#         lineinfo = line.rstrip('\n').split('\t')
#         lineage2phylogroup_dct[lineinfo[0]] = lineinfo[1]


# lineage_IDs = np.unique(list(iso2lineage_dct.values()))
# lineage_IDs_dropped = np.unique(list(iso2lineage_dropped_dct.values()))
# phylogroup_IDs = np.unique(list(lineage2phylogroup_dct.values()))

# phylogroup_IDs_dropped = ['1','2','4'] # S epidermidis

# #%% StrainGE different background comparisons
# sample_names = pd.read_csv(path_to_samples_csv)['Sample']

# # Read in true frequencies
# true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

# true_abundances_dropped = pd.read_csv(TRUE_DISTRIBUTIONS_FILE_dropped, header=0, index_col=0).T

# strain_names = list(iso2lineage_dropped_dct.keys())
# true_abundances_lineage = helper.merge_frequencies_by_lineage(true_abundances,
#                                                              iso2lineage_dct)

# true_abundances_phylo = helper.convert_to_phylogroup(true_abundances_lineage,
#                                                      lineage2phylogroup_file)

# true_abundances_lineage_dropped = helper.merge_frequencies_by_lineage(true_abundances_dropped,
#                                                                      iso2lineage_dct)
# true_abundances_phylo_dropped = helper.convert_to_phylogroup(true_abundances_lineage_dropped,
#                                                         lineage2phylogroup_file)

# true_abundances_lineage_dropped = true_abundances_lineage_dropped[lineage_IDs_dropped]
# true_abundances_phylo_dropped = true_abundances_phylo_dropped[phylogroup_IDs_dropped]

# strainge_out_zymo = helper.parse_strainge_out(path_to_strainge_zymo,
#                                             sample_names, strain_names)
# strainge_out_mockskin = helper.parse_strainge_out(path_to_strainge_mockskin,
#                                             sample_names, strain_names)
# strainge_out_dropped_zymo = helper.parse_strainge_out(path_to_strainge_dropped_zymo,
#                                             sample_names, strain_names)
# strainge_out_dropped_mockskin = helper.parse_strainge_out(path_to_strainge_dropped_mockskin,
#                                             sample_names, strain_names)

# strainge_freqs_zymo = helper.merge_frequencies_by_lineage(strainge_out_zymo, iso2lineage_dct)
# strainge_freqs_mockskin = helper.merge_frequencies_by_lineage(strainge_out_mockskin, iso2lineage_dct)
# strainge_freqs_dropped_zymo = helper.merge_frequencies_by_lineage(strainge_out_dropped_zymo,                iso2lineage_dropped_dct)
# strainge_freqs_dropped_mockskin = helper.merge_frequencies_by_lineage(strainge_out_dropped_mockskin, iso2lineage_dct)

# strainge_freqs_zymo_phylo = helper.convert_to_phylogroup(strainge_freqs_zymo,
#                                                         lineage2phylogroup_file)
# strainge_freqs_mockskin_phylo = helper.convert_to_phylogroup(strainge_freqs_mockskin,
#                                                         lineage2phylogroup_file)
# strainge_freqs_dropped_zymo_phylo = helper.convert_to_phylogroup(strainge_freqs_dropped_zymo,
#                                                         lineage2phylogroup_file)
# strainge_freqs_dropped_mockskin_phylo = helper.convert_to_phylogroup(strainge_freqs_dropped_mockskin,
#                                                         lineage2phylogroup_file)

# # Remove low-frequency abundances
# strainge_freqs_zymo[strainge_freqs_zymo < detection_lim] = 0
# strainge_freqs_mockskin[strainge_freqs_mockskin < detection_lim] = 0
# strainge_freqs_dropped_zymo[strainge_freqs_dropped_zymo < detection_lim] = 0
# strainge_freqs_dropped_mockskin[strainge_freqs_dropped_mockskin < detection_lim] = 0

# #%% StrainGE different background comparisons

# coverage_toint_dict = {'01X':0.1,
#                         '05X':0.5,
#                         '1X':1,
#                         '5X':5,
#                         '10X':10,
#                         '20X':20}

# coverages = [coverage_toint_dict[sam.split("_S", 1)[0]] for sam in sample_names] # S. epi

# method_results = [strainge_freqs_zymo,strainge_freqs_mockskin,strainge_freqs_dropped_zymo,strainge_freqs_dropped_mockskin]
# method_names = ['Zymo','Mockskin','Zymo_dropped','Mockskin_dropped']
# true_abundances_ls = [true_abundances_lineage,true_abundances_lineage,true_abundances_lineage_dropped,true_abundances_lineage_dropped]

# recall_ls = []; precision_ls = []; f1_ls = []; abserr_ls = []
# for freqs, method, truth in zip(method_results,
#                                   method_names,
#                                   true_abundances_ls):

#     recall = calc_recall(freqs, truth)
#     precision, FDR = calc_precision(freqs, truth)
#     abserr = calc_abs_error(freqs, truth)

#     recall_ls.append(recall)
#     precision_ls.append(precision)
#     f1_ls.append((2*recall*precision)/(recall+precision))
#     abserr_ls.append(np.vstack([abserr.T,[method]*len(abserr)]).T)

# recall_combined_melt = combine_melt_PRstats(recall_ls, method_names, coverages)
# precision_combined_melt = combine_melt_PRstats(precision_ls, method_names, coverages)
# f1_combined_melt = combine_melt_PRstats(f1_ls, method_names, coverages)

# recall_combined_melt['index'][recall_combined_melt['index']=='5.0'] = '1.2'
# precision_combined_melt['index'][precision_combined_melt['index']=='5.0'] = '1.2'
# f1_combined_melt['index'][f1_combined_melt['index']=='5.0'] = '1.2'
# f1_combined_melt = f1_combined_melt.fillna(0)

# ##### Absolute Error ######
# AbsErr_combined = pd.DataFrame(np.concatenate(abserr_ls),
#                                columns=['index','value','variable'])
# AbsErr_combined['value'] = pd.to_numeric(AbsErr_combined['value'])
# AbsErr_combined['index'][AbsErr_combined['index']=='10X'] = 'Z10X'
# AbsErr_combined.sort_values('index',inplace=True)

# #%% StrainGE different background comparisons

# recall_ls = []; precision_ls = []; f1_ls = []; abserr_ls = []

# method_results = [strainge_freqs_zymo_phylo,strainge_freqs_mockskin_phylo,strainge_freqs_dropped_zymo_phylo,strainge_freqs_dropped_mockskin_phylo]
# method_names = ['Zymo','Mockskin','Zymo_dropped','Mockskin_dropped']
# true_abundances_ls = [true_abundances_phylo,true_abundances_phylo,true_abundances_phylo_dropped,true_abundances_phylo_dropped]

# for freqs, method, truth in zip(method_results,
#                                method_names,
#                                true_abundances_ls):
    
#         recall = calc_recall(freqs, truth)
#         precision, FDR = calc_precision(freqs, truth)
#         abserr = calc_abs_error(freqs, truth)
    
#         recall_ls.append(recall)
#         precision_ls.append(precision)
#         f1_ls.append((2*recall*precision)/(recall+precision))
#         abserr_ls.append(np.vstack([abserr.T,[method]*len(abserr)]).T)

# recall_combined_phylo = combine_melt_PRstats(recall_ls, method_names, coverages)
# precision_combined_phylo = combine_melt_PRstats(precision_ls, method_names, coverages)
# f1_combined_phylo = combine_melt_PRstats(f1_ls, method_names, coverages)

# recall_combined_phylo['index'][recall_combined_phylo['index']=='5.0'] = '1.2'
# precision_combined_phylo['index'][precision_combined_phylo['index']=='5.0'] = '1.2'
# f1_combined_phylo['index'][f1_combined_phylo['index']=='5.0'] = '1.2'
# f1_combined_phylo = f1_combined_phylo.fillna(0)

# ##### Absolute Error ######
# AbsErr_combined_phylo = pd.DataFrame(np.concatenate(abserr_ls),
#                                columns=['index','value','variable'])
# AbsErr_combined_phylo['value'] = pd.to_numeric(AbsErr_combined_phylo['value'])
# AbsErr_combined_phylo['index'][AbsErr_combined_phylo['index']=='10X'] = 'Z10X'
# AbsErr_combined_phylo.sort_values('index',inplace=True)


# #%% StrainGE different background comparisons
# fig3b, ax2b = plt.subplots(2,2)
# fig3b.set_size_inches(8, 5)

# # Recall lineage level
# k=0
# for (n, grp) in recall_combined_melt.groupby("variable"):
#     x = grp.groupby("index").mean()
#     std = grp.groupby("index").std()/2
#     draw_dodge(ax2b[0,0].errorbar, x.index, x.values, 
#                yerr = std.values.flatten(), ax=ax2b[0,0], 
#                dodge=Dodge[k], marker="o", label=n)
#     k+=1
# ax2b[0,0].legend([])
# ax2b[0,0].set_ylabel('Recall', **fmt)
# ax2b[0,0].set_title('Lineage level', **fmt)
# ax2b[0,0].set_xticks([0,1,2,3,4,5])
# ax2b[0,0].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
# ax2b[0,0].set_ylim(-0.01,1.01)
# ax2b[0,0].tick_params(axis='both', which='major', labelsize=15)

# # Precision, lineage level
# k=0
# for (n, grp) in precision_combined_melt.groupby("variable"):
#     x = grp.groupby("index").mean()
#     std = grp.groupby("index").std()/2
#     draw_dodge(ax2b[0,1].errorbar, x.index, x.values, 
#                yerr =std.values.flatten(), ax=ax2b[0,1], 
#                dodge=Dodge[k], marker="o", label=n)
#     k+=1

# ax2b[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# ax2b[0,1].set_ylabel('Precision', **fmt)
# ax2b[0,1].set_title('Lineage level', **fmt)
# ax2b[0,1].set_xticks([0,1,2,3,4,5])
# ax2b[0,1].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
# ax2b[0,1].set_ylim(-0.01,1.01)
# ax2b[0,1].tick_params(axis='both', which='major', labelsize=15)

# # Recall, phylogroup level
# k=0
# for (n, grp) in recall_combined_phylo.groupby("variable"):
#     x = grp.groupby("index").mean()
#     std = grp.groupby("index").std()/2
#     draw_dodge(ax2b[1,0].errorbar, x.index, x.values, 
#                yerr = std.values.flatten(), ax=ax2b[1,0], 
#                dodge=Dodge[k], marker="o", label=n)
#     k+=1

# ax2b[1,0].legend([])
# ax2b[1,0].set_xlabel('Coverage', **fmt)
# ax2b[1,0].set_ylabel('Recall', **fmt)
# ax2b[1,0].set_title('Phylogroup level', **fmt)
# ax2b[1,0].set_xticks([0,1,2,3,4,5])
# ax2b[1,0].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
# ax2b[1,0].set_ylim(-0.01,1.01)
# ax2b[1,0].tick_params(axis='both', which='major', labelsize=15)

# # Precision, phylogroup level
# k=0
# for (n, grp) in precision_combined_phylo.groupby("variable"):
#     x = grp.groupby("index").mean()
#     std = grp.groupby("index").std()/2
#     draw_dodge(ax2b[1,1].errorbar, x.index, x.values, 
#                yerr =std.values.flatten(), ax=ax2b[1,1], 
#                dodge=Dodge[k], marker="o", label=n)
#     k+=1

# ax2b[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# ax2b[1,1].set_xlabel('Coverage', **fmt)
# ax2b[1,1].set_ylabel('Precision', **fmt)
# ax2b[1,1].set_title('Phylogroup level', **fmt)
# ax2b[1,1].set_xticks([0,1,2,3,4,5])
# ax2b[1,1].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
# ax2b[1,1].set_ylim(-0.01,1.01)
# ax2b[1,1].tick_params(axis='both', which='major', labelsize=15)

# fig3b.tight_layout()

# fig3b.savefig('fig3/fig3d1.pdf',format='pdf')

# #%% StrainGE different background comparisons
# fig3b, ax2b = plt.subplots(1, 2)
# fig3b.set_size_inches(9, 3)

# # F1, lineage level
# k=0
# for (n, grp) in f1_combined_melt.groupby("variable"):
#     x = grp.groupby("index").mean()
#     x.index = x.index.astype(str)
#     std = grp.groupby("index").std()/2
#     std.index = std.index.astype(str)
#     draw_dodge(ax2b[0].errorbar, x.index, x.values, 
#                yerr =std.values.flatten(), ax=ax2b[0], 
#                dodge=Dodge[k], marker="o", label=n)
#     k+=1

# ax2b[0].legend([])
# ax2b[0].set_xlabel('Coverage', **fmt)
# ax2b[0].set_ylabel('F1 score', **fmt)
# ax2b[0].set_title('Lineage level', **fmt)
# ax2b[0].set_xticks([0,1,2,3,4,5])
# ax2b[0].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
# ax2b[0].set_ylim(-0.01,1.01)
# ax2b[0].tick_params(axis='both', which='major', labelsize=15)

# # F1, phylogroup level
# k=0
# for (n, grp) in f1_combined_phylo.groupby("variable"):
#     x = grp.groupby("index").mean()
#     x.index = x.index.astype(str)
#     std = grp.groupby("index").std()/2
#     std.index = std.index.astype(str)
#     draw_dodge(ax2b[1].errorbar, x.index, x.values, 
#                yerr = std.values.flatten(), ax=ax2b[1], 
#                dodge=Dodge[k], marker="o", label=n)
#     k+=1

# ax2b[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# ax2b[1].set_xlabel('Coverage', **fmt)
# ax2b[1].set_title('Phylogroup level', **fmt)
# ax2b[1].set_xticks([0,1,2,3,4,5])
# ax2b[1].set_xticklabels(['0.1X','0.5X','1X','5X','10X','20X'])
# ax2b[1].set_ylim(-0.01,1.01)
# ax2b[1].tick_params(axis='both', which='major', labelsize=15)

# fig3b.tight_layout()

# fig3b.savefig('fig3/fig3d2.pdf',format='pdf')

#%% Proportion of a sample comprised of strains lacking phylogenetic representation
# (this is the go into tubes benchmark but on simulated data)

# Sepi_out_dir = 'fig3/percent_called_benchmark/Sepi'
# Sepi_out_dir = 'fig3/percent_called_benchmark/Sepi'
# path_to_Sepi_samples_csv = 'fig3/percent_called_benchmark/Sepi_samples.csv'
# Sepi_sample_names = pd.read_csv(path_to_Sepi_samples_csv)['Sample']
# path_to_Sepi_samples_csv = 'fig3/percent_called_benchmark/Sepi_samples.csv'
# Sepi_sample_names = pd.read_csv(path_to_Sepi_samples_csv)['Sample']

# # Sepi_frequencies = helper.read_phlame_frequencies_NEW(Sepi_sample_names,
# #                                                         Sepi_out_dir,
# #                                                         'SepidermidisATCC12228')
# # Sepi_percent_called = Sepi_frequencies.sum(axis=1)

# Sepi_frequencies = helper.read_phlame_frequencies_NEW(Sepi_sample_names,
#                                                       Sepi_out_dir,
#                                                       'SepidermidisATCC12228')
# Sepi_percent_called = Sepi_frequencies.sum(axis=1)

# true_missing_proportion = np.array([sample[-2] for sample in Sepi_sample_names], dtype=float)/10

# percent_called_means = []
# percent_called_stds = []
# for pcalled in [Sepi_percent_called, Sepi_percent_called]:
#     mean_ = []
#     std_ = []
#     for prop in np.unique(true_missing_proportion):
#         this_prop_bool = true_missing_proportion == prop
#         mean_.append(np.mean(pcalled[this_prop_bool]))
#         std_.append(np.std(pcalled[this_prop_bool]))
#     percent_called_means.append(mean_)
#     percent_called_stds.append(std_)

# fig3c, ax2c = plt.subplots()
# fig3c.set_size_inches(6, 5)

# k=0
# for (sp, means, stds) in zip(['C. acnes','S. epidermidis'],
#                           percent_called_means, percent_called_stds):
#     draw_dodge(ax2c.errorbar, 
#                np.unique(true_missing_proportion), means, 
#                yerr = stds, ax=ax2c, 
#                dodge=Dodge[k], marker="o", label=sp)
#     k+=1
# ax2c.legend()
# ax2c.set_xlabel('True proportion of missing strains', **fmt)
# ax2c.set_ylabel('Measured proportion of missing strains', **fmt)
# ax2c.set_title('Lineage level', **fmt)
# ax2c.plot([0,1],[0,1], color='red')
# ax2c.tick_params(axis='both', which='major', labelsize=15)


# #%% Fig 2C: Plot
# fig3c, ax2c = plt.subplots()
# fig3c.set_size_inches(5.5, 5)

# ax2c.scatter(true_missing_proportion, Sepi_percent_called,
#              color='black', alpha=0.5)
# ax2c.plot([0,1],[0,1], color='red')
# ax2c.set_xlabel('True proportion of novel strains', **fmt)
# ax2c.set_ylabel('Measured proportion of novel strains', **fmt)

# # fig3c.savefig('fig3/fig3c.pdf',format='pdf')
# # fig3c.savefig('fig3/fig3c.png',format='png')