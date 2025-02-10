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
from scipy import stats
import ete3
import glob
import gzip
import pickle

import phlame.classify as classify
from scipy.special import gamma, gammaln, loggamma, polygamma, factorial

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures')

import fig_helper_functions as helper

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

fmt={'fontsize':14,
     'fontname':'Helvetica'}

%matplotlib auto

#%% Functions 

# =============================================================================
# Input / Output Functions
# =============================================================================

def read_sample_frequencies(sample_names, 
                            phlame_out_dir,
                            reference_genome):
    '''
    Read and consolidate frequency files from an output directory
    '''
    
    jt_frequencies = pd.DataFrame()
    jt_divergence = pd.DataFrame()
    jt_confidence = pd.DataFrame()
    
    for sample_name in sample_names:

        idx = sample_name.split('idx')[1].split('_r')[0]
        
        frequencies_file = f'{phlame_out_dir}/{sample_name}_ref_{reference_genome}_cfr_{idx}_frequencies.csv'
        
        sample_freqs = helper.Frequencies(frequencies_file)
        
        frequencies = sample_freqs.freqs['Relative abundance']        
        divergence = sample_freqs.freqs['Estimated divergence']
        confidence = sample_freqs.freqs['Confidence score']
        
        frequencies.name = sample_name
        divergence.name = sample_name
        confidence.name = sample_name

        jt_frequencies = pd.concat((jt_frequencies,frequencies), axis=1)
        jt_divergence = pd.concat((jt_divergence,divergence), axis=1)
        jt_confidence = pd.concat((jt_confidence,confidence), axis=1)

    return jt_frequencies.T, jt_divergence.T, jt_confidence.T

def get_MAP_pi(sample_names,
               phlame_out_dir,
               reference_genome):
    
    jt_MAP_pi = pd.DataFrame()

    for sample_name in sample_names:
        idx = sample_name.split('idx')[1].split('_r')[0]
        
        data_file = f'{phlame_out_dir}/{sample_name}_ref_{reference_genome}_cfr_{idx}_fitinfo.data'

        data = helper.FrequenciesData(data_file)

        if len(data.chain[0]) > 0:
            pi_chain = data.chain[0]['pi']
            pi_counts, pi_bins = np.histogram(pi_chain, bins=np.arange(0,1.01,0.005))

            MAP_pi = pi_bins[np.where(pi_counts == pi_counts.max())][0]
        else:
            MAP_pi = -1

        MAP_pi_series = pd.Series(MAP_pi)
        MAP_pi_series.name = sample_name

        jt_MAP_pi = pd.concat((jt_MAP_pi, MAP_pi_series), axis=1)

    return jt_MAP_pi.T

def zinb_rvs(npts, mu, alpha_, pi):
    n_ = alpha_
    p_ =  alpha_/(alpha_+mu)

    if pi < 0 or pi > 1 or mu <= 0:
        return np.zeros(npts)
    else:
        
        inflated_zero = stats.bernoulli.rvs(pi, size=npts)
        return (1 - inflated_zero) * stats.nbinom.rvs(n_, p_, size=npts)
    
def zinb_pdf(x, mu, alpha_, pi):
    n_ = alpha_
    p_ =  alpha_/(alpha_+mu)
    
    return (1-pi)*stats.nbinom.pmf(x, n_, p_) + pi*stats.bernoulli.pmf(x, 0)


#%% Fig 2B: Extended graphic of the informative pos plot

# Random seed is just to find a toy example that looks nice
np.random.seed(1)

npts = 40
counts = zinb_rvs(npts, 10, 1e6, 0.2)
total_counts = zinb_rvs(npts, 25, 1e6, 0)

df = pd.DataFrame({'counts':counts,
                   'total_counts':total_counts-counts})

fig1e1, axs1e1 = plt.subplots()
fig1e1.set_size_inches(5,2)

df.plot(kind='bar', stacked=True, ax=axs1e1, color=['r', 'k'], alpha=0.75, edgecolor='k', linewidth=1,
        width=0.8, legend=False)

axs1e1.set_ylabel('Sequencing\ndepth', fontsize=16, fontname='Helvetica')
axs1e1.set_xlabel('Positions', fontsize=16, fontname='Helvetica')
axs1e1.set_xticks([])
axs1e1.set_yticks([])

fig1e1.tight_layout()

# fig1e1.savefig('fig2/fig2b.pdf', format='pdf')


#%% Fig 2C: Example of PHLAME modeling and full posterior over pi

# Set random seed
np.random.seed(0)

# Simulation parameters
npts = 50 # Number of data points to simulate

counts_ls = []; bincounts_ls = []; bins_ls = []
total_counts = zinb_rvs(npts, 25, 1e6, 0)

for i, (lambda_, pi_) in enumerate(zip([10,10],[0.01,0.25])):
    
    counts = zinb_rvs(npts, lambda_, 1e6, pi_)
    #1/alpha because rvs takes the alpha=0 -> poisson parameterization

    res = classify.countsCSS_NEW(counts, total_counts)
    prob, hpd = res.fit(max_pi = 0.3)

    pi_chain = res.chain['pi'][500:]
    bincounts, bins = np.histogram(pi_chain, bins=np.arange(0,1.01,0.01), density=True)

    counts_ls.append(counts)
    bincounts_ls.append(bincounts)
    bins_ls.append(bins)

#%% Fig. 2C plot

fig1b1, axs = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 0.7]})
fig1b1.set_size_inches(4,3)

for i, (lambdas, pis, bincounts, bins) in enumerate(zip([10,10],[0.00,0.25], bincounts_ls, bins_ls)):

    axs[i,0].plot(np.arange(0,40), zinb_pdf(np.arange(0,40), lambdas, 1e6, pis)*npts, color='r', lw=1)
    axs[i,0].fill_between(x=np.arange(0,40), y1=zinb_pdf(np.arange(0,40), lambdas, 1e6, pis)*npts,
                        color= 'r',
                        alpha= 0.5)
    axs[i,0].plot(np.arange(0,40), zinb_pdf(np.arange(0,40), 25, 1e6, 0)*npts, color='k', lw=1)
    axs[i,0].fill_between(x=np.arange(0,40), y1=zinb_pdf(np.arange(0,40), 25, 1e6, 0)*npts,
                        color= 'k',
                        alpha= 0.6)

    axs[i,0].set_ylim(0, 15)
    axs[i,0].set_xlim(-1, 40)
    axs[i,0].set_ylabel('# of positions', fontsize=12, fontname='Helvetica')
    axs[1,0].set_xlabel('Sequencing depth', fontsize=12, fontname='Helvetica')
    axs[i,0].tick_params(axis='both', which='major', labelsize=11)

    axs[i,1].plot(bins[:-1], bincounts, color='k')
    axs[1,1].set_xlabel('$\pi$', fontsize=12, fontname='Helvetica')
    axs[i,1].set_ylabel('Density', fontsize=12, fontname='Helvetica')
    axs[i,1].set_yticks([])
    axs[i,1].tick_params(axis='both', which='major', labelsize=11)
    axs[1,1].set_ylim(0, 10)
    axs[0,1].set_ylim(0, 45)

fig1b1.tight_layout(w_pad=0.1, h_pad=0.1)
# fig1b1.savefig('fig2/fig2c.pdf', dpi=300)

#%% Fig 1E3: Example of PHLAME modeling and full posterior over pi

np.random.seed(0)

# Simulation parameters
npts = 50 

counts_ls = []; total_counts_ls = [];
bincounts_ls = []; bins_ls = []

counts = zinb_rvs(npts, 1, 1/0.1, 0.2)

for i, alpha_ in enumerate([1e-6,0.5]):
    
    total_counts = zinb_rvs(npts, 15, 1/alpha_, 0)

    res = classify.countsCSS_NEW(counts, total_counts)
    prob, hpd = res.fit(max_pi = 0.3)

    pi_chain = res.chain['pi'][500:]
    bincounts, bins = np.histogram(pi_chain, bins=np.arange(0,1.01,0.01), density=True)

    counts_ls.append(counts)
    total_counts_ls.append(total_counts)
    bincounts_ls.append(bincounts)
    bins_ls.append(bins)

#%% Fig. 1F2 plot

fig1b1, axs = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 0.7]})
fig1b1.set_size_inches(4,3)

for i, (alphas, bincounts, bins) in enumerate(zip([0.5,1e-6], bincounts_ls[::-1], bins_ls[::-1])):

    axs[i,0].plot(np.arange(0,40), zinb_pdf(np.arange(0,40), 1.5, 1/0.1, 0.2)*npts, color='r', lw=1)
    axs[i,0].fill_between(x=np.arange(0,40), y1=zinb_pdf(np.arange(0,40), 1.5, 1/0.1, 0.2)*npts,
                        color= 'r',
                        alpha= 0.5)
    axs[i,0].plot(np.arange(0,40), zinb_pdf(np.arange(0,40), 15, 1/alphas, 0)*npts, color='k', lw=1)
    axs[i,0].fill_between(x=np.arange(0,40), y1=zinb_pdf(np.arange(0,40), 15, 1/alphas, 0)*npts,
                        color= 'k',
                        alpha= 0.6)

    axs[i,0].set_ylim(0, 25)
    axs[i,0].set_xlim(-1, 40)
    axs[i,0].set_ylabel('# of positions', fontsize=12, fontname='Helvetica')
    axs[1,0].set_xlabel('Depth across position', fontsize=12, fontname='Helvetica')
    axs[i,0].tick_params(axis='both', which='major', labelsize=11)

    axs[i,1].plot(bins[:-1], bincounts, color='k')
    axs[1,1].set_xlabel('$\pi$', fontsize=12, fontname='Helvetica')
    axs[i,1].set_ylabel('Density', fontsize=12, fontname='Helvetica')
    axs[i,1].set_yticks([])
    axs[i,1].tick_params(axis='both', which='major', labelsize=11)
    axs[i,1].set_ylim(0, 5)

fig1b1.tight_layout(w_pad=0.1, h_pad=0.1)
# fig1b1.savefig('fig2/fig2d.pdf', dpi=300)

#%% Fig 2e1, C. acnes
# =============================================================================
# C. acnes
# =============================================================================

path_to_samples_csv = 'fig2/drop_1_clade/Cacnes_samples_NEW.csv'
path_to_scaled_tree = 'fig2/drop_1_clade/Cacnes_megatree_GTR_isonames_calls_v2.tre'
path_to_clades_file = 'fig2/drop_1_clade/Cacnes_megatree_GTR_isonames_cladeIDs_v2.txt'
path_to_pusb_data = 'fig2/drop_1_clade/Cacnes_pusb.csv'

reference_genome = 'Pacnes_C1'
phlame_out_dir = 'fig2/drop_1_clade/Cacnes_NEW'

# Read in tree
tree = ete3.Tree(path_to_scaled_tree, format=1)
tree.standardize()

# Read in clade IDs
clade_IDs = pd.read_csv(path_to_clades_file,
                         sep="\t", index_col=0, header=None)


# Read in directional proportion_unshared_background between pair of sister clades
pusb_metadata = pd.read_csv(path_to_pusb_data)

# Read in sample names
sample_names = pd.read_csv(path_to_samples_csv, index_col=0)['Sample'].values

#%% Calculate estimated divergence from pi HPD region
# =============================================================================

tenxbool = np.array([sample_.startswith('10X') for sample_ in sample_names])
tenx_sample_names = sample_names[tenxbool]


frequencies, divergence, confidence = read_sample_frequencies(list(tenx_sample_names), 
                                                              phlame_out_dir,
                                                              reference_genome)
MAP_pis = get_MAP_pi(list(tenx_sample_names), 
                     phlame_out_dir,
                     reference_genome)

pusb_ls = []
MAP_95_lower = []
MAP_95_upper = []
ls = []

for sample in tenx_sample_names:

    cfr_idx = sample.split('idx')[1].split('_r')[0]
    pusb_ls.append(pusb_metadata['pusb'].iloc[np.where(pusb_metadata['classifier_idx'] == int(cfr_idx))].values[0])
    MAP_95_lower.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[0][1:])
    MAP_95_upper.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[1][:-1])
    ls.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0])

not_modeled_bool = np.where(MAP_pis[0] != -1)

pusb_ls_sortedidx = np.argsort(pusb_ls)


#%% Plot
# =============================================================================

tenx_sample_names_idx = [sample_.split('idx')[1] for sample_ in tenx_sample_names]

tenx_supports = pusb_metadata['support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values
tenx_parent_supports = pusb_metadata['parent_support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values

# include_bool = (tenx_supports[not_modeled_bool] > 75) | (tenx_parent_supports[not_modeled_bool] == 0)
include_bool = (tenx_parent_supports[not_modeled_bool] > 75) | (tenx_parent_supports[not_modeled_bool] == 0)

tenx_notmodeled_bool = np.array([sample_.startswith('10X') for sample_ in sample_names])

slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(pusb_ls)[not_modeled_bool][include_bool], 
                                                               np.array(MAP_pis)[not_modeled_bool][include_bool].flatten())

residuals = abs(np.array(MAP_pis)[not_modeled_bool][include_bool].flatten() - (slope*np.array(pusb_ls)[not_modeled_bool][include_bool] + intercept))


fig, axs = plt.subplots()
fig.set_size_inches(4,3.7)
# Plot proportion unshared background against MAP pi estimate
axs.plot([0,1],[0,1], color='k', lw=1.5, alpha = 0.5)
axs.scatter(np.array(pusb_ls)[not_modeled_bool][include_bool],
             np.array(MAP_pis)[not_modeled_bool][include_bool], 
            color='k', s=20, label='C. acnes')

# axs.scatter(np.array(pusb_ls)[not_modeled_bool][~include_bool],
#              np.array(MAP_pis)[not_modeled_bool][~include_bool], 
#             color='r', s=20, label='S. epidermidis')

axs.set_xlabel("$DV_b$", **fmt)
axs.set_ylabel("Estimated $\pi$",**fmt)
axs.tick_params(axis='both', which='major', labelsize=12)
# axs.legend(fontsize=14)

fig.tight_layout()

# Print correlation coeff, bottom right
axs.text(0.95, 0.05, f"r = {r_value:.2f}", 
         ha='right', va='bottom', transform=axs.transAxes, fontsize=14)

# fig.savefig('fig2/fig2e1.pdf', dpi=300, format='pdf', bbox_inches='tight')


#%% Drop off of accuracy with lower coverage

r2_ls = []
pval_ls = []
for coverage in ['10X','5X','1X','halfX','ptoneX']:

    covbool = np.array([sample_.startswith(coverage) for sample_ in sample_names])
    cov_sample_names = sample_names[covbool]

    frequencies, divergence, confidence = read_sample_frequencies(list(sample_names[covbool]),
                                                                    phlame_out_dir,
                                                                    reference_genome)
    MAP_pis = get_MAP_pi(list(sample_names[covbool]),
                            phlame_out_dir,
                            reference_genome)
    
    pusb_ls = []
    MAP_95_lower = []
    MAP_95_upper = []
    ls = []
    cfr_idx_ls = []

    for sample in cov_sample_names:

        cfr_idx = sample.split('idx')[1].split('_r')[0]
        cfr_idx_ls.append(cfr_idx)
        pusb_ls.append(pusb_metadata['pusb'].iloc[np.where(pusb_metadata['classifier_idx'] == int(cfr_idx))].values[0])
        MAP_95_lower.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[0][1:])
        MAP_95_upper.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[1][:-1])
        ls.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0])

    not_modeled_bool = np.where(MAP_pis[0] != -1)

    pusb_ls_sortedidx = np.argsort(pusb_ls)

    cov_parent_supports = pusb_metadata['parent_support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values

    include_bool = (cov_parent_supports[not_modeled_bool] > 75) | (cov_parent_supports[not_modeled_bool] == 0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(pusb_ls)[not_modeled_bool][include_bool], 
                                                                np.array(MAP_pis)[not_modeled_bool][include_bool].flatten())
    
    r2_ls.append(r_value)
    pval_ls.append(p_value)

#%% Plot
# =============================================================================

fig, axs = plt.subplots()

fig.set_size_inches(3,2)

axs.plot(np.arange(5), r2_ls, color='k', lw=1.5, marker='o')

axs.set_xticks(np.arange(5))
axs.set_xticklabels(['10X','5X','1X','0.5X','0.1X'])
axs.set_xlabel('Coverage', **fmt)
axs.set_ylabel('r', **fmt)
axs.set_ylim(0,1)
axs.tick_params(axis='both', which='major', labelsize=12)
fig.tight_layout()

# Add stars for significant p-values
for i, pval in enumerate(pval_ls):
    if pval < 0.05/15:
        axs.text(i, 1.05, '*', fontsize=14, ha='center', va='center')

fig.savefig('fig2/fig2f1.pdf', format='pdf')

#%% Fig 2e2, S. epidermidis drop 1 clade

# =============================================================================
# S. epi
# =============================================================================

path_to_samples_csv = 'fig2/drop_1_clade/Sepi_samples_NEW.csv'
path_to_scaled_tree = 'fig2/drop_1_clade/Sepi_acera_norecomb_GTR_calls.tre'
path_to_pusb_data = 'fig2/drop_1_clade/Sepi_acera_pusb.csv'

reference_genome = 'SepidermidisATCC12228'
# phlame_out_dir = 'fig2/drop_1_clade/Sepi'
phlame_out_dir = 'fig2/drop_1_clade/Sepi_NEW'

# Read in tree
tree = ete3.Tree(path_to_scaled_tree, format=1)
tree.standardize()

# Read in directional proportion_unshared_background between pair of sister clades
pusb_metadata = pd.read_csv(path_to_pusb_data)

# Read in sample names
sample_names = pd.read_csv(path_to_samples_csv, index_col=0)['Sample'].values

#%% Calculate estimated divergence from pi HPD region
# =============================================================================

tenxbool = np.array([sample_.startswith('10X') for sample_ in sample_names])
tenx_sample_names = sample_names[tenxbool]


frequencies, divergence, confidence = read_sample_frequencies(list(tenx_sample_names), 
                                                              phlame_out_dir,
                                                              reference_genome)
MAP_pis = get_MAP_pi(list(tenx_sample_names), 
                     phlame_out_dir,
                     reference_genome)

pusb_ls = []
MAP_95_lower = []
MAP_95_upper = []
ls = []

for sample in tenx_sample_names:

    cfr_idx = sample.split('idx')[1].split('_r')[0]
    pusb_ls.append(pusb_metadata['pusb'].iloc[np.where(pusb_metadata['classifier_idx'] == int(cfr_idx))].values[0])
    MAP_95_lower.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[0][1:])
    MAP_95_upper.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[1][:-1])
    ls.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0])

not_modeled_bool = np.where(MAP_pis[0] != -1)

pusb_ls_sortedidx = np.argsort(pusb_ls)


#%% Plot
# =============================================================================

tenx_sample_names_idx = [sample_.split('idx')[1] for sample_ in tenx_sample_names]

tenx_supports = pusb_metadata['support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values
tenx_parent_supports = pusb_metadata['parent_support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values

# include_bool = (tenx_supports[not_modeled_bool] > 75) | (tenx_parent_supports[not_modeled_bool] == 0)
include_bool = (tenx_parent_supports[not_modeled_bool] > 75) | (tenx_parent_supports[not_modeled_bool] == 0)

tenx_notmodeled_bool = np.array([sample_.startswith('10X') for sample_ in sample_names])

slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(pusb_ls)[not_modeled_bool][include_bool], 
                                                               np.array(MAP_pis)[not_modeled_bool][include_bool].flatten())

residuals = abs(np.array(MAP_pis)[not_modeled_bool][include_bool].flatten() - (slope*np.array(pusb_ls)[not_modeled_bool][include_bool] + intercept))


fig, axs = plt.subplots()
fig.set_size_inches(4,3.7)
# Plot proportion unshared background against MAP pi estimate
axs.plot([0,1],[0,1], color='k', lw=1.5, alpha = 0.5)
axs.scatter(np.array(pusb_ls)[not_modeled_bool][include_bool],
             np.array(MAP_pis)[not_modeled_bool][include_bool], 
            color='k', s=20, label='S. epidermidis')

# axs.scatter(np.array(pusb_ls)[not_modeled_bool][~include_bool],
#              np.array(MAP_pis)[not_modeled_bool][~include_bool], 
#             color='r', s=20, label='S. epidermidis')

axs.set_xlabel("$DV_b$", **fmt)
axs.set_ylabel("Estimated $\pi$",**fmt)
axs.tick_params(axis='both', which='major', labelsize=12)
# axs.legend(fontsize=14)

fig.tight_layout()

# Print correlation coeff, bottom right
axs.text(0.95, 0.05, f"r = {r_value:.2f}", 
         ha='right', va='bottom', transform=axs.transAxes, fontsize=14)

# fig.savefig('fig2/fig2e2.pdf', dpi=300, format='pdf', bbox_inches='tight')


#%% Drop off of accuracy with lower coverage

r2_ls = []
pval_ls = []

for coverage in ['10X','5X','1X','halfX','ptoneX']:

    covbool = np.array([sample_.startswith(coverage) for sample_ in sample_names])
    cov_sample_names = sample_names[covbool]

    frequencies, divergence, confidence = read_sample_frequencies(list(sample_names[covbool]),
                                                                    phlame_out_dir,
                                                                    reference_genome)
    MAP_pis = get_MAP_pi(list(sample_names[covbool]),
                            phlame_out_dir,
                            reference_genome)
    
    pusb_ls = []
    MAP_95_lower = []
    MAP_95_upper = []
    ls = []
    cfr_idx_ls = []

    for sample in cov_sample_names:

        cfr_idx = sample.split('idx')[1].split('_r')[0]
        cfr_idx_ls.append(cfr_idx)
        pusb_ls.append(pusb_metadata['pusb'].iloc[np.where(pusb_metadata['classifier_idx'] == int(cfr_idx))].values[0])
        MAP_95_lower.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[0][1:])
        MAP_95_upper.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[1][:-1])
        ls.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0])

    not_modeled_bool = np.where(MAP_pis[0] != -1)

    pusb_ls_sortedidx = np.argsort(pusb_ls)

    cov_parent_supports = pusb_metadata['parent_support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values

    include_bool = (cov_parent_supports[not_modeled_bool] > 75) | (cov_parent_supports[not_modeled_bool] == 0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(pusb_ls)[not_modeled_bool][include_bool], 
                                                                np.array(MAP_pis)[not_modeled_bool][include_bool].flatten())
    
    r2_ls.append(r_value)
    pval_ls.append(p_value)

#%% Plot
# =============================================================================

fig, axs = plt.subplots()

fig.set_size_inches(3,2)

axs.plot(np.arange(5), r2_ls, color='k', lw=1.5, marker='o')

axs.set_xticks(np.arange(5))
axs.set_xticklabels(['10X','5X','1X','0.5X','0.1X'])
axs.set_xlabel('Coverage', **fmt)
axs.set_ylabel('r', **fmt)
axs.set_ylim(0,1)
axs.tick_params(axis='both', which='major', labelsize=12)
fig.tight_layout()

# Add stars for significant p-values
for i, pval in enumerate(pval_ls):
    if pval < 0.05/15:
        axs.text(i, 1.05, '*', fontsize=14, ha='center', va='center')

fig.savefig('fig2/fig2f2.pdf', format='pdf')

#%% Fig 2e3 E coli drop 1 clade

# =============================================================================
# E coli
# =============================================================================

path_to_samples_csv = 'fig2/drop_1_clade/Ecoli_samples_NEW.csv'
path_to_scaled_tree = 'fig2/drop_1_clade/Ecoli_GTR_isonames.tre'
path_to_pusb_data = 'fig2/drop_1_clade/Ecoli_pusb.csv'

reference_genome = 'Ecoli_ASM584'
phlame_out_dir = 'fig2/drop_1_clade/Ecoli_NEW'

# Read in tree
tree = ete3.Tree(path_to_scaled_tree, format=1)
tree.standardize()

# Read in directional proportion_unshared_background between pair of sister clades
pusb_metadata = pd.read_csv(path_to_pusb_data)

# Read in sample names
sample_names = pd.read_csv(path_to_samples_csv, index_col=0)['Sample'].values

#%% Fig 2e3, E coli drop 1 clade
# Calculate estimated divergence from pi HPD region
# =============================================================================


tenxbool = np.array([sample_.startswith('10X') for sample_ in sample_names])
tenx_sample_names = sample_names[tenxbool]


frequencies, divergence, confidence = read_sample_frequencies(list(tenx_sample_names), 
                                                              phlame_out_dir,
                                                              reference_genome)
MAP_pis = get_MAP_pi(list(tenx_sample_names), 
                     phlame_out_dir,
                     reference_genome)

pusb_ls = []
MAP_95_lower = []
MAP_95_upper = []
ls = []

for sample in tenx_sample_names:

    cfr_idx = sample.split('idx')[1].split('_r')[0]
    pusb_ls.append(pusb_metadata['pusb'].iloc[np.where(pusb_metadata['classifier_idx'] == int(cfr_idx))].values[0])
    MAP_95_lower.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[0][1:])
    MAP_95_upper.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[1][:-1])
    ls.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0])

not_modeled_bool = np.where(MAP_pis[0] != -1)

pusb_ls_sortedidx = np.argsort(pusb_ls)


#%% Fig 2e3, E coli drop 1 clade
# Plot
# =============================================================================

tenx_sample_names_idx = [sample_.split('idx')[1] for sample_ in tenx_sample_names]

tenx_supports = pusb_metadata['support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values
tenx_parent_supports = pusb_metadata['parent_support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values

# include_bool = (tenx_supports[not_modeled_bool] > 75) | (tenx_parent_supports[not_modeled_bool] == 0)
include_bool = (tenx_parent_supports[not_modeled_bool] > 75) | (tenx_parent_supports[not_modeled_bool] == 0)

tenx_notmodeled_bool = np.array([sample_.startswith('10X') for sample_ in sample_names])

slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(pusb_ls)[not_modeled_bool][include_bool], 
                                                               np.array(MAP_pis)[not_modeled_bool][include_bool].flatten())

residuals = abs(np.array(MAP_pis)[not_modeled_bool][include_bool].flatten() - (slope*np.array(pusb_ls)[not_modeled_bool][include_bool] + intercept))


fig, axs = plt.subplots()
fig.set_size_inches(4,3.7)
# Plot proportion unshared background against MAP pi estimate
axs.plot([0,1],[0,1], color='k', lw=1.5, alpha = 0.5)
axs.scatter(np.array(pusb_ls)[not_modeled_bool][include_bool],
             np.array(MAP_pis)[not_modeled_bool][include_bool], 
            color='k', s=20, label='E. coli')

# axs.scatter(np.array(pusb_ls)[not_modeled_bool][~include_bool],
#              np.array(MAP_pis)[not_modeled_bool][~include_bool], 
#             color='r', s=20, label='E. coli')

axs.set_xlabel("$DV_b$", **fmt)
axs.set_ylabel("Estimated $\pi$",**fmt)
axs.tick_params(axis='both', which='major', labelsize=12)
# axs.legend(fontsize=14)

fig.tight_layout()

# Print correlation coeff, bottom right
axs.text(0.95, 0.05, f"r = {r_value:.2f}", 
         ha='right', va='bottom', transform=axs.transAxes, fontsize=14)

# fig.savefig('fig2/fig2e3.pdf', dpi=300, format='pdf', bbox_inches='tight')

#%% Drop off of accuracy with lower coverage

r2_ls = []
pval_ls = []

for coverage in ['10X','5X','1X','halfX','ptoneX']:

    covbool = np.array([sample_.startswith(coverage) for sample_ in sample_names])
    cov_sample_names = sample_names[covbool]

    frequencies, divergence, confidence = read_sample_frequencies(list(sample_names[covbool]),
                                                                    phlame_out_dir,
                                                                    reference_genome)
    MAP_pis = get_MAP_pi(list(sample_names[covbool]),
                            phlame_out_dir,
                            reference_genome)
    
    pusb_ls = []
    MAP_95_lower = []
    MAP_95_upper = []
    ls = []
    cfr_idx_ls = []

    for sample in cov_sample_names:

        cfr_idx = sample.split('idx')[1].split('_r')[0]
        cfr_idx_ls.append(cfr_idx)
        pusb_ls.append(pusb_metadata['pusb'].iloc[np.where(pusb_metadata['classifier_idx'] == int(cfr_idx))].values[0])
        MAP_95_lower.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[0][1:])
        MAP_95_upper.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0].split(' ')[1][:-1])
        ls.append(divergence.loc[sample][~divergence.loc[sample].isnull()][0])

    not_modeled_bool = np.where(MAP_pis[0] != -1)

    pusb_ls_sortedidx = np.argsort(pusb_ls)

    cov_parent_supports = pusb_metadata['parent_support'].iloc[np.where(pusb_metadata['classifier_idx'].isin([int(idx) for idx in tenx_sample_names_idx]))].values

    include_bool = (cov_parent_supports[not_modeled_bool] > 75) | (cov_parent_supports[not_modeled_bool] == 0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(pusb_ls)[not_modeled_bool][include_bool], 
                                                                np.array(MAP_pis)[not_modeled_bool][include_bool].flatten())
    
    r2_ls.append(r_value)
    pval_ls.append(p_value)

#%% Plot
# =============================================================================
fig, axs = plt.subplots()

fig.set_size_inches(3,2)

axs.plot(np.arange(5), r2_ls, color='k', lw=1.5, marker='o')

axs.set_xticks(np.arange(5))
axs.set_xticklabels(['10X','5X','1X','0.5X','0.1X'])
axs.set_xlabel('Coverage', **fmt)
axs.set_ylabel('r', **fmt)
axs.set_ylim(0,1)
axs.tick_params(axis='both', which='major', labelsize=12)
fig.tight_layout()

# Add stars for significant p-values
for i, pval in enumerate(pval_ls):
    if pval < 0.05/15:
        axs.text(i, 1.05, '*', fontsize=14, ha='center', va='center')

fig.savefig('fig2/fig2f3.pdf', format='pdf')


#%% Supplemental Figure: any possible % of missing clade-specific alleles >= DVb is possible by varying sequencing depth or read dispersion (Fig SXX)

# The different random seeds is just to find a toy example that looks nice,
# obviously random variable generation will result in slightly different curves each time

fmt = {'fontsize':12, 'fontname':'Helvetica'}
fig, axs = plt.subplots(2,2)
fig.set_size_inches(6,5)

npts = 100
mu_arr = [0.01,0.1,0.5,1,2,5,10,20,40]
alpha_arr = np.logspace(-6,3,20)

np.random.seed(2)
alpha_ = 1
for idxp, pi in enumerate([0,0.4]):
    mean_perc_zeros = []
    
    for idxm, mu_ in enumerate(mu_arr):

        perc_zeros_reps = np.zeros(50)

        for rep in range(50):

            counts = zinb_rvs(npts,mu_,alpha_,pi)
            perc_zeros = np.sum(counts == 0)/npts
            perc_zeros_reps[rep] = perc_zeros
        
        mean_perc_zeros.append(np.mean(perc_zeros_reps))

    axs[idxp,0].axhline(pi, color='r', lw=2, label='True $Dv_b$')
    axs[idxp,0].plot(mu_arr, mean_perc_zeros, color='k', lw=2)
    axs[idxp,0].set_xlabel('Mean sequencing depth', **fmt)
    axs[0,0].set_ylabel('% of alleles NOT\nobserved in sample', **fmt)
    axs[idxp,0].set_ylim(-0.1,1)
    # axs[idxp,1].legend()

np.random.seed(123)
mu_ = 5
for idxp, pi in enumerate([0,0.4]):
    mean_perc_zeros = []
    
    for idxa, alpha_ in enumerate(alpha_arr):
        perc_zeros_reps = np.zeros(50)

        for rep in range(50):
            counts = zinb_rvs(npts,mu_,1/alpha_,pi)
            perc_zeros = np.sum(counts == 0)/npts
            perc_zeros_reps[rep] = perc_zeros
        mean_perc_zeros.append(np.mean(perc_zeros_reps))

    axs[idxp,1].axhline(pi, color='r', lw=2, label='True $Dv_b$')
    axs[idxp,1].plot(alpha_arr, mean_perc_zeros, color='k', lw=2)
    axs[idxp,1].set_xlabel('Excess dispersion\nrelative to a Poisson', **fmt)
    axs[1,0].set_ylabel('% of alleles NOT\nobserved in sample', **fmt)
    axs[idxp,1].set_ylim(-0.1,1)
    axs[idxp,1].set_xscale('log')

fig.tight_layout()

# fig.savefig('supplemental/figS3A.pdf', dpi=300, format='pdf', bbox_inches='tight')

#%% Supplemental Figure: Likelihood calculations, a range of possible DVb and dispersion parameters can reasonably explain any given low-depth distribution 

# The random seed is just to find a toy example that looks nice,
# obviously random variable generation will result in slightly different curves each time
np.random.seed(2)

npts = 100
mu = 1
alpha_ = 1
pi = 0.4
counts = zinb_rvs(100, mu, alpha_, pi)

iter_pi = np.arange(0,1,0.01) #100
iter_lambda = np.arange(0.02,5.02,0.02) #250
iter_alpha = np.arange(.025,5.025,.025)[::-1] #200 


prob, conditionalprob = conditionalprob_ZINB(counts,
                                            iter_pi, iter_lambda, iter_alpha)

conditionalprob_lambdapi = np.mean(np.exp(conditionalprob),0)
conditionalprob_alphapi = np.mean(np.exp(conditionalprob),1)

#%%
import seaborn as sns

# PLot alpha against pi
fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [.4, 1]})
fig.set_size_inches(7,3)

axs[0].hist(counts, bins=np.arange(0,max(counts)+2),
             color='r', alpha=0.75, edgecolor='k', linewidth=1)
axs[0].set_xlabel('Sequencing depth', **fmt)
axs[0].set_ylabel('# of positions', **fmt)
axs[1] = sns.heatmap(conditionalprob_alphapi, linewidth=0)
axs[1].set_xticks(np.arange(0,101,20))
axs[1].set_xticklabels(["%.2f" % n for n in np.arange(0,1.01,0.2)])
axs[1].set_yticks(np.arange(0,201,50))
axs[1].set_yticklabels(["%.1f" % n for n in np.arange(.025,2.3,.5)[::-1]])
axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)
axs[1].set_xlabel('$DV_b$', **fmt)
axs[1].set_ylabel('Excess dispersion \n relative to a Poisson', **fmt)
fig.tight_layout()
# fig.savefig('supplemental/figS15B1.pdf', format='pdf')

fig2, axs2 = plt.subplots(1,2, gridspec_kw={'width_ratios': [.4, 1]})
fig2.set_size_inches(7,3)

axs2[0].hist(counts, bins=np.arange(0,max(counts)+2),
             color='r', alpha=0.75, edgecolor='k', linewidth=1)
axs2[0].set_xlabel('Sequencing depth', **fmt)
axs2[0].set_ylabel('# of positions', **fmt)
axs2[1] = sns.heatmap(conditionalprob_lambdapi, linewidth=0)
axs2[1].set_xticks(np.arange(0,101,20))
axs2[1].set_xticklabels(["%.2f" % n for n in np.arange(0,1.01,0.2)])
axs2[1].set_yticks(np.arange(0,250,50))
axs2[1].set_yticklabels(["%.1f" % n for n in np.arange(0.02,5.02,1)[::-1]])
axs2[1].set_yticklabels(axs2[1].get_yticklabels(), rotation=0)
axs2[1].set_xlabel('$DV_b$', **fmt)
axs2[1].set_ylabel('Mean sequencing depth', **fmt)

fig2.tight_layout()
# fig2.savefig('supplemental/figS15B2.pdf', format='pdf')

foo = np.mean(conditionalprob_lambdapi,0)

fig3, axs3 = plt.subplots()
fig3.set_size_inches(4.25,2)
axs3.plot(iter_pi, foo, 'k')
axs3.set_xticks(np.arange(0,1.01,0.2))
axs3.set_xticklabels(["%.2f" % n for n in np.arange(0,1.01,0.2)])
axs3.set_xlabel('$DV_b$', **fmt)
axs3.set_ylabel('Density', **fmt)
axs3.set_xlim(0,1)

fig3.tight_layout()
# fig3.savefig('supplemental/figS15B3.pdf', format='pdf')

#%%

import warnings
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import gamma, gammaln, loggamma, polygamma, factorial

def zinb_loglike_invalpha(params, counts):
    '''
    Zero-Inflated Negative Binomial LOG-likelihood function. Invalpha corresponds to 
    the parameterization where alpha->0, the distribution approaches a Poisson.
    '''
    # Expand params
    lambda_, alpha, pi_link = params
    # lambda_ = np.exp(np.clip(lambda_, None, EXP_UPPER_LIMIT))
    # alpha = np.exp(np.clip(alpha, None, EXP_UPPER_LIMIT))
    # pi_link = np.exp(np.clip(pi_link, None, EXP_UPPER_LIMIT))

    # lambda_ = np.exp(lambda_, dtype=np.float128)
    # alpha = np.exp(alpha, dtype=np.float128)
    # pi_link = np.exp(pi_link, dtype=np.float128)
    
    zero_idx = np.nonzero(counts == 0)[0]
    nonzero_idx = np.nonzero(counts)[0]
    
    ll_obs = np.zeros_like(counts, dtype=np.float64)
    
    ll_obs[zero_idx] = (np.log(pi_link + (1+(lambda_*alpha))**-(1/alpha)) -
                        np.log(1+pi_link))
                        
    ll_obs[nonzero_idx] = (loggamma(counts[nonzero_idx] + (1/alpha)) -
                           loggamma(counts[nonzero_idx]+1) -
                           loggamma(1/alpha) -
                           ((counts[nonzero_idx] + (1/alpha))*np.log(1+(lambda_*alpha))) + 
                           (counts[nonzero_idx]*np.log(alpha)) + 
                           (counts[nonzero_idx]*np.log(lambda_)) -
                           np.log(1+pi_link)
                           )        
    
    if np.isnan(np.sum(ll_obs)):
        print(params)

    return np.sum(ll_obs)

def conditionalprob_ZINB(counts,
                        iter_pi, iter_lambda, iter_alpha):
    
    '''Calculate the log conditional likelihood of 

    Args:
        counts (TYPE): DESCRIPTION.
        counts_lamb (TYPE): DESCRIPTION.
        counts_variance (TYPE): DESCRIPTION.

    Returns:
        None.

    '''
    counts_variance=np.var(counts)
    iter_pi = np.arange(0,1,0.01)
    
    conditionalprob=np.zeros((len(iter_alpha),len(iter_lambda),len(iter_pi)))
    for ida, alpha_ in enumerate(iter_alpha):
        for idl, lambda_ in enumerate(iter_lambda):
            for idp, pi_ in enumerate(iter_pi):
                conditionalprob[ida,idl,idp] = zinb_loglike_invalpha([lambda_,alpha_,pi_],counts)
                
    #Flatten along alphas & normalize to 1
    pi_likelihood = np.mean(conditionalprob,0)
    pi_likelihood_norm = np.exp(pi_likelihood, dtype=np.float128)/np.sum(np.exp(pi_likelihood, dtype=np.float128))
    
    #Probability that a set of parameters bounded by max_pi produced this data
    max_pi=0.25
    prob=np.sum(pi_likelihood_norm[:np.where(iter_pi == max_pi)[0][0]+1])/np.sum(pi_likelihood_norm)
    
    # Flatten along pis & normalize to 1
    alpha_likelihood = np.mean(conditionalprob,1)
    alpha_likelihood_norm = np.exp(alpha_likelihood, dtype=np.float128)/np.sum(np.exp(alpha_likelihood, dtype=np.float128))

    return prob, conditionalprob

from scipy.stats import beta
from statsmodels.base.model import GenericLikelihoodModel

def zinb_pmf(x, mu, alpha_, pi):
    
    #scipy will only take n and p as shape parameters, need to convert
    #n = number of successes
    #p = probability of success
    n_ = alpha_
    p_ =  alpha_/(alpha_+mu)

    if pi < 0 or pi > 1 or mu <= 0:
        return np.zeros_like(x)
    else:
        return ((x == 0) * pi) + ((1 - pi) * stats.nbinom.pmf(x, n_, p_))

def nb_pmf(x, mu, alpha_):
    
    #scipy will only take n and p as shape parameters, need to convert
    #n = number of successes
    #p = probability of success
    n_ = alpha_
    p_ =  alpha_/(alpha_+mu)
    
    return stats.nbinom.pmf(x, n_, p_)

class ZeroInflatedNB(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(ZeroInflatedNB, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        mu = params[0]
        alpha_ = params[1]
        pi = params[2]
        # alph = params[-1]
        # beta = params[:-1]
        # ll = _ll_nb2(self.endog, self.exog, beta, alph)
        nll = -np.log(zinb_pmf(self.endog, mu=mu, alpha_=alpha_, pi=pi))
        
        return nll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        # self.exog_names.append('alpha')
        if start_params == None:
            # Reasonable starting values
            mu_start = self.endog.mean()
            alpha_start = 1/(((self.endog.std()**2/mu_start)-1) / mu_start)
            pi_start = (self.endog == 0).mean() - nb_pmf(0, mu_start, alpha_start)
            
            start_params = np.array([mu_start, alpha_start, pi_start])

        return super(ZeroInflatedNB, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
def zinbinom_fit(counts):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedNB(counts)
        fit = model.fit()
        mu, alpha_, pi = fit.params
        
        if np.isnan(mu):
            print("Warning: mu parameter returned NaN value")
            mu=0
        if np.isnan(alpha_):
            print("Warning: alpha parameter returned NaN value")
            alpha_=0
        if np.isnan(pi):
            print("Warning: pi parameter returned NaN value")
            pi=0
        
    return abs(mu), abs(alpha_), abs(pi)

def zip_pmf(x, pi, lambda_):
    '''zero-inflated poisson function, pi is prob. of 0, lambda_ is the fit parameter'''
    if pi < 0 or pi > 1 or lambda_ <= 0:
        return np.zeros_like(x)
    else:
        return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)

class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]

        return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            lambda_start = self.endog.mean()
            excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            pi_start = excess_zeros if excess_zeros>0 else 0
            start_params = np.array([pi_start, lambda_start])
            
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)

def zip_fit(data):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedPoisson(data)
        fit = model.fit()
        pi, lambda_ = fit.params
    
    # return lambda_, pi
        CI_pi, CI_lambda_ = fit.conf_int()
        # range_CI_pi = CI_pi[1] - CI_pi[0]
        range_CI_lambda_ = CI_lambda_[1] - CI_lambda_[0]
    
    return lambda_, pi, range_CI_lambda_



#%% Supplemental Figure: attempts to measure the true DVb value from simulated read counts using either the proportion of clade-specific markers or maximum likelihood zero-inflation models return highly variable results 

np.random.seed(0)

npts = 100
mu = 1
alpha_ = 1
pi = 0.4

perc_missingalleles = []
MLE_ZIP = []
MLE_ZINB = []

MLE_ZINB_prior = []

MAP_Bayesian_prior = []
MAP_Bayesian_noprior = []

for rep in range(50):

    counts = zinb_rvs(npts, mu, alpha_, pi)
    total_counts = zinb_rvs(npts, 5, alpha_, 0)

    # Calculate the percent of missing alleles
    perc_missingalleles.append(np.sum(counts == 0)/npts)

    #  MLE of zero-inflated Poisson model
    _, pi_, _ = zip_fit(counts)
    MLE_ZIP.append(pi_)

    # MLE of zero-inflated negative binomial model
    _, _, pi_ = zinbinom_fit(counts)
    MLE_ZINB.append(pi_)

    # MLE with prior
    cts_fit = classify.countsCSS_NEW(counts,
                                    total_counts,
                                    seed=12345,
                                    mode='mle')
    
    prob = cts_fit.fit(max_pi = 0.35)
    
    MLE_ZINB_prior.append(cts_fit.counts_MLE[0])
    
    # Bayesian model (prior)
    cts_fit = classify.countsCSS_NEW(counts,
                                     total_counts,
                                     seed=12345,
                                     mode='bayesian')

    prob = cts_fit.fit(max_pi = 0.35,
                        nchain = 5000,
                        nburn = 500)
    
    MAP_Bayesian_prior.append(cts_fit.counts_MAP['pi'])

    # Bayesian model (no prior)

    cts_fit_noprior = classify.countsCSS_NEW(counts,
                                            total_counts,
                                            seed=12345, 
                                            mode='bayesian',
                                            prior=False)
    
    prob_noprior = cts_fit_noprior.fit(max_pi = 0.35,
                                       seed=12345,
                                       nchain = 5000,
                                       nburn = 500)
    
    MAP_Bayesian_noprior.append(cts_fit_noprior.counts_MAP['pi'])

    print(f"finished simulation {rep}")
    

#%%
fig, axs = plt.subplots()
fig.set_size_inches(2.5,2)

axs.hist(counts, bins=np.arange(0,max(counts)+2), color='r', alpha=0.5, edgecolor='k', linewidth=1)
axs.hist(total_counts, bins=np.arange(0,max(total_counts)+2), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs.set_xlabel('Sequencing depth', **fmt)
axs.set_ylabel('# of positions', **fmt)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.set_xlim(0,20)
fig.tight_layout()

# fig.savefig('supplemental/figS3C2.pdf', dpi=300, format='pdf', bbox_inches='tight')

#%% Plot all the fits that don't use the prior first

fig, axs = plt.subplots(4)
fig.set_size_inches(3.8,10)

axs[0].hist(perc_missingalleles, bins=np.arange(0,1.01,0.05), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs[0].axvline(pi, color='r', lw=2, label='True $DV_b$')
axs[0].set_xlabel('Estimated $DV_b$', **fmt)
axs[0].set_ylabel('# of simulations', **fmt)
axs[0].set_title('Percent of missing alleles', **fmt)
axs[0].text(0.72, 0.85, f"RMSE: {np.sqrt(np.mean((np.array(perc_missingalleles) - pi)**2)):.2f}", transform=axs[0].transAxes)
# axs[0].legend(fontsize=12)

axs[1].hist(MLE_ZIP, bins=np.arange(0,1.01,0.05), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs[1].axvline(pi, color='r', lw=2, label='True $DV_b$')
axs[1].set_xlabel('Estimated $DV_b$', **fmt)
axs[1].set_ylabel('# of simulations', **fmt)
axs[1].set_title('Maximum Likelihood\n(Zero-Inflated Poisson)', **fmt)
axs[1].text(0.72, 0.85, f"RMSE: {np.sqrt(np.mean((np.array(MLE_ZIP) - pi)**2)):.2f}", transform=axs[1].transAxes)

# axs[1].legend()

axs[2].hist(MLE_ZINB, bins=np.arange(0,1.01,0.05), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs[2].axvline(pi, color='r', lw=2, label='True $DV_b$')
axs[2].set_xlabel('Estimated $DV_b$', **fmt)
axs[2].set_ylabel('# of simulations', **fmt)
axs[2].set_title('Maximum Likelihood\n(Zero-Inflated Negative Binomial)', **fmt)
axs[2].text(0.72, 0.85, f"RMSE: {np.sqrt(np.mean((np.array(MLE_ZINB) - pi)**2)):.2f}", transform=axs[2].transAxes)

# axs[2].legend()

axs[3].hist(MAP_Bayesian_noprior, bins=np.arange(0,1.01,0.05), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs[3].axvline(pi, color='r', lw=2, label='True $DV_b$')
axs[3].set_xlabel('Estimated $DV_b$', **fmt)
axs[3].set_ylabel('# of simulations', **fmt)
axs[3].set_title('Bayesian \n(Zero-Inflated Negative Binomial,\n non-informative prior)', **fmt)
axs[3].text(0.72, 0.85, f"RMSE: {np.sqrt(np.mean((np.array(MAP_Bayesian_noprior) - pi)**2)):.2f}", transform=axs[3].transAxes)

# axs[3].legend()

fig.tight_layout()

fig.savefig('supplemental/figS3C3.pdf', dpi=300, format='pdf', bbox_inches='tight')

#%% Now plot the fits that use the prior

fig, axs = plt.subplots(2)
fig.set_size_inches(3.8,5.2)

axs[0].hist(MLE_ZINB_prior, bins=np.arange(0,1.01,0.05), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs[0].axvline(pi, color='r', lw=2, label='True $DV_b$')
axs[0].set_xlabel('Estimated $DV_b$', **fmt)
axs[0].set_ylabel('# of simulations', **fmt)
axs[0].set_title('Maximum Likelihood\n(Zero-Inflated Negative Binomial,\n forced dispersion)', **fmt)
axs[0].text(0.72, 0.85, f"RMSE: {np.sqrt(np.mean((np.array(MLE_ZINB_prior) - pi)**2)):.2f}", transform=axs[0].transAxes)

# axs[0].legend()
axs[1].hist(MAP_Bayesian_prior, bins=np.arange(0,1.01,0.05), color='k', alpha=0.5, edgecolor='k', linewidth=1)
axs[1].axvline(pi, color='r', lw=2, label='True $DV_b$')
axs[1].set_xlabel('Estimated $DV_b$', **fmt)
axs[1].set_ylabel('# of simulations', **fmt)
axs[1].set_title('Bayesian \n(Zero-Inflated Negative Binomial,\n informative prior for dispersion)', **fmt)
axs[1].text(0.72, 0.85, f"RMSE: {np.sqrt(np.mean((np.array(MAP_Bayesian_prior) - pi)**2)):.2f}", transform=axs[1].transAxes)

# axs[1].legend()

fig.tight_layout()

# fig.savefig('supplemental/figS3C4.pdf', dpi=300, format='pdf', bbox_inches='tight')

#%% Trees with bootstrap values 

Cacnes_tree = ete3.Tree('fig2/RAxML_bipartitions.Cacnes_acera_norecomb.tre', format=0)
midpoint_root = Cacnes_tree.get_midpoint_outgroup()

Cacnes_tree.set_outgroup(midpoint_root)

general_ts = ete3.TreeStyle()
general_ts.show_leaf_name=False
general_ts.scale = 0.0001
general_ts.optimal_scale_level = 'full'
# Remove blue dot on every node
nstyle = ete3.NodeStyle()
nstyle['shape']=''; nstyle['size']=0
nstyle['vt_line_width']=0.025
nstyle['hz_line_width']=0.025
general_ts.legend_position=3

Cacnes_tree.show(tree_style=general_ts)



#%% ####################### NOT USED ############################
# # Example of thresholding


# xs = np.arange(0,1,0.01)
# accepted_pi_dist = stats.beta.pdf(xs, 1, 7, loc=0, scale=1)
# rejected_pi_dist = stats.beta.pdf(xs, 7, 10, loc=0, scale=1)

# fig, axs = plt.subplots(2)
# fig.set_size_inches(2.5,3)

# axs[0].plot(xs, accepted_pi_dist, color='k', lw=1, label='Known clade')
# axs[0].axvline(0.35, color='k', lw=2, alpha=0.2)
# axs[0].fill_between(np.arange(0,.36,0.01), 
#                 stats.beta.pdf(np.arange(0,.36,0.01), 1, 7, loc=0, scale=1),
#                 color='k', alpha=0.1)

# axs[1].plot(xs, rejected_pi_dist, color='k', lw=1, label='Novel clade')
# axs[1].axvline(0.35, color='k', lw=2, alpha=0.2)
# axs[1].fill_between(np.arange(0,.36,0.01), 
#                 stats.beta.pdf(np.arange(0,.36,0.01), 7, 10, loc=0, scale=1),
#                 color='k', alpha=0.1)

# axs[1].set_xlabel('$\pi$', **fmt)
# axs[0].set_ylabel('Density', **fmt)
# axs[1].set_ylabel('Density', **fmt)
# axs[0].set_title('Known clade', **fmt)
# axs[1].set_title('Novel clade', **fmt)
# axs[1].tick_params(axis='both', which='major', labelsize=12)
# axs[1].set_yticks([])
# axs[0].tick_params(axis='both', which='major', labelsize=12)
# axs[0].set_yticks([]); axs[0].set_xticks([])
# axs[0].set_xticks([0,.5,1])
# axs[1].set_xticks([0,.5,1])
# axs[0].set_ylim(-.5, 8)
# axs[1].set_ylim(-.5, 5)
# fig.tight_layout(h_pad=0.15)


# def highest_posterior_density_interval(cdf, x, alpha=0.95):
#     # Number of points in the distribution
#     n_points = len(cdf)
#     hpdi_intervals = []

#     # Iterate through each possible starting point
#     for start in range(n_points):
#         for end in range(start, n_points):
#             # Calculate the interval probability
#             interval_prob = cdf[end] - (cdf[start - 1] if start > 0 else 0)
#             # Check if it meets the threshold for HPDI
#             if interval_prob >= alpha:
#                 hpdi_intervals.append((x[start], x[end], interval_prob))

#     # Select the interval with the smallest range
#     hpdi_intervals.sort(key=lambda interval: interval[1] - interval[0])  # Sort by interval length
#     return hpdi_intervals[0][:2] if hpdi_intervals else (None, None)

# accepted_pi_dist /= accepted_pi_dist.sum()
# hpdi = highest_posterior_density_interval(np.cumsum(accepted_pi_dist), xs, alpha=0.95)

# rejected_pi_dist /= rejected_pi_dist.sum()
# hpdi_rejected = highest_posterior_density_interval(np.cumsum(rejected_pi_dist), xs, alpha=0.95)

# axs[0].plot([hpdi[0], hpdi[1]], [-.25,-.25], color='k', lw=3)
# axs[1].plot([hpdi_rejected[0], hpdi_rejected[1]], [-.25,-.25], color='k', lw=3)
                    

# # fig.savefig('fig2/fig2f.pdf', dpi=300, format='pdf', bbox_inches='tight')
