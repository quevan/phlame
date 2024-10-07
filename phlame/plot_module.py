#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:04:12 2023

@author: evanqu
"""

# Phlame plot module

#%%
import numpy as np
import pandas as pd
import math
import pickle
import gzip
import matplotlib.pyplot as plt
import ete3
import phlame.helper_functions as helper
import scipy.stats as stats

import glob

import phlame.classify_module as classify

#%% Define variables
import os

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures/fig3')

path_to_counts_dir = '5-counts'
path_to_frequencies_dir = 'PHLAME/Sepi'

refgenome='SepidermidisATCC12228'
# refgenome='Gleopoldii_6420B'
# refgenome='Gpiotii_ASM339758' 
# refgenome='Gvaginalis_FDAARGOS_568'

sample='05X_Sepi_B'

#%% Plooting
path_to_frequencies_file=f'{path_to_frequencies_dir}/{sample}_ref_{refgenome}_frequencies.csv'
path_to_data_file=f'{path_to_frequencies_dir}/{sample}_ref_{refgenome}_fitinfo.data'

fig = plot_sample_hist(sample, 
                       path_to_frequencies_file,
                       path_to_data_file,
                       max_pi = 0.35,
                       min_prob = 0.5)

fig.tight_layout()
#
fig.savefig(f"/Users/evanqu/Desktop/{sample}.pdf", format='pdf')


#%% Re-model a specific clade

nchain = 10000
nburn = 500
seed = 1
max_pi = 0.3
min_prob = 0.5
clade_name = 'F'

def remodel(sample_name, clade_name,
            path_to_frequencies_file, path_to_data_file,
            max_pi, min_prob,
            nchain, nburn, seed):
    
    sample_frequencies = Frequencies(path_to_frequencies_file)
    data = FrequenciesData(path_to_data_file)

    cladeidx = np.where(sample_frequencies.freqs.index == clade_name)[0][0]

    counts = data.clade_counts[cladeidx][0] + data.clade_counts[cladeidx][1]
    total_counts = data.clade_counts[cladeidx][2] + data.clade_counts[cladeidx][3]

    # Fit the model
    cts_fit = classify.countsCSS_NEW(counts,
                                    total_counts,
                                    seed=seed,
                                    force_alpha=False)
    
    prob, hpd = cts_fit.fit(max_pi = max_pi,
                       nchain = nchain,
                       nburn = nburn)


    #################################### Plot ####################################
    
    fig, axs = plt.subplots(1,4)
    
    # Plot counts histogram
    axs[0].hist(counts, 
                bins=np.arange(0,max(counts)+2,1),
                color='r', alpha=0.5) # csSNP counts
    axs[0].hist(total_counts, 
                bins=np.arange(0,max(total_counts)+2,1),
                color='k', alpha=0.5) # Total counts
    axs[0].set_xlabel('Counts', **hfont)
    axs[0].set_ylabel('# of SNPs', **hfont)
    axs[0].tick_params('both', **{'labelsize':12})


    # Bin MCMC chain
    nchain_a = nchain - nburn
    pi_bins = np.histogram(np.array(cts_fit.chain['pi']),
                            np.arange(0,1.01,0.01),
                            density=True)
    lambda_bins = np.histogram(np.array(cts_fit.chain['a'])/np.array(cts_fit.chain['b']), 
                                bins=100,
                                density=True)
    alpha_bins = np.histogram(np.array(cts_fit.chain['a']),
                                bins=100,
                                density=True)

    # pi posterior
    axs[1].plot(pi_bins[1][:-1], 
                    pi_bins[0],  
                    color='g', label='Posterior dist.')
    axs[1].text(0.99, 0.8, 
                f"P(pi<{max_pi})={prob:.2f}", 
                ha='right', va='bottom', 
                transform=axs[1].transAxes,**hfont)
    axs[1].axvline(max_pi, color='g', ls='--', label='max_pi')
    # axs[c,1].axvline(data.counts_MAP[model_bool][c][2], color='r', label='MAP')
    axs[1].set_xlabel('Pi',**hfont); axs[1].set_ylabel('Probability',**hfont)
    axs[1].set_xlim(0,1)
    axs[1].plot( hpd, [0.1,0.1], color='k', 
                alpha=0.5, linewidth=4, label='HPD')

    # lambda posterior
    axs[2].plot(lambda_bins[1][:-1],
                    lambda_bins[0], 
                    color='b', label='MCMC')
    # axs[c,2].axvline(data.counts_MLE[model_bool][c][0], color='k', label='MLE')
    # axs[c,2].axvline(data.counts_MAP[model_bool][c][0], color='r', label='Posterior mean')
    axs[2].set_xlabel('Lambda',**hfont); axs[2].set_ylabel('Probability',**hfont)
    axs[2].legend()

    # alpha posterior
    axs[3].plot(alpha_bins[1][:-1],
                    alpha_bins[0], 
                    color='b', label='MCMC')
    # axs[c,3].axvline(data.counts_MLE[model_bool][c][1], color='k', label='MLE')
    # axs[c,3].axvline(data.counts_MAP[model_bool][c][1], color='r', label='Posterior mean')
    axs[3].set_xlabel('Alpha',**hfont); axs[3].set_ylabel('Probability',**hfont)
    axs[3].legend()

    ##############################################################################


    ################################### Plot 2 ###################################
    
    fig, axs = plt.subplots(3)
    fig.set_size_inches(12, 8)
    # Bin MCMC chain
    pi_chain = np.array(cts_fit.chain['pi'])
    lambda_chain = np.array(cts_fit.chain['a'])/np.array(cts_fit.chain['b'])
    alpha_chain = np.array(cts_fit.chain['a'])
    nchain = len(pi_chain)

    # Pi chain
    axs[0].plot(np.arange(0,nchain), 
                pi_chain,  
                color='g', label='Posterior dist.')
    axs[0].set_xlabel('Iteration',**hfont); axs[0].set_ylabel('Pi',**hfont)
    axs[0].set_ylim(0,1)
    axs[0].set_title(f"Pi chain")

    # lambda posterior
    axs[1].plot(np.arange(0,nchain),
                lambda_chain, 
                color='b')
    axs[1].set_xlabel('Iteration',**hfont); axs[1].set_ylabel('Lambda',**hfont)
    axs[1].set_title(f"Lambda chain")


    # alpha posterior
    axs[2].plot(np.arange(0,nchain),
                alpha_chain, 
                color='r')
    axs[2].set_xlabel('Iteration',**hfont); axs[2].set_ylabel('Alpha',**hfont)
    axs[2].set_title(f"Alpha chain")

    fig.tight_layout()
    ##############################################################################

    return fig

# fig = remodel(sample, '52',
#             path_to_frequencies_file, path_to_data_file,
#             max_pi = 0.30,
#             min_prob = 0.5,
#             nchain = 10000,
#             nburn = 500,
#             seed = 12345)


#%% Fxns

hfont = {'fontname':'Helvetica',
         'fontsize':12}

class Frequencies():
    '''
    Holds clade frequency information from a given sample
    '''
    
    def __init__(self, path_to_frequencies_file):
        
        self.freqs = pd.read_csv(path_to_frequencies_file,
                                       index_col=0)
                
class FrequenciesData():
    '''
    Holds csSNP counts and modeling information from a given sample.
    '''

    def __init__(self, path_to_data_file):
        
        with gzip.open(path_to_data_file, 'rb') as f:
            
            data_dct, fit_info_dct = pickle.load(f)
            
            # clade_counts structured as follows
            
            self.clade_counts = data_dct['clade_counts']
            self.clade_counts_pos = data_dct['clade_counts_pos']
            
            self.counts_MLE = fit_info_dct['counts_MLE']
            self.total_MLE = fit_info_dct['total_MLE']
            self.counts_MAP = fit_info_dct['counts_MAP']
            self.chain = fit_info_dct['chain']
            self.prob = fit_info_dct['prob']
    
                       
def plot_sample_hist(sample_name, 
                     path_to_frequencies_file, path_to_data_file,
                     max_pi, min_prob):
    '''
    Plot histogram and fit information for a single sample
    '''
    
    # Load in sample 
    sample_frequencies = Frequencies(path_to_frequencies_file)
    data = FrequenciesData(path_to_data_file)
    
    # Pick which clades to model
    model_bool = (np.logical_or.reduce((data.counts_MLE != -1),1))
    # model_bool = np.in1d(sample_frequencies.freqs.index , [36, 83])
    
    counts_tmp = []
    total_tmp = []
    clade_names = []
    
    # ========================================================================
    #  Organize information by clade
    # ========================================================================
    
    for i in range(len(data.clade_counts)):
        
        # Ignore clades with not enough counts to be modeled
        if not model_bool[i]:
            continue
        
        counts_tmp.append( data.clade_counts[i][0] + 
                            data.clade_counts[i][1] )
        
        total_tmp.append( data.clade_counts[i][2] + 
                            data.clade_counts[i][3] )
        
        clade_names.append( sample_frequencies.freqs.index[i] )
            

    # ========================================================================
    #  Plot stuff
    # ========================================================================
    
    nplots = np.sum(model_bool)

    if nplots < 2:
        nplots = 2 # Quick fix
    fig, axs = plt.subplots(nplots,4)
    fig.set_size_inches(12, nplots*2)

    plt.suptitle(f"{sample_name} called at max_pi<{max_pi};min_prob>{min_prob}")
    
    # Iterate through clades
    for c, clade in enumerate(clade_names):
        
        # Plot counts histogram
        axs[c,0].hist(counts_tmp[c], 
                      bins=np.arange(0,max(counts_tmp[c])+2,1),
                      color='r', alpha=0.5) # csSNP counts
        axs[c,0].hist(total_tmp[c], 
                      bins=np.arange(0,max(total_tmp[c])+2,1),
                      color='k', alpha=0.5) # Total counts
        axs[c,0].set_xlabel('Counts', **hfont)
        axs[c,0].set_ylabel('# of SNPs', **hfont)
        axs[c,0].tick_params('both', **{'labelsize':12})

        # counts_histo = np.histogram(counts_tmp[c],
        #                      bins=np.arange(0,max(total_tmp[c])+2,1))
        # total_histo = np.histogram(total_tmp[c],
        #              bins=np.arange(0,max(total_tmp[c])+2,1))
        # axs[c,0].set_ylim(0,np.max(counts_histo[0])+10)
        # axs[c,0].set_xlim(0,np.max(total_tmp[c])+2)
        # axs[c,0].set_xscale('log')

        # Bin MCMC chain
        nchain = len(np.array(data.chain)[model_bool][c]['pi'])
        pi_bins = np.histogram(np.array(data.chain)[model_bool][c]['pi'],
                        np.arange(0,1.01,0.01),
                        density=True)
        lambda_bins = np.histogram(np.array(data.chain)[model_bool][c]['a']/np.array(data.chain)[model_bool][c]['b'], 
                      bins=100,
                      density=True)
        alpha_bins = np.histogram(np.array(data.chain)[model_bool][c]['a'],
                      bins=100,
                      density=True)
        
        # Calc prob that chain < max_pi
        prob = np.sum(np.array(data.chain)[model_bool][c]['pi']<max_pi)/nchain
        
        # Calc HPD
        hpd = get_hpd(np.array(data.chain)[model_bool][c]['pi'], 0.95)

        # Report the output frequency
        
        if prob > min_prob:
            frequency = sample_frequencies.freqs.loc[clade].values[0]
            freq_color='g'
        else:
            frequency = 0
            freq_color='k'
            
        axs[c,0].text(0.95, 0.95, 
        f"{clade} frequency: {frequency:.2f}", 
        ha='right', va='top',
        transform=axs[c,0].transAxes,**{'fontsize':8, 'color':freq_color})

        # pi posterior
        axs[c,1].plot(pi_bins[1][:-1], 
                      pi_bins[0],  
                      color='g', label='Posterior dist.')
        axs[c,1].text(0.99, 0.8, 
                    f"P(pi<{max_pi})={prob:.2f}", 
                    ha='right', va='bottom', 
                    transform=axs[c,1].transAxes,**hfont)
        axs[c,1].axvline(max_pi, color='g', ls='--', label='max_pi')
        # axs[c,1].axvline(data.counts_MAP[model_bool][c][2], color='r', label='MAP')
        axs[c,1].set_xlabel('Pi',**hfont); axs[c,1].set_ylabel('Probability',**hfont)
        axs[c,1].set_xlim(0,1)
        axs[c,1].plot( hpd, [0.1,0.1], color='k', 
                      alpha=0.5, linewidth=4, label='HPD')
        # axs[c,1].legend()
 
        # lambda posterior
        axs[c,2].plot(lambda_bins[1][:-1],
                      lambda_bins[0], 
                      color='b', label='MCMC')
        # axs[c,2].axvline(data.counts_MLE[model_bool][c][0], color='k', label='MLE')
        # axs[c,2].axvline(data.counts_MAP[model_bool][c][0], color='r', label='Posterior mean')
        axs[c,2].set_xlabel('Lambda',**hfont); axs[c,2].set_ylabel('Probability',**hfont)
        axs[c,2].legend()

        # alpha posterior
        axs[c,3].plot(alpha_bins[1][:-1],
                      alpha_bins[0], 
                      color='b', label='MCMC')
        # axs[c,3].axvline(data.counts_MLE[model_bool][c][1], color='k', label='MLE')
        # axs[c,3].axvline(data.counts_MAP[model_bool][c][1], color='r', label='Posterior mean')
        axs[c,3].set_xlabel('Alpha',**hfont); axs[c,3].set_ylabel('Probability',**hfont)
        
        # Also plot the prior 
        gaussian_xs = np.arange(0,6,0.01)
        gaussian_ys = stats.lognorm.pdf(gaussian_xs, 
                                        s=0.1,
                                        loc = (np.var(total_tmp[c])-np.mean(total_tmp[c]))/(np.mean(total_tmp[c])**2))
        axs[c,3].plot(gaussian_xs, gaussian_ys, color='g', label='Prior')
        axs[c,3].set_xlim(0,6)
        axs[c,3].set_ylim(0,5)

        axs[c,3].legend()

        
    return fig
        
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


def csSNP_vs_branchlen(tree, csSNPs, clade_names, label):
    
    
    
    names_to_exclude = np.array(['C.1','C.2'])
    exclude_bool = ~np.in1d(clade_names, names_to_exclude)
   
    num_csSNPs = np.zeros(len(clade_names[exclude_bool]))
    branch_lens = np.zeros(len(clade_names[exclude_bool]))

    for i, name in enumerate(clade_names[exclude_bool]):
        
        num_csSNPs[i] = np.count_nonzero(csSNPs[:,exclude_bool][:,i])
        branch_lens[i] = tree.search_nodes(name=name)[0].dist
    
    # branch_lens_sort = np.sort(branch_lens)
    # 
    
    plt.scatter(branch_lens, num_csSNPs, label=label, alpha=.5)
    plt.plot([0,np.max(branch_lens)],[0,np.max(branch_lens)/10],color='k')
    # plt.yscale('log'); plt.xscale('log')
    plt.xlabel('Branch length (# SNPs)')
    plt.ylabel('# csSNPs')
    plt.xlim(0,np.max(branch_lens)+1000)
    plt.ylim(0,np.max(num_csSNPs)+100)

    # fig.show()
    
    return

def nisos_vs_dist_from_tips(tree, 
                            clades, clade_names, 
                            label):
    '''
    Iteratively include clades from tips and plot # of isolates included as a
    fxn of dist from tips.
    '''   
    
    dist_from_tips = np.zeros(len(clade_names))
    
    for i, name in enumerate(clade_names):

        node = tree.search_nodes(name=name)[0]
        
        # Calc average distance from tips
        dist=[]
        for node_leaf in node.iter_leaves():
            dist.append(node.get_distance(node_leaf))
        
        dist_from_tips[i] = np.mean(dist)
        
        
    dist_from_tips_sort = np.sort(dist_from_tips)
    clade_names_sort = clade_names[np.argsort(dist_from_tips)]
    
    isos = []
    nisos = []
    
    for clade in clade_names_sort:
        
        isos = np.unique(np.append(isos,clades[clade]))
        nisos.append( len(isos) )
        
    plt.plot(dist_from_tips_sort, 
              np.array(nisos)/len(tree), label=label)
    # plt.plot(np.arange(0, len(clade_names))/len(clade_names), 
    #           np.array(nisos)/len(tree), label=label)

    plt.ylabel('Percent isolates included')
    plt.xlabel('Average distance from tips (# SNPs)')
    # plt.xlabel('Nth percentile distance from tips')
        
    return


def callable_dist_from_tips(tree, 
                            clade_names,
                            cand_clade_names,
                            lbl):
    '''
    Plot hist of all possible callable clades / clades > 10 csSNPs as a fxn of
    dist from tips
    '''
    
    dist_from_tips = np.zeros(len(cand_clade_names))
    
    for i, name in enumerate(cand_clade_names):

        node = tree.search_nodes(name=name)[0]
        
        # Calc average distance from tips
        dist=[]
        for node_leaf in node.iter_leaves():
            dist.append(node.get_distance(node_leaf))
        
        dist_from_tips[i] = np.mean(dist)
    
    cssnps_bool = np.in1d(cand_clade_names,clade_names)
    
    plt.hist(dist_from_tips, 
             np.logspace(np.log10(1),np.log10(np.max(dist_from_tips)), 40),
             alpha=0.5, color='k', label='candidate clades')
    plt.hist(dist_from_tips[cssnps_bool], 
             np.logspace(np.log10(1),np.log10(np.max(dist_from_tips)), 40),
             alpha=0.5, color='r', label='clades with csSNPs')
    plt.xscale('log')
    plt.xlabel('Average distance from tips (# SNPs)')
    plt.ylabel('Number of clades')
    plt.title(f"{lbl}:  {len(clade_names)/len(cand_clade_names):.2f} candidate clades had csSNPs")
    
    return

def plot_classifier(path_to_classifier):
    
    with gzip.open(path_to_classifier) as f:
        
        cssnp_dct = pickle.load(f)
        
    clade_names = cssnp_dct['clade_names']
    cssnps = cssnp_dct['cssnps']
    
    cssnp_ct = np.zeros(len(clade_names))
    for c in range(len(clade_names)):
        cssnp_ct[c] = (np.count_nonzero(cssnps[:,c]))
        
    # Order decreasing
    cssnp_ct_sort = np.sort(cssnp_ct)
    clade_names_sort = clade_names[np.argsort(cssnp_ct)]
    
    # =========================================================================
    #  Plot
    # =========================================================================
    fig = plt.figure()
    plt.barh(np.arange(0,len(clade_names),1),
             width=cssnp_ct_sort,
             height=0.8, alpha=0.8,
             color='b')
    plt.xlabel('Number of csSNPs'); plt.ylabel('Clade')
    plt.xscale('log')
    plt.yticks(np.arange(0,len(clade_names),1), clade_names_sort)
    plt.xlim(1,max(cssnp_ct_sort)+100)
    # Get number of csssnps to show above each
    for i, v in enumerate(cssnp_ct_sort):
        plt.text(v + 3, i + .25, str(v),
                color = 'k')

    fig.show()
    plt.tight_layout()
    
    return fig

# %%
