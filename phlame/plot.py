#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:04:12 2023

@author: evanqu
"""

# Phlame plot module

#%%
import numpy as np
import os
import pickle
import gzip
import matplotlib.pyplot as plt

import phlame.helper_functions as helper

#%%

class PlotSample():
    '''
    Plot classify data for a single sample.
    '''

    def __init__(self,
                 path_to_frequencies_file,
                 path_to_data_file,
                 path_to_output_plot,
                 max_pi = 0.35,
                 min_prob = 0.5):

        
        self.path_to_frequencies_file = path_to_frequencies_file
        self.path_to_data_file = path_to_data_file
        self.path_to_output_plot = path_to_output_plot
        self.hfont = {'fontname':'Helvetica',
                            'fontsize':12}
        
        self.max_pi = max_pi
        self.min_prob = min_prob

        if not self.path_to_output_plot.endswith('.pdf'):
            self.path_to_output_plot += '.pdf'

        
    def main(self):

        sample_name = os.path.basename(self.path_to_frequencies_file)

        fig = self.plot_sample(sample_name, 
                               self.path_to_frequencies_file,
                               self.path_to_data_file,
                               max_pi = self.max_pi,
                               min_prob = self.min_prob)
        
        fig.tight_layout()

        fig.savefig(self.path_to_output_plot, format='pdf')

  
    def plot_sample(self, sample_name, 
                    path_to_frequencies_file,
                    path_to_data_file,
                    max_pi, min_prob):
        '''
        Plot all classify information for a single sample.
        '''
        
        # Load in sample 
        sample_frequencies = helper.Frequencies(path_to_frequencies_file)
        data = helper.FrequenciesData(path_to_data_file)
        
        # Pick which clades to model
        model_bool = (np.logical_or.reduce((data.counts_MLE != -1),1))
        # model_bool = np.in1d(sample_frequencies.freqs.index , [36, 83])
        
        counts_tmp = []
        total_tmp = []
        clade_names = []
        total_lambda = []
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

            total_lambda.append( data.total_MLE[i][0] )
                

        # ========================================================================
        #  Plot stuff
        # ========================================================================
        
        nplots = np.sum(model_bool)

        if nplots < 2:
            nplots = 2 # Quick fix
        fig, axs = plt.subplots(nplots,4)
        fig.set_size_inches(12, (nplots*2)+2)
        
        # Iterate through clades
        for c, clade in enumerate(clade_names):
            
            # Plot counts histogram
            axs[c,0].set_title(f"{clade}")
            axs[c,0].hist(counts_tmp[c], 
                        bins=np.arange(0,max(counts_tmp[c])+2,1),
                        color='r', alpha=0.5) # csSNP counts
            axs[c,0].hist(total_tmp[c], 
                        bins=np.arange(0,max(total_tmp[c])+2,1),
                        color='k', alpha=0.5) # Total counts
            axs[c,0].set_xlabel('Counts', **self.hfont)
            axs[c,0].set_ylabel('# of SNVs', **self.hfont)
            axs[c,0].tick_params('both', **{'labelsize':12})

            # Bin MCMC chain
            nchain = len(np.array(data.chain)[model_bool][c]['pi'])

            lambda_chain = np.array(data.chain)[model_bool][c]['a']/np.array(data.chain)[model_bool][c]['b']
            pi_chain = np.array(data.chain)[model_bool][c]['pi']

            pi_bins = np.histogram(pi_chain,
                                    np.arange(0,1.01,0.01),
                                    density=True)
            lambda_bins = np.histogram(lambda_chain, 
                                        bins=100,
                                        density=True)
            freq_bins = np.histogram(lambda_chain/data.total_MLE[model_bool][c][0],
                                     np.arange(0,1.01,0.01),
                                     density=True)
            
            
            
            # Calc prob that chain < max_pi
            prob = np.sum(np.array(data.chain)[model_bool][c]['pi']<max_pi)/nchain
            
            # Calc HPD
            hpd = self.get_hpd(np.array(data.chain)[model_bool][c]['pi'], 0.95)

            # Report the output frequency
            
            if prob > min_prob:
                frequency = sample_frequencies.freqs.loc[clade].values[0]
                freq_color='g'
            else:
                frequency = 0
                freq_color='k'
                
            axs[c,0].text(0.95, 0.95, 
            f"Frequency: {frequency:.2f}", 
            ha='right', va='top',
            transform=axs[c,0].transAxes,**{'fontsize':8, 'color':freq_color})

            # pi posterior
            axs[c,1].plot(pi_bins[1][:-1], 
                        pi_bins[0],  
                        color='g', label='Posterior dist.')
            axs[c,1].text(0.99, 0.8, 
                        f"P($\pi$<{max_pi})={prob:.2f}", 
                        ha='right', va='bottom', 
                        transform=axs[c,1].transAxes,**self.hfont)
            axs[c,1].axvline(max_pi, color='k', ls='--', label='max_pi')
            axs[c,1].set_xlabel('$\pi$',**self.hfont); axs[c,1].set_ylabel('Density',**self.hfont)
            axs[c,1].set_xlim(0,1)
            axs[c,1].plot( hpd, [0.1,0.1], color='k', 
                        alpha=0.5, linewidth=4, label='HPD')
            # axs[c,1].legend()
    
            # lambda posterior
            axs[c,2].plot(lambda_bins[1][:-1],
                            lambda_bins[0], 
                            color='b', label='MCMC')
            axs[c,2].set_xlabel('$\lambda$',**self.hfont); axs[c,2].set_ylabel('Density',**self.hfont)
            # axs[c,2].legend()

            # alpha posterior
            axs[c,3].plot(freq_bins[1][:-1],
                          freq_bins[0], 
                          color='b', label='MCMC')
            axs[c,3].set_xlabel('Estimated frequency',**self.hfont); axs[c,3].set_ylabel('Density',**self.hfont)
            
            # Also plot the prior 
                    
        return fig
    
    @staticmethod
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

# #%% Re-model a specific clade

# nchain = 10000
# nburn = 500
# seed = 1
# max_pi = 0.3
# min_prob = 0.5
# clade_name = 'F'

# def remodel(sample_name, clade_name,
#             path_to_frequencies_file, path_to_data_file,
#             max_pi, min_prob,
#             nchain, nburn, seed):
    
#     sample_frequencies = Frequencies(path_to_frequencies_file)
#     data = FrequenciesData(path_to_data_file)

#     cladeidx = np.where(sample_frequencies.freqs.index == clade_name)[0][0]

#     counts = data.clade_counts[cladeidx][0] + data.clade_counts[cladeidx][1]
#     total_counts = data.clade_counts[cladeidx][2] + data.clade_counts[cladeidx][3]

#     # Fit the model
#     cts_fit = classify.countsCSS_NEW(counts,
#                                     total_counts,
#                                     seed=seed,
#                                     force_alpha=False)
    
#     prob, hpd = cts_fit.fit(max_pi = max_pi,
#                        nchain = nchain,
#                        nburn = nburn)


#     #################################### Plot ####################################
    
#     fig, axs = plt.subplots(1,4)
    
#     # Plot counts histogram
#     axs[0].hist(counts, 
#                 bins=np.arange(0,max(counts)+2,1),
#                 color='r', alpha=0.5) # csSNP counts
#     axs[0].hist(total_counts, 
#                 bins=np.arange(0,max(total_counts)+2,1),
#                 color='k', alpha=0.5) # Total counts
#     axs[0].set_xlabel('Counts', **self.hfont)
#     axs[0].set_ylabel('# of SNPs', **self.hfont)
#     axs[0].tick_params('both', **{'labelsize':12})


#     # Bin MCMC chain
#     nchain_a = nchain - nburn
#     pi_bins = np.histogram(np.array(cts_fit.chain['pi']),
#                             np.arange(0,1.01,0.01),
#                             density=True)
#     lambda_bins = np.histogram(np.array(cts_fit.chain['a'])/np.array(cts_fit.chain['b']), 
#                                 bins=100,
#                                 density=True)
#     alpha_bins = np.histogram(np.array(cts_fit.chain['a']),
#                                 bins=100,
#                                 density=True)

#     # pi posterior
#     axs[1].plot(pi_bins[1][:-1], 
#                     pi_bins[0],  
#                     color='g', label='Posterior dist.')
#     axs[1].text(0.99, 0.8, 
#                 f"P(pi<{max_pi})={prob:.2f}", 
#                 ha='right', va='bottom', 
#                 transform=axs[1].transAxes,**self.hfont)
#     axs[1].axvline(max_pi, color='g', ls='--', label='max_pi')
#     # axs[c,1].axvline(data.counts_MAP[model_bool][c][2], color='r', label='MAP')
#     axs[1].set_xlabel('Pi',**self.hfont); axs[1].set_ylabel('Probability',**self.hfont)
#     axs[1].set_xlim(0,1)
#     axs[1].plot( hpd, [0.1,0.1], color='k', 
#                 alpha=0.5, linewidth=4, label='HPD')

#     # lambda posterior
#     axs[2].plot(lambda_bins[1][:-1],
#                     lambda_bins[0], 
#                     color='b', label='MCMC')
#     # axs[c,2].axvline(data.counts_MLE[model_bool][c][0], color='k', label='MLE')
#     # axs[c,2].axvline(data.counts_MAP[model_bool][c][0], color='r', label='Posterior mean')
#     axs[2].set_xlabel('Lambda',**self.hfont); axs[2].set_ylabel('Probability',**self.hfont)
#     axs[2].legend()

#     # alpha posterior
#     axs[3].plot(alpha_bins[1][:-1],
#                     alpha_bins[0], 
#                     color='b', label='MCMC')
#     # axs[c,3].axvline(data.counts_MLE[model_bool][c][1], color='k', label='MLE')
#     # axs[c,3].axvline(data.counts_MAP[model_bool][c][1], color='r', label='Posterior mean')
#     axs[3].set_xlabel('Alpha',**self.hfont); axs[3].set_ylabel('Probability',**self.hfont)
#     axs[3].legend()

#     ##############################################################################


#     ################################### Plot 2 ###################################
    
#     fig, axs = plt.subplots(3)
#     fig.set_size_inches(12, 8)
#     # Bin MCMC chain
#     pi_chain = np.array(cts_fit.chain['pi'])
#     lambda_chain = np.array(cts_fit.chain['a'])/np.array(cts_fit.chain['b'])
#     alpha_chain = np.array(cts_fit.chain['a'])
#     nchain = len(pi_chain)

#     # Pi chain
#     axs[0].plot(np.arange(0,nchain), 
#                 pi_chain,  
#                 color='g', label='Posterior dist.')
#     axs[0].set_xlabel('Iteration',**self.hfont); axs[0].set_ylabel('Pi',**self.hfont)
#     axs[0].set_ylim(0,1)
#     axs[0].set_title(f"Pi chain")

#     # lambda posterior
#     axs[1].plot(np.arange(0,nchain),
#                 lambda_chain, 
#                 color='b')
#     axs[1].set_xlabel('Iteration',**self.hfont); axs[1].set_ylabel('Lambda',**self.hfont)
#     axs[1].set_title(f"Lambda chain")


#     # alpha posterior
#     axs[2].plot(np.arange(0,nchain),
#                 alpha_chain, 
#                 color='r')
#     axs[2].set_xlabel('Iteration',**self.hfont); axs[2].set_ylabel('Alpha',**self.hfont)
#     axs[2].set_title(f"Alpha chain")

#     fig.tight_layout()
#     ##############################################################################

#     return fig

# # fig = remodel(sample, '52',
# #             path_to_frequencies_file, path_to_data_file,
# #             max_pi = 0.30,
# #             min_prob = 0.5,
# #             nchain = 10000,
# #             nburn = 500,
# #             seed = 12345)



