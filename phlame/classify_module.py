#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov  7 13:39:50 2022

@author: evanqu
"""

#%%
import os
import numpy as np
import pandas as pd
import pickle
import gzip
import warnings
from scipy import stats
from scipy.optimize import minimize 
import subprocess
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import digamma, gammaln, loggamma, polygamma
import zeus

#%%

# Wishlist:
# Make runtimeerror with loglikelihood in counts_css fail out after 3 times or so

# os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_02_Vaginal_gardnerella')

# path_to_cts_file='5-counts_benchmark/Gardnerella_benchmark_r19_ref_Gpiotii_ASM339758.counts.pickle.gz'

# path_to_classifier='/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_02_Gardnerella_classifiers/Gardnerella_classifiers/Gleopoldii_HKY85.classifier'

# # level_input='Bvulgatus_dorei_phylogroups.txt'

# path_to_output_frequencies='32334_test_frequencies.csv'
# path_to_output_data='32334_test_fitinfo.data'

# results = Classify(path_to_cts_file=path_to_cts_file,
#                     path_to_classifier=path_to_classifier,
#                     path_to_output_frequencies=path_to_output_frequencies,
#                     path_to_output_data=path_to_output_data)

# results.main()

# # ###############################################################################
# Gardnerella_benchmark_r19

# path_to_data_file = '6-frequencies_benchmark/Gardnerella_benchmark_r19_ref_Gleopoldii_6420B_fitinfo.data'
# path_to_classifier='/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2024_02_Gardnerella_classifiers/Gardnerella_classifiers/Gleopoldii_HKY85.classifier'


# data = FrequenciesData(path_to_data_file)
# cfr = PhlameClassifier.read_file(path_to_classifier)

# # The index of C.2.2.1 is 12
# clade_counts = data.clade_counts[12]

# cts2model = clade_counts[0] + clade_counts[1]
# total2model = clade_counts[2] + clade_counts[3]

# # To have a frequency, a call must have > min_SNP pos with nonzero counts in either fwd OR rev reads
# assert np.count_nonzero(cts2model) >= 10
                        
# cts_fit = countsCSS_NEW(cts2model,
#                         total2model,
#                         seed=12345)

# prob = cts_fit.fit(max_pi = 0.35,
#                     nchain = 10000,
#                     nburn = 500)

# print(f"MLE fit: cts lambda={cts_fit.counts_mle[0]:.2f} cts pi={cts_fit.counts_mle[1]:.2f}")
# print(f"total lambda={cts_fit.total_mle[0]:.2f} total pi={cts_fit.total_mle[1]:.2f}")


# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(3)
# MAP_values = []
# for i, (var, chain) in enumerate(cts_fit.chain.items()):
    
#     counts, bins = np.histogram(chain, bins=100, density=True)
#     MAP = calc_MAP(chain)

#     axs[i].plot(bins[:-1], counts)
#     axs[i].axvline(MAP, color='r', linestyle='--')
#     axs[i].set_title(var)

#     MAP_values.append(MAP)
 
# lambda_chain = cts_fit.chain['a']/cts_fit.chain['b']
# lambda_counts, lambda_bins = np.histogram(lambda_chain, bins=100, density=True)
# plt.plot(lambda_bins[:-1], lambda_counts)
# plt.axvline(MAP_values[1]/MAP_values[2], color='r', linestyle='--')

# def calc_MAP(chain, bins=False):

#     if not bins:
#         bins = 100

#     counts, bins = np.histogram(chain, bins=bins)
#     max_idx = np.argmax(counts)
    
#     return bins[max_idx]

#%%

# Limit for exponentials to avoid overflow
EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0

    
class Frequencies():
    '''
    Holds clade frequency information from a given sample.
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


class CountsMat():
    '''
    Hold data and methods for a counts matrix
    '''
    def __init__(self, path_to_cts_file):
        
        with gzip.open(path_to_cts_file,'rb') as f:
            counts, pos = pickle.load(f)
                
        self.counts = counts
        self.pos = pos
        
class Classify:
    '''
    Main controller of the classify step.
    
    Args:
    path_to_cts_file (str): Path to PhLAMe counts file.\n
    
    path_to_classifier (str): Path to PhLAMe classifier file.
    level_input (str): Path to file defining the phylogenetic level.
    to type strains at. This file can either list clade names as 
    they appear in the PhLAMe classifier OR custom group genomes 
    into clades in a 2 column .tsv {genomeID, cladeID}.\n
    
    path_to_output_frequencies (str): Path to output frequencies file.\n
    
    path_to_output_data (str, optional): If True, outputs a data file 
    with counts and modeling information at the defined level.
    Defaults to False.\n
    
    max_perc_diff (TYPE, optional): Maximum . Defaults to 0.3.\n
    
    max_snp_diff (TYPE, optional): If True, . Defaults to False.\n
    
    min_snps (int, optional): Minimum number of csSNPs with >0 counts
    to call a lineage as present. Defaults to 10.\n
    
    '''
    
    def __init__(self, path_to_cts_file, path_to_classifier,
                 path_to_output_frequencies,
                 level_input=False,
                 path_to_output_data=False,
                 min_snps=10, max_pi=0.3, min_prob=0.5, min_hpd=0.1,
                 nchain=10000, perc_burn=0.1, seed=False):

        self.__input_cts_file = path_to_cts_file
        self.__classifier_file = path_to_classifier
        self.__levels_input = level_input
        self.__output_freqs_file = path_to_output_frequencies
        self.__output_data_file = path_to_output_data

        self.max_pi = max_pi
        self.min_snps = min_snps
        self.min_prob = min_prob
        self.min_hpd = min_hpd

        self.nchain = nchain
        self.perc_burn = perc_burn
        self.seed = seed
                
    def main(self):
        
        # =====================================================================
        #  Load Data
        # =====================================================================
        print("Reading in file(s)...")
        
        self.countsmat = CountsMat(self.__input_cts_file)
        
        self.classifier = PhlameClassifier.read_file(self.__classifier_file)
        
        if self.__levels_input:
            self.mylevel = PhyloLevel(self.__levels_input, 
                                      self.classifier.clades, 
                                      self.classifier.clade_names)
            
            # Grab just information for the specific level
            self.level_cfr = self.classifier.grab_level(self.mylevel)
        
        else:
            self.level_cfr = self.classifier
        # =====================================================================
        #  Sort through counts mat to grab just the relevant positions
        # =====================================================================
        print("Sorting counts information...")
        
        self.index_counts()
        
        self.get_allele_counts()

        self.get_counts_alpha()

        # =====================================================================
        #  Calculate & Save frequencies
        # =====================================================================
        print("Modeling counts...")
        
        self.calc_frequencies()
            
        self.save_frequencies()

    def index_counts(self):
        '''
        Make structures to index through counts array.
        '''
        
        # Informative positions for this level
        # Note this will be > true pos because of pos with multiple alleles
        informative_bool = np.nonzero(self.level_cfr.csSNPs)[0]
        self.informative_pos = self.level_cfr.csSNP_pos[informative_bool]
        
        # Corresponding index on counts mat
        self.informative_pos_idx = np.arange(0,len(self.countsmat.pos))\
            [np.in1d(self.countsmat.pos,self.informative_pos)]\
                [informative_bool]

        # informative_pos_idx = np.arange(0,len(pos))[np.sum(csSNPs,1)>0]
        counts_CSS_bool = np.in1d(self.countsmat.pos,self.informative_pos)

        # Check that all positions are accounted for
        if np.sum(counts_CSS_bool) != len(np.unique(self.informative_pos)):
            raise Exception('Counts and classifier positions do not match. ',
                            'Check that reference genomes are same.')
    
    def get_allele_counts(self):
        
        np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 warnings
        
        counts = self.countsmat.counts
        counts_idx = self.informative_pos_idx
        alleles = self.level_cfr.alleles
        
        #Initialize data structures
        cts_f = np.zeros(len(counts_idx))
        cts_r = np.zeros(len(counts_idx))
        tot_f = np.zeros(len(counts_idx))
        tot_r = np.zeros(len(counts_idx))
        
        for i, p in enumerate(self.informative_pos):
            
            # -1 +3 is to convert 01234 NATCG to index 
            cts_f[i] = counts[counts_idx[i],
                              int(alleles[i]-1)]
            
            cts_r[i] = counts[counts_idx[i],
                              int(alleles[i]+3)]
            
            tot_f[i] = np.sum(counts[counts_idx[i],:4])
            tot_r[i] = np.sum(counts[counts_idx[i],4:])
            
        self.allele_counts = tuple([cts_f,cts_r,tot_f,tot_r])

    def get_counts_alpha(self):
        '''
        Get overdispersion metric (alpha) across all informative positions.
        '''
        
        counts = self.countsmat.counts
        counts_sum = np.sum(counts, axis=1)

        self.measured_alpha = np.mean(counts_sum)**2/max(1e-6,np.var(counts_sum)-np.mean(counts_sum))
    
    def calc_frequencies(self):

        clade_names = self.level_cfr.clade_names
        nclades = len(clade_names)
        
        frequencies = np.zeros(nclades)
    
        save_cts = []
        save_cts_pos = []
        save_chain = []
        save_hpd = []
        save_cts_map = []

        save_prob=np.full(nclades,-1,dtype=np.float64)
        save_cts_mle=np.full((nclades,2),-1,dtype=np.float64)
        save_total_mle=np.full((nclades,2),-1,dtype=np.float64)

        save_flag_highalpha = np.full(nclades,False,dtype=bool)
        save_flag_map_mle_diff = np.full(nclades,False,dtype=bool)

        # Reshape data to group by clade
        for c in range(len(clade_names)):
            
            byclade_cts = self.reshape_byclade(self.allele_counts,
                                               self.level_cfr.allele_cidx, c)
            
            byclade_cts_pos = self.informative_pos[np.where(self.level_cfr.allele_cidx == c)]
    
            cts2model = byclade_cts[0] + byclade_cts[1]
            total2model = byclade_cts[2] + byclade_cts[3]


            # To model, a clade must have a nonzero_zero ratio > 0.03 
            # and at least 10 SNPs with nonzero counts
            if (np.count_nonzero(cts2model) >= self.min_snps):

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_nonzero_zero = np.true_divide(np.count_nonzero(cts2model>0),
                                                        np.count_nonzero(cts2model==0))
                    if ratio_nonzero_zero == np.inf:
                        ratio_nonzero_zero = 1.0
                    
                                
                print(f"Fit results for clade: {clade_names[c]}")
                
                cts_fit = countsCSS_NEW(cts2model,
                                        total2model,
                                        seed=self.seed)
                                                    
                prob, hpd = cts_fit.fit(max_pi = self.max_pi,
                                        nchain = self.nchain,
                                        nburn = int(self.perc_burn*self.nchain))

                print(f"MLE fit: cts lambda={cts_fit.counts_mle[0]:.2f} cts pi={cts_fit.counts_mle[1]:.2f}")
                print(f"total lambda={cts_fit.total_mle[0]:.2f} total pi={cts_fit.total_mle[1]:.2f}")

                print(f"HPD interval: {hpd[0]:.2f}-{hpd[1]:.2f}")

                # Only count clades if pass probability and HPD thresholds
                if (prob > self.min_prob) & (hpd[0] <= self.min_hpd) & (ratio_nonzero_zero > 0.03):

                    frequencies[c] = ((cts_fit.counts_MAP['a']/cts_fit.counts_MAP['b'])/
                                      cts_fit.total_mle[0])
                    
                    # Flag results with issues
                    if abs(frequencies[c] - cts_fit.counts_mle[0]/cts_fit.total_mle[0]) > 0.2:
                        save_flag_map_mle_diff[c] = True

                # if nonzero_zero ratio is > 0.1, accept call
                # elif (ratio_nonzero_zero > 0.1):
                #     frequencies[c] = ((cts_fit.counts_MAP['a']/cts_fit.counts_MAP['b'])/
                #                       cts_fit.total_mle[0])

                if cts_fit.measured_alpha > 100000:
                    save_flag_highalpha[c] = True

                # Save fit information
                save_chain.append(cts_fit.chain)
                save_hpd.append(cts_fit.hpd)
                save_cts_map.append(cts_fit.counts_MAP)
                save_cts_mle[c] = cts_fit.counts_mle
                save_total_mle[c] = cts_fit.total_mle
                save_prob[c] = prob

            # Otherwise is zero
            else:
                print(f"Clade: {clade_names[c]} ",
                      "does not have not enough SNPs to model")
                save_chain.append({})
                save_hpd.append(np.array((-1,-1)))
                save_cts_map.append({})

            # Save counts data
            save_cts.append( byclade_cts )
            save_cts_pos.append( byclade_cts_pos )
            
        frequencies_df = pd.DataFrame({'Relative abundance':frequencies,
                                       'Estimated divergence':save_hpd,
                                       'Confidence score':save_prob},
                                      index=self.level_cfr.clade_names)

        self.frequencies = frequencies_df
        self.data = {'clade_counts':save_cts,
                     'clade_counts_pos':save_cts_pos}
        self.fit_info = {'counts_MLE': save_cts_mle,
                         'total_MLE':save_total_mle,
                         'counts_MAP':save_cts_map,
                         'chain':save_chain,
                         'prob':save_prob,
                         'hpd':save_hpd,
                         'flag_highalpha':save_flag_highalpha,
                         'flag_map_mle_diff':save_flag_map_mle_diff}
    
    def save_frequencies(self):
        '''
        Write frequencies and data to files.
        '''        
        
        # Save frequencies
        self.frequencies.to_csv(self.__output_freqs_file, sep=',')
        
        # Save data and fit info
        if self.__output_data_file:
            
            with gzip.open(self.__output_data_file,'wb') as f:
                
                pickle.dump([self.data, self.fit_info],f)

    @staticmethod
    def reshape_byclade(allele_counts, clade_idxs, i):
        '''
        Grab only allele counts belonging to a certain clade (i)
        '''
        
        clade_cts_f = allele_counts[0][np.where(clade_idxs==i)]
        clade_cts_r = allele_counts[1][np.where(clade_idxs==i)]
        clade_tot_f = allele_counts[2][np.where(clade_idxs==i)]
        clade_tot_r = allele_counts[3][np.where(clade_idxs==i)]

        
        return tuple([clade_cts_f,
                      clade_cts_r,
                      clade_tot_f,
                      clade_tot_r])

        
class PhlameClassifier():
    '''
    Holds data and methods for a single Phlame Classifier object
    '''
    def __init__(self,
                 csSNPs, csSNP_pos,
                 clades, clade_names):

        self.csSNPs = csSNPs
        self.csSNP_pos = csSNP_pos
        self.clades = clades
        self.clade_names = clade_names
        
        # Get allele information
        self.get_alleles()
    
    def read_file(path_to_classifier_file):

        with gzip.open(path_to_classifier_file, 'rb') as f:
            cssnp_dct = pickle.load(f)
            
            csSNPs = cssnp_dct['cssnps']
            csSNP_pos = cssnp_dct['cssnp_pos']
            clades = cssnp_dct['clades']
            clade_names = cssnp_dct['clade_names']
            
        return PhlameClassifier(csSNPs, csSNP_pos, clades, clade_names)

    def grab_level(self, PhyloLevel):
        '''
        Grab just information for a specific level.
        '''
        
        idx=[]
        
        for clade in PhyloLevel.clade_names:
            
            idx.append(np.where(self.clade_names==clade)[0][0])
        
        level_csSNPs = self.csSNPs[:,idx]

        level_csSNP_pos = self.csSNP_pos[~np.all(level_csSNPs == 0, axis=1)]
        
        return PhlameClassifier(level_csSNPs[~np.all(level_csSNPs == 0, axis=1)],
                                level_csSNP_pos,
                                PhyloLevel.clades,
                                PhyloLevel.names)
    
    def get_alleles(self):
        '''
        Get 1D list of every allele and corresponding clade.
        '''
        self.alleles = self.csSNPs[np.nonzero(self.csSNPs)]
        # corresponding clade index
        self.allele_cidx = np.nonzero(self.csSNPs)[1]


class countsCSS_NEW:
    '''
    Hold and model counts data covering a single set of cluster-specific SNPs.
    '''
    
    def __init__(self, 
                 counts, total_counts,
                 force_alpha=False,
                 prior_strength=20,
                 seed=False):
        
        self.counts = counts
        self.total_counts = total_counts
                
        # Maximum Likelihood fit
        self.counts_mle = self.zip_fit_mle(self.counts)
        self.total_mle = self.zip_fit_mle(self.total_counts)
        
        self.force_alpha = force_alpha

        self.prior_strength = prior_strength
        
        self.seed = seed
        
    def fit(self, max_pi,
            nchain=10000, 
            nburn=500,
            interval_size=0.95):
        '''
        Fit counts data to model using JAGS.
        '''
        
        if self.seed:
            np.random.seed(self.seed)
        
        # =====================================================================
        #  Set hyperparameters
        # =====================================================================
        
        if self.force_alpha:
            self.measured_alpha = self.force_alpha
            
        else:
            self.measured_alpha = np.mean(self.total_counts)**2/max(1e-6,np.var(self.total_counts)-np.mean(self.total_counts))

        m = self.prior_strength # Prior strength; higher values = stronger prior
        logp = m * digamma(self.measured_alpha)
        v = 0; s = 0
        self.params = [m, logp, v, s]
        
        # =====================================================================
        #  Gibbs sampling
        # =====================================================================
        
        try:
            param_chains, lv_chains = self.ZINB_gibbs_sampler(nchain,
                                                              nburn)
            
        except RuntimeError:
            
            print('Loglikelihood became too low and triggered an overflow error.\
                  Randomly subsampling positions and trying again...')
            
            subsample_idx = np.random.choice(np.arange(0,len(self.counts)),
                                             int(len(self.counts)/2))
            
            self.counts = self.counts[subsample_idx]
            self.total_counts = self.total_counts[subsample_idx]
            
            param_chains, lv_chains = self.ZINB_gibbs_sampler(nchain,
                                                              nburn)

        
        self.chain = {'pi':param_chains[:,0],
                      'a':param_chains[:,1],
                      'b':param_chains[:,2]}

        # =====================================================================
        #  Calc summary statistics 
        # =====================================================================
        
        prob = np.sum(self.chain['pi'] < max_pi)/len(self.chain['pi'])

        self.counts_MAP = {'pi':self.calc_MAP(self.chain['pi']),
                           'a':self.calc_MAP(self.chain['a']),
                           'b':self.calc_MAP(self.chain['b'])}
        
        self.hpd = self.get_hpd(self.chain['pi'], interval_size)
        self.prob = prob
        
        return prob, self.hpd

    @staticmethod
    def calc_MAP(chain, bins=False):

        if not bins:
            bins = 100

        counts, bins = np.histogram(chain, bins=bins)
        max_idx = np.argmax(counts)
        
        return bins[max_idx]

    def ZINB_gibbs_sampler(self, nchain, nburn):
        '''
        Gibbs sampling from the following hierarchical model: 
        '''
        nparams = 3 
        npts = len(self.counts)    
        m, logp, v, s = self.params
        
        zero_idx = np.nonzero(self.counts == 0)[0]
        nonzero_idx = np.nonzero(self.counts)[0]    
        
        # Initialize chain
        param_inits, latent_var_inits = self.initialize_chain()
        
        param_samples = np.zeros([nchain, nparams])
        param_samples[0] = param_inits
        latent_var_samples = np.zeros([nchain, 2, npts])
        latent_var_samples[0] = latent_var_inits
    
        for i in range(nchain-1):

            # sample from 𝜆i|...        
            lambda_i = self.sample_lambda_i_conditional(param_samples[i,1], 
                                                       param_samples[i,2], 
                                                       self.counts, 
                                                       latent_var_samples[i,1])
            
            # sample from ri|...
            ri = self.sample_ri_conditional(npts, param_samples[i,0], lambda_i, 
                                           zero_idx, nonzero_idx)
            
            # sample from pi|...
            pi = self.sample_p_conditional(ri, npts)
            
            # sample from b|...
            b = max(1e-6,self.sample_b_conditional(param_samples[i,1], v, s, lambda_i, npts))
            
            # sample from a|...
            a_chain = self.sample_a_conditional_gp([lambda_i,b,v,m,logp], 
                                                    param_samples[i,1])
            a = a_chain[-1]
            
            param_samples[i+1] = np.array([pi, a, b])
            latent_var_samples[i+1] = np.array([lambda_i,ri])
        
        # Convert pi into percent zero-inflated
        param_samples[:,0] = (1 - param_samples[:,0])
        
        return param_samples, latent_var_samples
    
    def initialize_chain(self):
        '''
        Initialize parameters and latent variables for Gibbs sampling.
        '''
        
        counts_mean = self.counts.mean()
        
        pi_init = min(1 - ((self.counts == 0).mean() - stats.poisson.pmf(0, counts_mean)),0.99)
        a_init = self.measured_alpha
        b_init = a_init/counts_mean
        
        param_inits = np.array([pi_init, a_init, b_init])
        
        
        ri_init = np.int64(self.counts > 0)
        predicted_zeros = (1/(1+self.measured_alpha*counts_mean))**(1/self.measured_alpha)
        ri_init[ri_init == 0] = stats.bernoulli.rvs(predicted_zeros, size=np.sum(ri_init==0))
        lambda_init = [self.counts.mean()]*len(self.counts)

        latent_var_inits = np.array([lambda_init,ri_init])
        
        return param_inits, latent_var_inits

    
    def zip_fit_mle(self, cts):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ZeroInflatedPoisson(cts)
            fit = model.fit()
            pi, lambda_ = fit.params
        
        # return lambda_, pi
            # CI_pi, CI_lambda_ = fit.conf_int()
            # # range_CI_pi = CI_pi[1] - CI_pi[0]
            # range_CI_lambda_ = CI_lambda_[1] - CI_lambda_[0]
        
        return np.array([lambda_, pi])
        

    @staticmethod
    def _calc_p_posterior(total_counts, scaleby,
                          b=1,
                          c=1):
        '''
        Empirical posterior distribution over p for Nbinom-distributed counts.
        Calculate with a beta(b,c) prior over p
        '''
        
        #If underdispersed relative to Poisson skip calc because it will break
        if np.var(total_counts) <= np.mean(total_counts):
            
            return (np.sum(total_counts),1)
            
        k = np.mean(total_counts)**2/(np.var(total_counts)-np.mean(total_counts))
        T = len(total_counts)
        alpha_update = (k * T) + b
        beta_update = np.sum(total_counts) + c
        
        # Empirical max value of p
        # p_max = np.mean(total_counts)/np.var(total_counts)
        
        return (alpha_update/scaleby, beta_update/scaleby)

    
    @staticmethod
    def _zip_nloglike(params, counts):
        '''
        Negative log-likelihood function for zero-inflated Poisson
        '''
        
        lambda_ = params[0]
        pi = params[1]
        
        if lambda_ <= 0:
            return np.inf
        if pi < 0 or pi > 1:
            return np.inf
    
        Y=len(counts) - np.count_nonzero(counts)
        n=len(counts)
        
        return -( Y*np.log(pi + (1 - pi)*np.exp(-lambda_)) +
                  (n - Y) * np.log(1 - pi) - 
                  (n - Y) * lambda_ + 
                  n * np.mean(counts) * np.log(lambda_) -
                  np.sum(np.log(np.array([np.math.factorial(int(c)) for c in counts], dtype=float))) )
                  #np.log(np.product(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) )
    
    @staticmethod
    def sample_lambda_i_conditional(a, b, counts, r_i):
        
        return stats.gamma.rvs(a = a+counts,
                               scale = 1/(r_i+b))
    
    @staticmethod
    def sample_p_conditional(ri, npts):
        
        return stats.beta.rvs(np.sum(ri)+1, npts - np.sum(ri) + 1)
    
    @staticmethod
    def sample_ri_conditional(npts, pi, lambda_i, 
                              zero_idx, nonzero_idx):
        # If yi > 0, then point i belongs to the at-risk class,
        # and hence by definition, ri = 1 with probability 1.
        # If yi = 0, then we observe either a structural zero (ri = 0) 
        # or an at-risk zero (ri = 1). 
        
        # Draw wi from a Bernoulli distribution with probability
        ri = np.zeros(npts)
        
        predicted_prob_zero = (1/(1+(np.var(lambda_i)/np.mean(lambda_i))))**(np.mean(lambda_i)**2/np.var(lambda_i))
        
        # zero_bern = ( (pi*np.exp(-np.sum(lambda_i)))/
        #               ((pi*np.exp(-np.sum(lambda_i)))+(1-pi)) )
        zero_bern = ( (pi*predicted_prob_zero)/
                      ((pi*predicted_prob_zero)+(1-pi)) )
    
        # Note as np.exp(-np.sum(lambda_i)) reaches the floating
        # point limit the overall term will go to 0
        if len(zero_idx) > 0:
            ri[zero_idx] = stats.bernoulli.rvs(zero_bern, size=len(zero_idx))
        if len(nonzero_idx) > 0:
            ri[nonzero_idx] = stats.bernoulli.rvs(1, size=len(nonzero_idx))
    
        return ri
    
    @staticmethod
    def sample_a_conditional_gp(aparams,
                                a_old):
        
        def log_a_conditional_dist_gp(a, params):
            ''' 
            Conditional distribution of a (gamma shape parameter) 
            in a gamma-poisson model
            '''
            
            lambda_i, b, v, m, logp = params
            
            # Will cause -inf error
            lambda_i = lambda_i[np.nonzero(lambda_i)]
            
            npts = len(lambda_i)
            
            return (a*(npts+v)*np.log(b) + 
                    (a)*logp + 
                    a*np.sum(np.log(lambda_i)) - 
                    (m + npts)*loggamma(a))

        if log_a_conditional_dist_gp(a_old, aparams) == -np.inf:
            raise RuntimeError('Loglikelihood is too low!')
            
        chain = slice_sampler(x0=a_old, 
                              loglike=log_a_conditional_dist_gp, 
                              params=aparams,
                              niter=50,
                              sigma=a_old/10, 
                              step_out=True)
        
        return chain
    
    @staticmethod
    def sample_b_conditional(a,
                             v, s,
                             data, npts):
        
        return stats.gamma.rvs(a = a*(npts + v),
                                scale = 1/(s + np.sum(data)))    
    
    
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
    
def slice_sampler(x0, 
                  loglike, params,
                  niter, sigma, 
                  step_out=True):
    """
    based on http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/
    """

    # set up empty sample holder
    samples = np.zeros(niter)

    # initialize
    xx = float(x0)

    last_llh = loglike(xx, params)

    for i in range(niter):
                    
        # =================================================================
        #  Randomly sample y from (0,L(X))
        # =================================================================
        # aka multiply likelihood by uniform[0,1]
        llh0 = last_llh + np.log(np.random.rand())
        
        # =================================================================
        #  Get straight line segment at y under the curve L(x)
        # =================================================================
        rr = np.random.rand(1)

        # Initialize left and right of segment with total distance sigma
        x_l = float(xx)
        x_l = x_l - rr * sigma
        x_r = float(xx)
        x_r = x_r + (1 - rr) * sigma
        
        # Continue to take steps of sigma out until you get ends outside the curve
            #"stepping out"
        if step_out:
            llh_l = loglike(x_l, params)
            while llh_l > llh0:
                x_l = x_l - sigma
                llh_l = loglike(x_l, params)
            llh_r = loglike(x_r, params)
            while llh_r > llh0:
                x_r = x_r + sigma
                llh_r = loglike(x_r, params)

        # =================================================================
        #  Randomly sample a new x from within the total segment 
        # =================================================================
            #"shrinkage"
        x_cur = float(xx)
        while True:
            xd = np.random.rand() * (x_r - x_l) + x_l
            x_cur = float(xd)
            last_llh = loglike(x_cur, params)
            
            #If the loglikelihood of new x is higher than y, good to go
            if last_llh > llh0:
                xx = float(xd)
                break
            # Otherwise make new x the new left/right end of segment
            elif xd > xx:
                x_r = xd
            elif xd < xx:
                x_l = xd
            else:
                raise RuntimeError('Slice sampler shrank too far.')

        samples[i] = float(xx)

    return samples

class ZeroInflatedPoisson(GenericLikelihoodModel):
    
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]

        return -np.log(self._zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            lambda_start = self.endog.mean()
            excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            pi_start = excess_zeros if excess_zeros>0 else 0
            start_params = np.array([pi_start, lambda_start])
            
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)
    
    @staticmethod
    def _zip_pmf(x, pi, lambda_):
        '''zero-inflated poisson function, pi is prob. of 0, lambda_ is the fit parameter'''
        if pi < 0 or pi > 1 or lambda_ <= 0:
            return np.zeros_like(x)
        else:
            return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)

class PhyloLevel:
    '''
    Holds a set phylogenetic level in reference to a classifier.
    '''
    
    def __init__(self, levelin, allclades, allclade_names):
        
        # Import information about the tree
        # Note: ideally this will be replaced by an import of a classifier class
        self.__allclades = allclades
        self.__allclade_names = allclade_names
        
        # Import levels from file or string
        if os.path.isfile(levelin):
            print('Reading in levels file...')
            self.clades, self.clade_names, self.names = self._parse_file(levelin)
        
        else: 
            print('Searching for the following levels in classifier:')
            print(levelin)
            self.clades, self.clade_names, self.names = self._parse_str(levelin)
            
    def _parse_file(self, levelin_file, uncl='-1'):
        '''
        Parse a 2 column delimited levels file 
        ( e.g genome_name1, clade_ID1
              genome_name2, clade_ID2 )
        '''
        
        with open(levelin_file,'r') as file:
            firstline = file.readline()
       
        if len(firstline.strip().split('\t'))==2:
            dlim='\t'
        elif len(firstline.strip().split(','))==2:
            dlim=','


        clade_ids = np.loadtxt(levelin_file, delimiter=dlim, dtype=str)
        
        clades, names = self._reshape(clade_ids[:,0],
                                      clade_ids[:,1],
                                      '-1')
        
        # Check that level is valid within classifier
        clade_names = self._check_file(clades, names)
        
        # Check that levels are not direct ancestors or descendants
        # of each other.
        self._ancdesc(clades, clade_names)
        
        return clades, clade_names, names

    def _parse_str(self, levelin_str):
        '''
        Parse a comma-delimited list of clade names (e.g. C.1,C.2,C.3)
        '''
        clade_names = levelin_str.strip().split(',')
        
        # Check that level is valid within classifier
        clades = self._check_str(clade_names)
        names = clade_names
        
        # Check that levels are not direct ancestors or descendants
        # of each other.
        self._ancdesc(clades, clade_names)
        
        return clades, clade_names, names
    
    def _check_file(self, clades, names):
        '''
        Check that groupings specified by a file are valid clades in classifer.
        '''
        
        clade_names=[]
        allclades_ls = list(self.__allclades.values())
        allclade_names = np.array(list(self.__allclades.keys()))
        
        # Plus return the missing data (clade_names)
        for i, clade in enumerate(clades):
        
            # Check if grouping of genomes is actually a clade in classifier
            match_bool = [ set(clade) == set(genomes) for genomes in allclades_ls ]
            
            if np.sum(match_bool)==0:
                raise Exception(f"Error: Could not find a valid clade corresponding to {names[i]} in classifier.")
            
            # Ugly
            clade_names.append(str(allclade_names[match_bool][0]))
            
        return clade_names
    
    def _check_str(self, clade_names):
        '''
        Check if clade names specified are valid clades in classifier.
        '''
        clades = []
        
        for name in clade_names:

            if name not in self.__allclade_names:
                raise Exception(f"Error: {name} is not a valid clade in the classifier!")

            clades.append(self.__allclades[name])
            
        return clades
    
    @staticmethod
    def _ancdesc(clades, clade_names):
        '''
        Check that no two clades in a level are direct ancestors or descendants 
        of each other.
        '''

        for i, clade in enumerate(clades):
            
            # Get other clades to cp against
            cp = (clades[:i] + clades[i+1 :])
            
            # Are any genomes duplicated in 2 clades
            dup_bool = [len(set(clade).intersection(set(genomes)))>0 for genomes in cp]

            if np.sum(dup_bool) > 0:
                raise Exception(f"Error: {clade_names[i]} is either an ancestor or descendant of {clade_names[dup_bool.index(True)]}.",
                                "Note that within a level the same genome cannot be included in two clades.")

    @staticmethod
    def _reshape(names_long, clade_ids_long, uncl_marker):
        '''
        Reshape a two column 'long' levels array into list of arrays
        '''
        
        clades = []; names = []
        
        for c in np.unique(clade_ids_long):
            names.append(str(c))
            clades.append(names_long[np.in1d(clade_ids_long,c)])
        
        # Remove things designated as unclassifed
        if uncl_marker in names:
            # Get index of unclassified names
            idx = names.index(uncl_marker)
            # And remove
            del names[idx]; del clades[idx]
        
        else:
            print('Nothing was found as unclassified in clade IDs. Ignore if intentional!')
                
        return clades, names

#%% Old countsCSS methods

# class countsCSS_zeus:
#     '''
#     Hold and model counts data covering a single set of clade-specific SNPs.
#     '''
    
#     def __init__(self, path_to_outdir,
#                  counts, total_counts):
        
#         self.outdir = path_to_outdir

#         self.counts = counts
#         self.total_counts = total_counts
        
#         # Maximum Likelihood fit
#         self.counts_mle = self.zip_fit_mle(self.counts)
#         self.total_mle = self.zip_fit_mle(self.total_counts)

#     def fit(self, 
#             nparams = 3,
#             nsteps = 500,
#             burn_interval = 0.1,
#             max_pi = 0.3):
#         ''' Fit counts data to model.

#         Args:
#             nsteps (TYPE, optional): DESCRIPTION. Defaults to 500.
#             burn_interval (TYPE, optional): DESCRIPTION. Defaults to 0.1.
#             max_pi (TYPE, optional): DESCRIPTION. Defaults to 0.3.

#         '''
#         self.nparams = nparams
        
#         # =====================================================================
#         #  Estimate overdispersion from total counts data
#         # =====================================================================

#         self.alpha_est = (np.var(self.total_counts)-np.mean(self.total_counts))/(np.mean(self.total_counts)**2)        

#         # =====================================================================
#         #  Run MCMC
#         # =====================================================================

#         self.chain = self.run_zeus(nsteps)
        
#         # =====================================================================
#         #  Calculate point estimates
#         # =====================================================================
        
#         self.counts_map = self.get_map_est() 
#         # In order lambda, alpha, pi
        
#         prob = np.sum(self.chain[:,2] < max_pi)/len(self.chain[:,2])
        
#         return prob
    
#     def run_zeus(self, nsteps):
#         '''
#         Run zeus Ensemble Slice Sampler.
#         '''
        
#         # Number of walkers to use. Should be at least 2x the number of dimensions.
#         nwalkers = self.nparams*2 
    
#         # Initial positions of the walkers.
#         init_ = 0.01 * np.random.randn(nwalkers, self.nparams) 
                
#         # Initialise the sampler
#         sampler = zeus.EnsembleSampler(nwalkers,
#                                        self.nparams, 
#                                        self.zinb_logpost,
#                                        args=[self.counts, self.alpha_est])
        
#         # Run sampling
#         sampler.run_mcmc(init_, nsteps)
#         # print(sampler.summary)
        
#         chain = sampler.get_chain(flat=True, discard=nsteps//10, thin=1)
#         chain[:,0] = np.exp(chain[:,0])  # lambda
#         chain[:,1] = np.exp(chain[:,1])  # alpha
#         chain[:,2] = logit_link(chain[:,2]) # pi

#         return chain
    
#     def get_map_est(self):
        
#         map_arr = np.median(self.chain, axis=0)
                
#         return map_arr
    
#     def zip_fit_mle(self, cts):
        
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model = ZeroInflatedPoisson(cts)
#             fit = model.fit()
#             pi, lambda_ = fit.params
                
#         return np.array([lambda_, pi])
    
#     def zinb_logpost(self, params, counts, laplace_est):
#         '''Log of the posterior distribution'''
        
#         laplace_est = max((1e-6,laplace_est))
        
#         return self._zinb_logprior(params, laplace_est) + self._zinb_loglike_obs(params, counts)

#     @staticmethod 
#     def _zinb_loglike_obs(params, counts):
#         '''
#         Negative log-likelihood function of a zero-inflated negative binomial model.
#         Parameterized where alpha->0, the distribution approaches a Poisson.
#         '''
#         # Expand params
#         lambda_, alpha, pi_link = params
#         lambda_ = np.exp(np.clip(lambda_, None, EXP_UPPER_LIMIT))
#         alpha = np.exp(np.clip(alpha, None, EXP_UPPER_LIMIT))
#         pi_link = np.exp(np.clip(pi_link, None, EXP_UPPER_LIMIT))
#         # pi_link = pi/(1-pi)
        
#         zero_idx = np.nonzero(counts == 0)[0]
#         nonzero_idx = np.nonzero(counts)[0]
        
#         ll_obs = np.zeros_like(counts, dtype=np.float64)
        
#         ll_obs[zero_idx] = (np.log(pi_link + (1+(lambda_*alpha))**-(1/alpha)) -
#                             np.log(1+pi_link))
                            
#         ll_obs[nonzero_idx] = (loggamma(counts[nonzero_idx] + (1/alpha)) -
#                                loggamma(counts[nonzero_idx]+1) -
#                                loggamma(1/alpha) -
#                                ((counts[nonzero_idx] + (1/alpha))*np.log(1+(lambda_*alpha))) + 
#                                (counts[nonzero_idx]*np.log(alpha)) + 
#                                (counts[nonzero_idx]*np.log(lambda_)) -
#                                np.log(1+pi_link)
#                                )        
        
#         # if np.isnan(np.sum(ll_obs)):
#         #     print(params)
    
#         return np.sum(ll_obs)
    
#     @staticmethod
#     def _zinb_logprior(params, laplace_est):
#         '''
#         Prior distribution on zero-inflated negative binomial  
#         '''
#         # Expand params
#         lambda_, alpha, p = params
        
#         # Because alpha is [0,inf) and we are doing laplace estimator we can
#         # specify the Gaussian prior directly in log space 
#         # (e.g using a lognormal prior over actual parameter space)
    
#         # Loglikelihood of gaussian prior on alpha ( Laplace approximation )
#         sigma = 0.1 # standard deviation of the Gaussian prior
#         lp = -0.5*((alpha - np.log(laplace_est))/sigma)**2
        
#         return lp

#     # @staticmethod
#     # def _zip_nloglike(params, counts):
#     #     '''
#     #     Negative log-likelihood function for zero-inflated Poisson
#     #     '''
        
#     #     lambda_ = params[0]
#     #     pi = params[1]
        
#     #     if lambda_ <= 0:
#     #         return np.inf
#     #     if pi < 0 or pi > 1:
#     #         return np.inf
    
#     #     Y=len(counts) - np.count_nonzero(counts)
#     #     n=len(counts)
        
#     #     return -( Y*np.log(pi + (1 - pi)*np.exp(-lambda_)) +
#     #              (n - Y) * np.log(1 - pi) - 
#     #              (n - Y) * lambda_ + 
#     #              n * np.mean(counts) * np.log(lambda_) -
#     #              np.sum(np.log(np.array([np.math.factorial(int(c)) for c in counts], dtype=float))) )

# class countsCSS_NEW:
#     '''
#     Hold and model counts data covering a single set of cluster-specific SNPs.
#     '''
    
#     def __init__(self, path_to_outdir,
#                  counts, total_counts):
        
#         self.outdir = path_to_outdir

#         self.counts = counts
#         self.total_counts = total_counts
        
#         # Names of temp files for JAGS
#         self.model_filename = "model.bug"
#         self.data_filename = "data.txt"
#         self.run_filename = "jags.bash"

#         # Maximum Likelihood fit
#         self.counts_mle = self.zip_fit_mle(self.counts)
#         self.total_mle = self.zip_fit_mle(self.total_counts)
        
#         # Load in models
#         # self.models()

#     def fit(self, 
#             nchains = 1,
#             niter = 100000,
#             nburn = 5000,
#             max_pi = 0.3):
#         '''
#         Fit counts data to model using JAGS.
#         '''

#         # =====================================================================
#         #  Calc posterior over p for total_counts
#         # =====================================================================
#         scaleby=10

#         self.p_update = self._calc_p_posterior(self.total_counts, scaleby)
#         print(self.p_update)

#         # =====================================================================
#         #  Write JAGS files
#         # =====================================================================
#         self.make_zinb_model(p_hyper=self.p_update)
#         self.make_zinb_run(nchains, round(niter/nchains), nburn)
        
#         # =====================================================================
#         #  Run JAGS
#         # =====================================================================
#         subprocess.run(f"mkdir -p {self.outdir}", shell=True)

#         print("Writing JAGS files..")

#         self.write_jags()
        
#         print("Running JAGS..")
        
#         self.run_jags(nchains, niter)        
        
#         # =====================================================================
#         #  Calc summary statistics
#         # =====================================================================
        
#         # self.chain_arr = np.array((self.chain['alpha'],
#         #                             self.chain['p'],
#         #                             self.chain['pi']))

#         prob = np.sum(self.chain[:,0] < max_pi)/len(self.chain[:,0])
        
#         return prob

#     def run_jags(self, nchains, niter):
#         '''
#         Run JAGS on command line
#         '''
#         # print(f'bash -c "source activate base; jags {self.outdir}/{self.runfile}"')
#         subprocess.run(f'bash -c "source activate base; jags {self.outdir}/{self.run_filename}"', 
#                        shell=True)
        
#         self.params = {}
        
#         with open(f'{self.outdir}/jags_index.txt') as file:
            
#             for line in file:
#                 ls = line.rstrip('\n').split(' ')
#                 # params[ls[0]] = ls[1:]
#                 self.params[ls[0]] = [int(s) for s in ls[1:]]
                
        
#         chain_ls = []
        
#         for chain_num in range(nchains):
            
#             jags_chain = np.loadtxt(f'{self.outdir}/jags_chain{chain_num+1}.txt')[:,1]
            
#             chain_tmp = np.zeros(round(niter))
            
#             for pidx, param in enumerate(self.params.keys()):
                
#                 chain_idxs = self.params[param]
                
#                 #-1 to convert 1-index to 0-index
#                 chain_tmp = jags_chain[chain_idxs[0]-1:chain_idxs[1]]
                
#                 chain_ls.append(chain_tmp)
        
#         self.chain = np.vstack(chain_ls).T

#     def write_jags(self):
#         '''
#         Write files to run JAGS on command line
#         '''
        
#         with open(f'{self.outdir}/{self.run_filename}','w') as f:

#             f.write(self.zinb_run)
        
#         with open(f'{self.outdir}/{self.model_filename}','w') as f:
#             f.write(self.zinb_model)
            
#         with open(f'{self.outdir}/{self.data_filename}','w') as f:
#             f.write(f'"y" <- c({",".join(self.counts.astype(str))})\n')
#             f.write(f'"N" <- {len(self.counts)}')
#             # f.write(f'"mean" <- {max(2,round(np.mean(counts)))}')

#     # def fit_mle(self, cts):
#     #     '''
#     #     Maximum likelihood fit to zero-inflated poisson
#     #     '''
#     #     # Inital guess
#     #     lambda_init = cts.mean()
#     #     excess_zeros = ((cts == 0).mean() - 
#     #                     stats.poisson.pmf(0, lambda_init) )
#     #     pi_init = excess_zeros if excess_zeros>0 else 0
        
#     #     # Fit parameters
#     #     result = minimize(self._zip_nloglike, 
#     #                       [lambda_init, pi_init], 
#     #                       args=cts)
#     #     return result.x
    
#     def zip_fit_mle(self, cts):
        
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model = ZeroInflatedPoisson(cts)
#             fit = model.fit()
#             pi, lambda_ = fit.params
        
#         # return lambda_, pi
#             # CI_pi, CI_lambda_ = fit.conf_int()
#             # # range_CI_pi = CI_pi[1] - CI_pi[0]
#             # range_CI_lambda_ = CI_lambda_[1] - CI_lambda_[0]
        
#         return np.array([lambda_, pi])
        
#     def make_zinb_run(self,
#                       nchain,
#                       niter,
#                       nburn):
#         '''
#         JAGS format code to run MCMC on zero-inflated binomial model
#         '''
#         self.zinb_run = (
#             f'model in "{self.outdir}/{self.model_filename}"\n'
#             f'data in "{self.outdir}/{self.data_filename}"\n'
#             f'compile, nchains({nchain})\n'
#             f'initialize\n'
#             f'update {nburn}\n'
#             f'monitor pi\n'
#             f'monitor p\n'
#             f'monitor alpha\n'
#             f'monitor beta\n'
#             f'update {niter}\n'
#             f'coda *, stem("{self.outdir}/jags_")\n'
#             'exit')
    
#     def make_zinb_model(self, 
#                         alpha_hyper=(0.001,0.001), 
#                         pi_hyper=(1.001,1.001),
#                         p_hyper=(1.001,1.001)):
#         '''
#         JAGS format code to create a zero-inflated binomial model
#         Parameterized as follows:
#         Y ~ Gamma*Z
#         Z ~ Bernoulli(1-pi)
#         Gamma ~ Nbinom(alpha,p)
#         '''
        
#         self.zinb_model = ("""model {
#         ## Likelihood ##
#         for( i in 1:N ) {
#           y[i] ~ dpois( lambda.corrected[i] )
#           lambda.corrected[i] <- zero[i] * gamma[i] + 0.00001
#           zero[i] ~ dbern(1 - pi) # Bernoulli component for zero-inflation
#           gamma[i] ~ dgamma(alpha, beta)
#         }
#         beta <- p/(1 - p)
#         """
#         "## Priors ##\n"
#         f"    alpha ~ dgamma({alpha_hyper[0]},{alpha_hyper[1]})\n"
#         f"    pi ~ dbeta({pi_hyper[0]},{pi_hyper[1]}) # Beta prior on pi\n"
#         f"    p ~ dbeta({p_hyper[0]},{p_hyper[1]})\n"
#         "}"
#             )

#     def models(self):
#         '''
#         Old code to hold all the models
#         '''
#         self.zinb_model = """model {
#         ## Likelihood ##
#         for( i in 1:N ) {
#           y[i] ~ dpois( lambda.corrected[i] )
#           lambda.corrected[i] <- zero[i] * gamma[i] + 0.00001
#           zero[i] ~ dbern(1 - pi) # Bernoulli component for zero-inflation
#           gamma[i] ~ dgamma(alpha, beta)
#         }
#         beta <- p/(1 - p)

#         ## Priors ##
#         alpha ~ dgamma(0.001,0.001)
#         p ~ dbeta(1.001,1.001)
#         pi ~ dbeta(1.001,1.001) # Beta prior on pi        
#         }
#         """
        
#         self.nbinom_model = """model {
#         ## Likelihood ##
#         for( i in 1:N ) {
#           y[i] ~ dnegbin( p , n )
#         }
        
#         ## Priors ##
#         p ~ dbeta(1.001,1.001)
#         n ~ dgamma(0.001,0.001)
        
#         ## NB mean and variance
#         mu <- n*(1-p)/p
#         variance <- n*(1-p)/(p*p)
#         }
#         """
        
#         self.nbinom_run = (
#         f'model in "{self.outdir}/{self.modelfile}"\n'
#         f'data in "{self.outdir}/{self.datafile}"\n'
#         f'compile, nchains(1)\n'
#         f'initialize\n'
#         f'update 1000\n'
#         f'monitor p\n'
#         f'monitor n\n'
#         f'monitor mu\n'
#         f'monitor variance\n'
#         f'update 100000\n'
#         f'coda *, stem("{self.outdir}/jags_")\n'
#         'exit'
#         )
        
#         self.zinb_run = (
#         f'model in "{self.outdir}/{self.modelfile}"\n'
#         f'data in "{self.outdir}/{self.datafile}"\n'
#         f'compile, nchains(1)\n'
#         f'initialize\n'
#         f'update 5000\n'
#         f'monitor pi\n'
#         f'monitor p\n'
#         f'monitor alpha\n'
#         f'monitor beta\n'
#         f'update 100000\n'
#         f'coda *, stem("{self.outdir}/jags_")\n'
#         'exit'
#         )

#     @staticmethod
#     def _calc_p_posterior(total_counts, scaleby,
#                          b=1,
#                          c=1):
#         '''
#         Empirical posterior distribution over p for Nbinom-distributed counts.
#         Calculate with a beta(b,c) prior over p
#         '''
        
#         #If underdispersed relative to Poisson skip calc because it will break
#         if np.var(total_counts) <= np.mean(total_counts):
            
#             return (np.sum(total_counts),1)
            
#         k = np.mean(total_counts)**2/(np.var(total_counts)-np.mean(total_counts))
#         T = len(total_counts)
#         alpha_update = (k * T) + b
#         beta_update = np.sum(total_counts) + c
        
#         # Empirical max value of p
#         # p_max = np.mean(total_counts)/np.var(total_counts)
        
#         return (alpha_update/scaleby, beta_update/scaleby)

    
#     @staticmethod
#     def _zip_nloglike(params, counts):
#         '''
#         Negative log-likelihood function for zero-inflated Poisson
#         '''
        
#         lambda_ = params[0]
#         pi = params[1]
        
#         if lambda_ <= 0:
#             return np.inf
#         if pi < 0 or pi > 1:
#             return np.inf
    
#         Y=len(counts) - np.count_nonzero(counts)
#         n=len(counts)
        
#         return -( Y*np.log(pi + (1 - pi)*np.exp(-lambda_)) +
#                  (n - Y) * np.log(1 - pi) - 
#                  (n - Y) * lambda_ + 
#                  n * np.mean(counts) * np.log(lambda_) -
#                  np.sum(np.log(np.array([np.math.factorial(int(c)) for c in counts], dtype=float))) )
#                  #np.log(np.product(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) ) 

# class countsCSS:
#     '''
#     Hold and model counts data covering a single set of cluster-specific SNPs.
#     '''
    
#     def __init__(self, counts):
        
#         self.counts = counts
        
#         #Maximum Likelihood fit
#         self.mle = self.mle_fit()
                
#     def bayesian_fit(self, max_pi=0.3, niter=100000, nburn=2000):
#         '''
#         Bayesian fit to zero-inflated poisson
#         '''
        
#         # Initialize chain
#         bayes_init = self.mle
        
#         # Run chain
#         chain, accept_rate, lnprob = self._MCMC(self.counts, bayes_init,
#                                                self._zip_loglike, 
#                                                self._gaussian_prop_fn, 
#                                                prop_fn_kwargs={'sigma':np.array([0.2,0.4])},
#                                                niter=niter)
        
#         # Burn off first part of chain
#         chain_burn = chain[nburn:]
        
#         # Bin chain to nearest 0.01
        
#         lambda_bins = np.arange(0,
#                                 max(chain_burn[:,0]) + 0.01,
#                                 0.01)
#         pi_bins = np.arange(0,1.01,0.01)

#         # # 'Bin' values go leftmost (e.g. 0.01->0.02 count get assigned to 0.01)
#         lambda_hist = np.histogram(chain_burn[:,0],
#                                     bins=lambda_bins)
#         pi_hist = np.histogram(chain_burn[:,1],
#                                     bins=pi_bins)
        
#         #Multiply bins & grab MAP value
#         jt_bincount = np.outer(lambda_hist[0], pi_hist[0])
        
#         MAP_idx = np.unravel_index(jt_bincount.argmax(), 
#                                    jt_bincount.shape)
#         MAP_lambda = lambda_bins[MAP_idx[0]]
#         MAP_pi = pi_bins[MAP_idx[1]]

#         # Probability of a set of parameters where pi < max_pi
#         prob = np.sum(chain_burn[:,1] < max_pi)/(len(chain)-nburn)
       
#         # Save outputs
#         self.map = tuple([MAP_lambda,MAP_pi])
#         self.prob = prob
#         self.chain = chain_burn
        
#         return prob
        
#     def mle_fit(self):
#         '''
#         Maximum likelihood fit to zero-inflated poisson
#         '''
#         # Inital guess
#         lambda_init = self.counts.mean()
#         excess_zeros = ((self.counts == 0).mean() - 
#                         stats.poisson.pmf(0, lambda_init) )
#         pi_init = excess_zeros if excess_zeros>0 else 0
        
#         # Fit parameters
#         result = minimize(self._zip_nloglike, 
#                           [lambda_init, pi_init], 
#                           args=self.counts)
#         return result.x
    
#     @staticmethod
#     def _MCMC(counts, x0,
#           lnprob_fn, prop_fn, 
#           prop_fn_kwargs={}, 
#           niter=100000):
#         '''Metropolis-Hastings MCMC sampler
        
#         Args:
#             counts (arr): Vector of counts data to model
#             x0 (arr): Initial vector of parameters.
#             lnprob_fn (fxn): Function to compute log-posterior probability.
#             prop_fn (fxn): Function to draw proposals.
#             prop_fn_kwargs (TYPE, optional): Arguments for proposal function.
#             niter (int, optional): Number of iterations to run chain.
        
#         Returns:
#             chain (arr): Vector of chain.
#             accept_rate (float): Acceptance rate.
#             lnprob (arr): Log-posterior chain.
        
#         '''
        
#         ndim = len(x0)
        
#         # Data structures
#         chain = np.zeros((niter, ndim))
#         lnprob = np.zeros(niter)
#         accept_rate = np.zeros(niter)
        
#         # Generate accept decisions all @ once
#         u = np.random.uniform(0,1,niter)
        
#         # Initialize first samples
#         x0[x0<=0]=0.0001 # change 0-valued/neg initials to some small number
#         x = np.log(x0)    # transform initial parameters (lognormal) to new (normal)
#         chain[0] = tuple(x)
#         lnprob0 = lnprob_fn(x0, counts)
#         lnprob[0] = lnprob0
        
#         # start loop
#         naccept = 0
#         for ii in range(1, niter):
            
#             # Counter
#             if ii%10000==0:
#                 print('.')
        
#             # Propose the following parameters
#             x_star, factor = prop_fn(tuple(x), **prop_fn_kwargs)
            
#             # Because we log-transformed we cannot use the normal factor
#             # q(xi|xi+1)/q(xi+1|xi) = xi+1/xi for lognormal distribution
#             factor = np.product(np.exp(x_star)/np.exp(x))
            
#             if np.exp(x_star[1]) >=1: #auto-reject if pi > 1
#                 chain[ii] = x
#                 lnprob[ii] = lnprob0
#                 accept_rate[ii] = naccept / ii
                
#                 continue
        
#             # Compute Hastings ratio
#             # Log-posterior prob.
#             lnprob_plus1 = lnprob_fn(np.exp(x_star), counts)
#             # Proposal prob.
#             H = np.exp(lnprob_plus1 - lnprob0) * factor 
#             # equivalent to H = exp(ln(prob_plus1/prob0))
        
#             # Accept/Reject step (update acceptance counter)
#             if u[ii-1] < H:
#                 x = x_star
#                 lnprob0 = lnprob_plus1
#                 naccept += 1
        
#             # update chain
#             chain[ii] = x
#             lnprob[ii] = lnprob0
#             accept_rate[ii] = naccept / ii
        
#         return np.exp(chain), accept_rate, lnprob

#     @staticmethod
#     def _zip_loglike(params, counts):
#         '''
#         Log-likelihood function for zero-inflated Poisson
#         '''
        
#         lambda_ = params[0]
#         pi = params[1]
        
#         if lambda_ <= 0:
#             return -np.inf
#         if pi < 0 or pi > 1:
#             return -np.inf
    
#         Y=len(counts) - np.count_nonzero(counts)
#         n=len(counts)
        
#         return ( Y*np.log(pi + (1 - pi)*np.exp(-lambda_)) +
#                (n - Y) * np.log(1 - pi) - 
#                (n - Y) * lambda_ + 
#                 n * np.mean(counts) * np.log(lambda_) -
#                 np.sum(np.log(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) )
#                 #np.log(np.product(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) ) 

#     @staticmethod
#     def _zip_nloglike(params, counts):
#         '''
#         Negative log-likelihood function for zero-inflated Poisson
#         '''
        
#         lambda_ = params[0]
#         pi = params[1]
        
#         if lambda_ <= 0:
#             return np.inf
#         if pi < 0 or pi > 1:
#             return np.inf
    
#         Y=len(counts) - np.count_nonzero(counts)
#         n=len(counts)
        
#         return -( Y*np.log(pi + (1 - pi)*np.exp(-lambda_)) +
#                  (n - Y) * np.log(1 - pi) - 
#                  (n - Y) * lambda_ + 
#                  n * np.mean(counts) * np.log(lambda_) -
#                  np.sum(np.log(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) )
#                  #np.log(np.product(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) ) 
    
#     @staticmethod
#     def _gaussian_prop_fn(x0, sigma=0.2):
#         """
#         Gaussian proposal distribution.
    
#         Propose new parameters based on Gaussian distribution with
#         mean at current position and standard deviation sigma.
    
#         Since the mean is the current position and the standard
#         deviation is fixed. This proposal is symmetric so the ratio
#         of proposal densities is 1.
        
#         :param x0: Parameter array
#         :param sigma:
#             Standard deviation of Gaussian distribution. Can be scalar
#             or vector of len(x0)
    
#         :returns: (new parameters, ratio of proposal densities)
#         """
    
#         # Propose new parameters based on gaussian
#         # (Every normal is a version of stdnormal stretched
#         # by sigma and translated by x)
#         if hasattr(x0, '__len__'):
#             x_star = x0 + np.random.randn(len(x0)) * sigma
            
#         else:
#             x_star = x0 + np.random.randn() * sigma
        
#         # proposal ratio factor is 1 since jump is symmetric
#         qxx = 1
    
#         return (x_star, qxx)
        

# counts = clade_counts[7][0] + clade_counts[7][1]
# init = np.array([0,0.1])
# niter=20000
# nburn=2000

# def bayesian_fit(self, max_pi=0.3, niter=100000, nburn=2000):
#     '''
#     Bayesian fit to zero-inflated poisson.
#     '''
    
#     # Initialize chain
#     bayes_init = init
    
#     # Run chain
#     chain, accept_rate, lnprob = MCMC(counts, bayes_init,
#                                         zip_loglike, 
#                                         gaussian_prop_fn, 
#                                         prop_fn_kwargs={'sigma':np.array([0.2,0.4])},
#                                         niter=niter)
    
#     # Burn off first part of chain
#     chain_burn = chain[nburn:]
    
#     # Bin chain to nearest 0.01
#     lambda_bins = np.arange(0,
#                             max(chain_burn[:,0]) + 0.02,
#                             0.01)
#     pi_bins = np.arange(0,1.01,0.01)

#     # 'Bin' values go leftmost (e.g. 0.01->0.02 count get assigned to 0.01)
#     lambda_hist = np.histogram(chain_burn[:,0],
#                                 bins=lambda_bins)
#     pi_hist = np.histogram(chain_burn[:,1],
#                                 bins=pi_bins)

#     # lambda_bincount = np.bincount( np.digitize(chain_burn[:,0],
#     #                                             bins=lambda_bins) )
#     # pi_bincount = np.bincount( np.digitize(chain_burn[:,1],
#     #                                         bins=pi_bins) )
    
#     #Multiply bins & grab MAP value
#     jt_bincount = np.outer(lambda_hist[0], pi_hist[0])
#     # jt_bincount = np.outer(lambda_bincount, pi_bincount)

#     MAP_idx = np.unravel_index(jt_bincount.argmax(), 
#                                jt_bincount.shape)
    
#     MAP_lambda = lambda_bins[MAP_idx[0]]
#     MAP_pi = pi_bins[MAP_idx[1]]
#         # nothing is binned to pi=0 (want to fix)
    
#     # Probability of a set of parameters where pi < max_pi
#     prob = np.sum(chain_burn[:,1] < max_pi)/(len(chain)-nburn)
   
#     # Save outputs
#     self.map = tuple([MAP_lambda,MAP_pi])
#     self.prob = prob
    
#     return prob

# def zip_loglike(params, counts):
#     '''
#     Log-likelihood function for zero-inflated Poisson
#     '''
    
#     lambda_ = params[0]
#     pi = params[1]
    
#     if lambda_ <= 0:
#         return -np.inf
#     if pi < 0 or pi > 1:
#         return -np.inf

#     Y=len(counts) - np.count_nonzero(counts)
#     n=len(counts)
    
#     return ( Y*np.log(pi + (1 - pi)*np.exp(-lambda_)) +
#            (n - Y) * np.log(1 - pi) - 
#            (n - Y) * lambda_ + 
#             n * np.mean(counts) * np.log(lambda_) -
#             np.sum(np.log(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) )
#             #np.log(np.product(np.array([np.math.factorial(int(c)) for c in counts], dtype='float128'))) ) 

# def gaussian_prop_fn(x0, sigma=0.2):
#     """
#     Gaussian proposal distribution.

#     Propose new parameters based on Gaussian distribution with
#     mean at current position and standard deviation sigma.

#     Since the mean is the current position and the standard
#     deviation is fixed. This proposal is symmetric so the ratio
#     of proposal densities is 1.
    
#     :param x0: Parameter array
#     :param sigma:
#         Standard deviation of Gaussian distribution. Can be scalar
#         or vector of len(x0)

#     :returns: (new parameters, ratio of proposal densities)
#     """

#     # Propose new parameters based on gaussian
#     # (Every normal is a version of stdnormal stretched
#     # by sigma and translated by x)
#     if hasattr(x0, '__len__'):
#         x_star = x0 + np.random.randn(len(x0)) * sigma
        
#     else:
#         x_star = x0 + np.random.randn() * sigma
    
#     # proposal ratio factor is 1 since jump is symmetric
#     qxx = 1

#     return (x_star, qxx)

# def MCMC(counts, x0,
#       lnprob_fn, prop_fn, 
#       prop_fn_kwargs={}, 
#       niter=100000):
#     '''Metropolis-Hastings MCMC sampler
    
#     Args:
#         counts (arr): Vector of counts data to model
#         x0 (arr): Initial vector of parameters.
#         lnprob_fn (fxn): Function to compute log-posterior probability.
#         prop_fn (fxn): Function to draw proposals.
#         prop_fn_kwargs (TYPE, optional): Arguments for proposal function.
#         niter (int, optional): Number of iterations to run chain.
    
#     Returns:
#         chain (arr): Vector of chain.
#         accept_rate (float): Acceptance rate.
#         lnprob (arr): Log-posterior chain.
    
#     '''
    
#     ndim = len(x0)
    
#     # Data structures
#     chain = np.zeros((niter, ndim))
#     lnprob = np.zeros(niter)
#     accept_rate = np.zeros(niter)
    
#     # Generate accept decisions all @ once
#     u = np.random.uniform(0,1,niter)
    
#     # Initialize first samples
#     x0[x0<=0]=0.0001 # change 0-valued/neg initials to some small number
#     x = np.log(x0)    # transform initial parameters (lognormal) to new (normal)
#     chain[0] = tuple(x)
#     lnprob0 = lnprob_fn(x0, counts)
#     lnprob[0] = lnprob0
    
#     # start loop
#     naccept = 0
#     for ii in range(1, niter):
        
#         # Counter
#         if ii%10000==0:
#             print('.')
    
#         # Propose the following parameters
#         x_star, factor = prop_fn(tuple(x), **prop_fn_kwargs)
        
#         # Because we log-transformed we cannot use the normal factor
#         # q(xi|xi+1)/q(xi+1|xi) = xi+1/xi for lognormal distribution
#         factor = np.product(np.exp(x_star)/np.exp(x))
        
#         if np.exp(x_star[1]) >=1: #auto-reject if pi > 1
#             chain[ii] = x
#             lnprob[ii] = lnprob0
#             accept_rate[ii] = naccept / ii
            
#             continue
    
#         # Compute Hastings ratio
#         # Log-posterior prob.
#         lnprob_plus1 = lnprob_fn(np.exp(x_star), counts)
#         # Proposal prob.
#         H = np.exp(lnprob_plus1 - lnprob0) * factor 
#         # equivalent to H = exp(ln(prob_plus1/prob0))
    
#         # Accept/Reject step (update acceptance counter)
#         if u[ii-1] < H:
#             x = x_star
#             lnprob0 = lnprob_plus1
#             naccept += 1
    
#         # update chain
#         chain[ii] = x
#         lnprob[ii] = lnprob0
#         accept_rate[ii] = naccept / ii
    
#     return np.exp(chain), accept_rate, lnprob

# #%% OLD

# def classify(path_to_cts_file, path_to_classifier, level_input, 
#               path_to_output_frequencies, path_to_output_data=False,
#               max_perc_diff=0.3, max_snp_diff=False, min_snps=10):
#     '''Type strains in a metagenome sample given a PhLAMe classifier file.

#     Args:
#         path_to_cts_file (str): Path to PhLAMe counts file.
#         path_to_classifier (str): Path to PhLAMe classifier file.
#         level_input (str): Path to file defining the phylogenetic level
#         to type strains at. This file can either list clade names comma-delimited,
#         or group genomes into clades with labels in a 2 column .tsv.
#         path_to_output_frequencies (str): Path to output frequencies file (.csv).
#         path_to_output_data (str, optional): If True, outputs a data file with counts
#         and modeling information at the defined level. Defaults to False.
#         max_perc_diff (TYPE, optional): Maximum . Defaults to 0.3.
#         max_snp_diff (TYPE, optional): If True, . Defaults to False.
#         min_snps (int, optional): Minimum number of csSNPs with >0 counts to call
#                                   a lineage as present. Defaults to 10.

#     Returns:
#         cts_cov (TYPE): DESCRIPTION.
#         fit_info (TYPE): DESCRIPTION.
#         sample_frequencies (TYPE): DESCRIPTION.

#     '''
    
#     print("Reading in counts file...")
#     with gzip.open(path_to_cts_file,'rb') as f:
#         counts, pos = pickle.load(f)
                
#     print("Reading in classifier file...")
#     with gzip.open(path_to_classifier, 'rb') as f:
#         cssnp_dct = pickle.load(f)
        
#         all_csSNPs = cssnp_dct['cssnps']
#         csSNP_pos = cssnp_dct['cssnp_pos']
#         all_clades = cssnp_dct['clades']
#         all_clade_names = cssnp_dct['clade_names']
        
        
#     print("Reading in levels...")
#     mylevel = PhyloLevel(level_input, all_clades, all_clade_names)
    
#     #grab csSNPs for level clades
#     csSNPs = all_csSNPs[:,np.in1d(all_clade_names, mylevel.clade_names)]

#     print("Classifying...")
#     cts_cov, fit_info, frequencies = counts_CSS(counts, pos, 
#                                                 csSNPs, csSNP_pos,
#                                                 mylevel.names,
#                                                 min_snps=min_snps)
#     #Save output
#     frequencies.to_csv(path_to_output_frequencies, sep=',')
    
#     if path_to_output_data:
#         with gzip.open(path_to_output_data,'wb') as f:
#             pickle.dump([cts_cov, fit_info],f)
    
#     return cts_cov, fit_info, frequencies


# def counts_CSS(counts, pos, csSNPs, csSNP_pos, clade_names, 
#                 min_snps=10, max_pi=0.3, min_prob=0.85):
#     '''
#     Args:
#         counts (arr): Array of allele counts across informative positions (8xp).
#         pos (arr): Corresponding positions on the reference genome for counts (p x 1).
#         csSNPs (arr): p x c matrix giving csSNPs for each clade in typing scheme (01234=NATCG).
#         csSNP_pos (arr): Corresponding positions on the reference for csSNPs.
#         clade_names (list): List of clade names to label output frequencies file.
#         min_snps (int): Minimum number SNPs with non-zero counts in sample to count a clade as present.

#     Returns:
#         counts_coverage (TYPE): DESCRIPTION.
#         filter_counts_coverage (TYPE): DESCRIPTION.
#         fit_info (TYPE): DESCRIPTION.
#         sample_frequencies (TYPE): DESCRIPTION.

#     '''
    
#     # =========================================================================
#     #     Make some data structures to index through arrays & store results
#     # =========================================================================
    
#     # positions on reference for this level
#     # counts_CSS_pos = csSNP_pos[np.sum(csSNPs,1)>0]
#     counts_CSS_pos = csSNP_pos[np.nonzero(csSNPs)[0]]

#     # corresponding index on counts mat
#     # counts_CSS_idx = np.arange(0,len(pos))[np.sum(csSNPs,1)>0]
#     counts_CSS_idx = np.arange(0,len(pos))[np.nonzero(csSNPs)[0]]

#     # cssnp allele 01234=NATCG 
#     # (note will be larger than counts_CSS_pos because pos can have >1 allele)
#     alleles = csSNPs[np.nonzero(csSNPs)]
#     # corresponding clade index
#     clades = np.nonzero(csSNPs)[1]

#     # Check
#     counts_CSS_bool = np.in1d(pos,counts_CSS_pos)
#     if np.sum(counts_CSS_bool) != len(np.unique(counts_CSS_pos)):
#         raise Exception('Counts and classifier positions do not match. ',
#                         'Check that reference genomes are same.')
                
#     # =========================================================================
#     #     Grab just the allele info from counts
#     # =========================================================================
    
#     np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 warnings
    
#     cts_f = np.zeros(len(counts_CSS_pos))
#     cts_r = np.zeros(len(counts_CSS_pos))
#     tot_f = np.zeros(len(counts_CSS_pos))
#     tot_r = np.zeros(len(counts_CSS_pos))
    
#     for i, p in enumerate(counts_CSS_pos):
        
#         # -1 +3 is to convert 01234 NATCG to index 
#         cts_f[i] = counts[counts_CSS_idx[i],int(alleles[i]-1)]
#         cts_r[i] = counts[counts_CSS_idx[i],int(alleles[i]+3)]
        
#         tot_f[i] = np.sum(counts[counts_CSS_idx[i],:4])
#         tot_r[i] = np.sum(counts[counts_CSS_idx[i],4:])

#     # def get_allele_counts(self):
        
#     #     # =====================================================================
#     #     #     Grab just the allele info from counts
#     #     # =====================================================================
        
#     #     np.seterr(divide='ignore', invalid='ignore') #ignore divide by 0 warnings
        
#     #     counts = self.countsmat.counts
#     #     counts_idx = self.informative_pos_idx
#     #     alleles = self.level_cfr.alleles
        
#     #     #Initialize data structures
#     #     cts_f = np.zeros(len(counts_idx))
#     #     cts_r = np.zeros(len(counts_idx))
#     #     tot_f = np.zeros(len(counts_idx))
#     #     tot_r = np.zeros(len(counts_idx))
        
#     #     for i, p in enumerate(self.informative_pos):
            
#     #         # -1 +3 is to convert 01234 NATCG to index 
#     #         cts_f[i] = counts[counts_idx[i],
#     #                           int(alleles[i]-1)]
            
#     #         cts_r[i] = counts[counts_idx[i],
#     #                           int(alleles[i]+3)]
            
#     #         tot_f[i] = np.sum(counts[counts_idx[i],:4])
#     #         tot_r[i] = np.sum(counts[counts_idx[i],4:])
            
#     #     self.allele_counts = tuple([cts_f,cts_r,tot_f,tot_r])


#     # =========================================================================
#     #     For each clade - fit data to expected counts model
#     # =========================================================================
    
#     frequencies = np.zeros(len(clade_names))
#     save_cts_f = []; save_cts_r = []
#     save_tot_f = []; save_tot_r = []
#     save_cts_pos = []


#     # Reshape data to group by clade
#     for c in range(len(clade_names)):
        
#         # Grab fwd and rev counts for this clade
#         clade_cts_f = cts_f[np.where(clades==c)]
#         clade_cts_r = cts_r[np.where(clades==c)]
#         clade_tot_f = tot_f[np.where(clades==c)]
#         clade_tot_r = tot_r[np.where(clades==c)]
        
#         clade_cts_pos = counts_CSS_pos[np.where(clades==c)]

#         # To have a frequency, a call must have > min_SNP pos with 
#         # nonzero counts in either fwd OR rev reads
#         cts2model = clade_cts_f + clade_cts_r
#         total2model = clade_tot_f + clade_tot_r

#         if np.count_nonzero(cts2model) > min_snps:
            
#             print(f"Fit results for clade: {clade_names[c]}")
                        
#             # Zero-inflated Poisson model
                
#             cts_lambda, cts_pi, _ = zip_fit(cts2model)
#             tot_lambda, tot_pi, _ = zip_fit(total2model)
            
#             print(f"MLE FIT: lambda={cts_lambda:.2f}, pi={cts_pi:.2f}")
#             # Only count clades with good probability
#             if cts_pi < max_pi:
#                 frequencies[c] = cts_lambda/tot_lambda
        
#         # Otherwise is zero
#         else:
#             print(f"clade: {clade_names[c]} does not have not enough csSNPs to model")
        
#         # Save data and fit information
#         save_cts_f.append( clade_cts_f )
#         save_cts_r.append( clade_cts_r )
#         save_tot_f.append( clade_tot_f )
#         save_tot_r.append( clade_tot_r )
#         save_cts_pos.append( clade_cts_pos )

                
#     # Output everything as nice data structures
#     frequencies[np.isnan(frequencies)] = 0
#     sample_frequencies = pd.DataFrame(frequencies)
#     sample_frequencies.columns = clade_names

#     data = [save_cts_f, save_cts_r, 
#             save_tot_f, save_tot_r,
#             save_cts_pos]
    
#     fit_info = 'placeholder'
#     # fit_info = [cts_lambda_arr, cts_pi_arr,
#     #             total_lambda_arr, total_pi_arr]

#     return data, fit_info, sample_frequencies

# def zip_pmf(x, pi, lambda_):
#     '''zero-inflated poisson function, pi is prob. of 0, lambda_ is the fit parameter'''
#     if pi < 0 or pi > 1 or lambda_ <= 0:
#         return np.zeros_like(x)
#     else:
#         return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)

# class ZeroInflatedPoisson(GenericLikelihoodModel):
#     def __init__(self, endog, exog=None, **kwds):
#         if exog is None:
#             exog = np.zeros_like(endog)
            
#         super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
#     def nloglikeobs(self, params):
#         pi = params[0]
#         lambda_ = params[1]

#         return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
#     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
#         if start_params is None:
#             lambda_start = self.endog.mean()
#             excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
#             pi_start = excess_zeros if excess_zeros>0 else 0
#             start_params = np.array([pi_start, lambda_start])
            
#         return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
#                                                     maxiter=maxiter, maxfun=maxfun, **kwds)

# def zip_fit(data):
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         model = ZeroInflatedPoisson(data)
#         fit = model.fit()
#         pi, lambda_ = fit.params
    
#     # return lambda_, pi
#         CI_pi, CI_lambda_ = fit.conf_int()
#         # range_CI_pi = CI_pi[1] - CI_pi[0]
#         range_CI_lambda_ = CI_lambda_[1] - CI_lambda_[0]
    
#     return lambda_, pi, range_CI_lambda_

# def plot_chain(counts, chain, 
#                accept_rate, lnprob,
#                nburn=2000):
    
#     # Burn initial samples    
#     chain_burn = chain[nburn:]
#     accept_rate_burn = accept_rate[nburn:]
#     lnprob_burn = lnprob[nburn:]

#     # Bin chain parameters
#     lambda_bins=np.arange(0,max(chain_burn[:,0])+0.01,0.01)
#     lambda_bincount = np.bincount(np.digitize(chain_burn[:,0], bins=lambda_bins))
    
#     pi_bins=np.arange(0,1,0.01)
#     pi_bincount = np.bincount(np.digitize(chain_burn[:,1], bins=pi_bins))

#     # PLOT
#     fig, axs = plt.subplots(4,2)
    
#     # 2D histogram of draws
#     x_bins=np.arange(min(chain_burn[:,0])-0.5,max(chain_burn[:,0])+0.5,0.01)
#     y_bins=np.arange(0,1,0.01)
#     axs[0,0].hist2d(chain_burn[:,0], chain_burn[:,1], bins = [x_bins, y_bins], cmap='inferno')
#     axs[0,0].set_xlabel('Lambda'); axs[0,0].set_ylabel('Pi')
#     axs[0,0].set_title('2D Histogram of draws')
    
#     # Histogram of counts
#     axs[0,1].hist(counts,bins=np.arange(0,max(counts)+2),color='k', alpha=0.5)
#     axs[0,1].set_title('Histogram of counts')
#     axs[0,1].legend()
    
#     # Lambda posterior estimate marginalized over pi
#     axs[1,0].plot(lambda_bins[:len(lambda_bincount)], lambda_bincount/np.sum(lambda_bincount), color='b', label='MCMC')
#     axs[1,0].set_xlabel('Lambda'); axs[1,1].set_ylabel('Probability')
#     axs[1,0].set_title('Lambda posterior pdf')
#     axs[1,0].axvline(chain[0,0], color='r', label='MLE Lambda estimate')
#     axs[1,0].axvline(np.mean(chain_burn[:,0]), color='k', label='MAP Lambda estimate')
#     axs[1,0].legend()

#     # Pi posterior estimate marginalized over lambda
#     axs[1,1].plot(np.arange(0,1,0.01)[:len(pi_bincount)], pi_bincount/np.sum(pi_bincount), color='g', label='MCMC')
#     axs[1,1].set_xlabel('Pi'); axs[1,1].set_ylabel('Probability')
#     axs[1,1].set_title('Pi posterior pdf')
#     axs[1,1].axvline(chain[0,1], color='r', label='MLE Pi estimate')
#     axs[1,1].axvline(np.mean(chain_burn[:,1]), color='k', label='MAP Pi estimate')
#     axs[1,1].legend()    

#     # Lambda trace plot
#     axs[2,0].plot(np.arange(0,nburn), chain[:nburn,0], linewidth=0.3, color='r')
#     axs[2,0].plot(np.arange(nburn,len(chain)), chain_burn[:,0], linewidth=0.3, color='b')
#     axs[2,0].set_xlabel('Iteration number'); axs[2,0].set_ylabel('Lambda')
#     # axs[2,0].set_xlim(2700,2900)
#     axs[2,0].set_title('Lambda trace plot')
#     # Pi trace plot
#     axs[2,1].plot(np.arange(0,nburn), chain[:nburn,1], linewidth=0.3, color='r')
#     axs[2,1].plot(np.arange(nburn,len(chain)), chain_burn[:,1],linewidth=0.3, color='g')
#     axs[2,1].set_xlabel('Iteration number'); axs[2,1].set_ylabel('Pi')
#     axs[2,1].set_title('Pi trace plot')
#     # axs[2,1].set_xlim(2700,2900)
    
#     # Acceptance Rate
#     axs[3,0].plot(np.arange(0,len(accept_rate)), accept_rate, color='k', linewidth=0.3)
#     axs[3,0].plot(np.arange(0,nburn), accept_rate[:nburn], color='r', linewidth=0.3)
#     axs[3,0].set_xlabel('Iteration number'); axs[3,0].set_ylabel('Acceptance rate')
#     axs[3,0].set_title('Acceptance Rate')
#     # Log-posterior
#     axs[3,1].plot(np.arange(0,len(lnprob)), lnprob, color='k', linewidth=0.3)
#     axs[3,1].plot(np.arange(0,nburn), lnprob[:nburn], color='r', linewidth=0.3)
#     axs[3,1].set_xlabel('Iteration number'); axs[3,1].set_ylabel('Log-posterior')
#     axs[3,1].set_title('Log-posterior plot')

#     return fig