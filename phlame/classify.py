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
import tempfile
import subprocess
import shlex 
import contextlib
import io

from scipy import stats
from scipy.optimize import minimize 
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import digamma, loggamma

import phlame.helper_functions as helper

#%%

# Limit for exponentials to avoid overflow
EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0
        
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
    
    path_to_frequencies (str): Path to output frequencies file.\n
    
    path_to_data (str, optional): If True, outputs a data file 
    with counts and modeling information at the defined level.
    Defaults to False.\n
    
    max_perc_diff (TYPE, optional): Maximum . Defaults to 0.3.\n
    
    max_snp_diff (TYPE, optional): If True, . Defaults to False.\n
    
    min_snps (int, optional): Minimum number of csSNPs with >0 counts
    to call a lineage as present. Defaults to 10.\n
    
    '''
    
    def __init__(self,
                 path_to_bam,
                 path_to_classifier,
                 ref_file,
                 path_to_frequencies,
                 path_to_cts_file=None,
                 path_to_pileup=None,
                 level_input=False,
                 path_to_data=False,
                 mode='mle',
                 min_snps=10, max_pi=0.3, min_prob=0.5, min_hpd=0.1,
                 nchain=10000, perc_burn=0.1, seed=False, verbose=True):

        self.__path_to_bam_file = path_to_bam
        self.__path_to_pileup = path_to_pileup

        self.__path_to_counts_file = path_to_cts_file
        self.__ref_file = ref_file
        self.__classifier_file = path_to_classifier
        self.__levels_input = level_input
        self.__output_freqs_file = path_to_frequencies
        self.__output_data_file = path_to_data

        self.max_pi = max_pi
        self.min_snps = min_snps
        self.min_prob = min_prob
        self.min_hpd = min_hpd

        self.nchain = nchain
        self.perc_burn = perc_burn
        self.seed = seed

        self.mode = mode

    def main(self):
        
        # =====================================================================
        #  Load Data
        # =====================================================================
        print("Reading in file(s)...")

        self.load_classifier()

        self.load_data()
    
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


    def get_positions(self,
                      output_chrpos_file):
        ''' From a classifier object, produce a samtools compatible list of positions
            that are informative to calling clades

        Args:
            path_to_classifier (str): Single path to input classifier file, or path 
                                    to directory with multiple classifier files.\n
            output_allpos_file (str): Path to file giving positions as they appear in
                                    the classifier object (NOT samtools compatible).\n 
            output_chrpos_file (str): Path to samtools compatible list of positions.
            refgenome_folder (str): DESCRIPTION.

        Raises:
            Exception: DESCRIPTION.

        Returns:
            None.

        '''
        cat_pos = np.array([], dtype=np.int32)
        chr_starts, _, scaf_names = helper.genomestats(self.__ref_file)
        
        path_to_cfrs_ls = []
        # Parse whether file or directory of files
        if os.path.isdir(self.__classifier_file):
            
            for filename in os.listdir(self.__classifier_file):
                
                if filename.endswith('.classifier'):
                    path_to_cfrs_ls.append(self.__classifier_file+'/'+filename)
        else:
            path_to_cfrs_ls.append(self.__classifier_file)
            
        
        # Get data from each classifier
        for cfr in path_to_cfrs_ls:
            
            with gzip.open(cfr,'rb') as f:
                csSNPs = pickle.load(f)
                                
                pos = csSNPs['cssnp_pos']
                cat_pos = np.concatenate([pos,cat_pos])

        allpos = np.unique(np.sort(cat_pos))
        
        print(f"{len(allpos)} informative positions found across {len(path_to_cfrs_ls)} classifier(s)")
        
        chr_pos = helper.p2chrpos(allpos, chr_starts)
        chr_names = np.array([scaf_names[i-1] for i in chr_pos[:,0]])
        chr_pos_final = np.vstack((chr_names,chr_pos[:,1])).T
        
        #save as samtools compatible txt file
        # np.savetxt(output_allpos_file, allpos, fmt='%i')
        np.savetxt(output_chrpos_file, chr_pos_final, delimiter='\t', fmt='%s')
        
        return


    def load_classifier(self):
        '''
        Load in classifier and level information
        '''
        
        self.classifier = helper.PhlameClassifier.read_file(self.__classifier_file)
        
        if self.__levels_input:
            self.mylevel = PhyloLevel(self.__levels_input, 
                                      self.classifier.clades, 
                                      self.classifier.clade_names)
            
            # Grab just information for the specific level
            self.level_cfr = self.classifier.grab_level(self.mylevel)
        
        else:
            self.level_cfr = self.classifier

    def load_data(self):

        with tempfile.TemporaryDirectory() as temp_dir:
        
            chrpositions_file = shlex.quote(os.path.join(temp_dir, 'chrpositions.txt'))
            pileup_file = shlex.quote(os.path.join(temp_dir, 'temp.pileup'))
            
            self.get_positions(chrpositions_file)
            
            print("Running samtools mpileup at informative positions...")
            print(f"samtools mpileup -q30 -x -s -O -d3000 "+ \
                           f"-l {chrpositions_file} "+ \
                           f"{shlex.quote(self.__ref_file)} "+ \
                           f"{shlex.quote(self.__path_to_bam_file)} > {pileup_file}")
            
            subprocess.run(f"samtools mpileup -q30 -x -s -O -d3000 "+ \
                           f"-l {chrpositions_file} "+ \
                           f"-f {shlex.quote(self.__ref_file)} "+ \
                           f"{shlex.quote(self.__path_to_bam_file)} > {pileup_file}", shell=True)
            # pileup_file.flush()  # Ensure data is written

            self.countsmat = helper.CountsMat(pileup_file,
                                            self.__ref_file,
                                            self.__classifier_file)
            self.countsmat.main()

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
        save_pi=np.full(nclades,-1,dtype=np.float64)
        save_counts_MLE=np.full((nclades,2),-1,dtype=np.float64)
        save_total_MLE=np.full((nclades,2),-1,dtype=np.float64)

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
                
                fit = countsCSS_NEW(cts2model,
                                    total2model,
                                    seed=self.seed,
                                    mode=self.mode)
                                                    
                frequency, prob = fit.fit(max_pi = self.max_pi,
                                          nchain = self.nchain,
                                          nburn = int(self.perc_burn*self.nchain))

                # print(f"MLE fit: cts lambda={fit.counts_MLE[0]:.2f} cts pi={fit.counts_MLE[1]:.2f}")
                # print(f"total lambda={fit.total_MLE[0]:.2f} total pi={fit.total_MLE[1]:.2f}")
                # print(f"HPD interval: {fit.hpd[0]:.2f}-{fit.hpd[1]:.2f}")
                
                # Only count clades if pass probability and HPD thresholds
                if (prob > self.min_prob) & \
                    (fit.hpd[0] <= self.min_hpd) & \
                        (ratio_nonzero_zero > 0.03):

                    frequencies[c] = fit.frequency
                    
                if fit.measured_alpha > 100000:
                    save_flag_highalpha[c] = True

                # Save fit information
                save_hpd.append(fit.hpd)
                save_cts_map.append(fit.counts_MAP)
                save_counts_MLE[c] = fit.counts_MLE
                save_total_MLE[c] = fit.total_MLE
                save_prob[c] = prob
                save_pi[c] = np.round(fit.pi,4)
                # print(fit.counts_MAP.keys())
                
                # save_chain.append(fit.chain)

            # Otherwise is zero
            else:
                print(f"Clade: {clade_names[c]} ",
                      "does not have not enough SNPs to model")
                # save_chain.append({})
                save_hpd.append(np.array((-1,-1)))
                save_cts_map.append({})

            # Save counts data
            save_cts.append( byclade_cts )
            save_cts_pos.append( byclade_cts_pos )
            
        frequencies_df = pd.DataFrame({'Relative abundance':frequencies,
                                       'DVb':save_pi,
                                       'Probability score':save_prob},
                                      index=self.level_cfr.clade_names)

        self.frequencies = frequencies_df
        self.data = {'clade_counts':save_cts,
                     'clade_counts_pos':save_cts_pos}
        self.fit_info = {'counts_MLE': save_counts_MLE,
                         'total_MLE':save_total_MLE,
                         'counts_MAP':save_cts_map,
                         'chain':[],
                         'prob':save_prob,
                         'hpd':save_hpd}
    
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

class countsCSS_NEW:
    '''
    Hold and model counts data covering a single set of cluster-specific SNPs.
    '''
    
    def __init__(self, 
                 counts, total_counts,
                 force_alpha=False,
                 prior=True,
                 prior_strength=20,
                 seed=False,
                 mode='bayesian'):
        
        self.counts = counts
        self.total_counts = total_counts
                
        # Maximum Likelihood fit
        with contextlib.redirect_stdout(io.StringIO()):
            self.counts_MLE = self.zip_fit_mle(self.counts)
            self.total_MLE = self.zip_fit_mle(self.total_counts)
        
        self.force_alpha = force_alpha

        self.prior_strength = prior_strength
        
        self.seed = seed

        self.mode = mode

        # Set hyperparameters
        if force_alpha:
            self.measured_alpha = force_alpha
            
        else:
            self.measured_alpha = np.mean(self.total_counts)**2/max(1e-6,np.var(self.total_counts)-np.mean(self.total_counts))

        m = self.prior_strength # Prior strength; higher values = stronger prior
        logp = m * digamma(self.measured_alpha)
        v = 0; s = 0
        self.params = [m, logp, v, s]

        if not prior:
            self.params = [0, np.log(1), 0, 0]

    def fit(self, max_pi,
            nchain=10000, 
            nburn=500,
            interval_size=0.95,
            subsample=True
            ):
        '''
        Fit counts data to model.
        '''
        
        if self.seed:
            np.random.seed(self.seed)
        
        # =====================================================================
        #  Maximum Likelihood
        # =====================================================================
        if self.mode == 'mle':
        
            lambda_MLE, pi_MLE = self.ZINB_MLE()

            self.counts_MLE = (pi_MLE,lambda_MLE)
            
            self.hpd = np.array([-1,-1])

            self.counts_MAP = {}

            self.prob = int(pi_MLE < max_pi)

            self.frequency = (lambda_MLE/self.total_MLE[0])

            self.pi = pi_MLE


        # =====================================================================
        #  Gibbs sampling
        # =====================================================================
        elif self.mode == 'bayesian':

            lambda_MLE, pi_MLE = self.ZINB_MLE()

            self.counts_MLE = (pi_MLE,lambda_MLE)

            if subsample & (len(self.counts) > 1000):
                self.counts, self.total_counts = self.subsample_positions(1000)
                
            try:
                param_chains, lv_chains = self.ZINB_gibbs_sampler(nchain,
                                                                nburn)
                
            except RuntimeError:
                
                print('Loglikelihood became too low and triggered an overflow error.\
                    Randomly subsampling positions and trying again...')
                
                # subsample_idx = np.random.choice(np.arange(0,len(self.counts)),
                #                                 int(len(self.counts)/2))
                
                # self.counts = self.counts[subsample_idx]
                # self.total_counts = self.total_counts[subsample_idx]

                self.counts, self.total_counts = self.subsample_positions(int(len(self.counts)/2))
                
                param_chains, lv_chains = self.ZINB_gibbs_sampler(nchain,
                                                                nburn)

            
            self.chain = {'pi':param_chains[:,0],
                        'a':param_chains[:,1],
                        'b':param_chains[:,2]}

            self.counts_MAP = {'pi':self.calc_MAP(self.chain['pi']),
                            'a':self.calc_MAP(self.chain['a']),
                            'b':self.calc_MAP(self.chain['b'])}
            
            self.hpd = self.get_hpd(self.chain['pi'], interval_size)
            self.prob = np.sum(self.chain['pi'] < max_pi)/len(self.chain['pi'])

            self.frequency = ((self.counts_MAP['a']/self.counts_MAP['b'])/self.total_MLE[0])

            self.pi = self.calc_MAP(self.chain['pi'])

        else:
            raise ValueError('Mode must be either "mle" or "bayesian"')
            
        return self.frequency, self.prob

    def ZINB_MLE(self):
        
        init = (np.log(np.mean(self.counts)), 0)
        result = minimize(self.zinb_nloglike, init, 
                          args=(self.counts,self.measured_alpha,))

        # Output the optimized parameters
        lambda_MLE, pi_MLE = result.x
        # lambda_ests.append(np.exp(lambda_est))
        # pi_ests.append(1 / (1 + np.exp(-pi_est)))
        # Convert back to bounded values
        return np.exp(lambda_MLE), 1 / (1 + np.exp(-pi_MLE))

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

    def subsample_positions(self, n):
        '''
        Subsample n positions from the counts data.
        '''
        
        idx = np.random.choice(np.arange(0,len(self.counts)), n)
        
        return self.counts[idx], self.total_counts[idx]
        
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
    def zinb_nloglike(params, data, alpha_prior):
        
        # Unpack the real-valued input params
        x_lambda, x_pi = params
        
        # Apply transformations to get the actual parameters
        lambd = np.exp(x_lambda)  # Lambda must be positive
        pi = 1 / (1 + np.exp(-x_pi))  # Pi must be in the interval (0, 1)
        
        # Calculate the probability of success p using lambda and alpha
        p = alpha_prior / (lambd + alpha_prior)
        
        # Create a boolean mask for zero observations
        zero_mask = (data == 0)
        
        # Calculate the Negative Binomial PMF for all y
        nb_pmf = stats.nbinom.pmf(data, alpha_prior, p)
        
        # Compute log likelihood for zeros
        log_likelihood_zeros = np.log(pi + (1 - pi) * nb_pmf)  # Log for the zeros
        
        # Compute log likelihood for non-zeros
        log_likelihood_nonzeros = np.log((1 - pi) * nb_pmf)  # Log for the non-zeros
        
        # Apply the appropriate log likelihood based on whether y = 0 or y > 0
        log_likelihood = np.sum(zero_mask * log_likelihood_zeros + (1 - zero_mask) * log_likelihood_nonzeros)
        
        return -log_likelihood

    @staticmethod
    def calc_MAP(chain, bins=False):

        if not bins:
            bins = 100

        counts, bins = np.histogram(chain, bins=bins)
        max_idx = np.argmax(counts)
        
        return bins[max_idx]
    
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
