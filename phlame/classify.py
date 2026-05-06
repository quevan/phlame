#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov  7 13:39:50 2022
Modified for Parallel Execution 2026 by John Connolly

@author: evanqu
"""

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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from scipy import stats
from scipy.optimize import minimize 
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import digamma, loggamma

import phlame.helper_functions as helper

# Limit for exponentials to avoid overflow
EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0

# =====================================================================
#  Parallel Worker Function
# =====================================================================

def fit_clade_worker(clade_args):
    """
    Standalone worker function to process a single clade in a separate process.
    """
    (clade_name, byclade_cts, mode, seed, nchain, perc_burn, 
     max_pi, min_snps, min_prob, min_hpd, informative_pos_subset) = clade_args
    
    cts2model = byclade_cts[0] + byclade_cts[1]
    total2model = byclade_cts[2] + byclade_cts[3]

    # Initialize results dictionary
    res = {
        'name': clade_name, 'freq_raw': 0.0, 'freq_thresholded': 0.0, 'prob': -1.0, 'pi': -1.0,
        'pi_threshold_for_presence': np.nan,
        'hpd': np.array([-1, -1]), 'map': {}, 'chain': None,
        'mle_cts': np.array([-1, -1]), 'mle_tot': np.array([-1, -1]),
        'high_alpha': False, 'cts_data': byclade_cts, 'cts_pos': informative_pos_subset
    }

    # Modeling Check: minimum SNPs
    if np.count_nonzero(cts2model) >= min_snps:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_nonzero_zero = np.true_divide(np.count_nonzero(cts2model > 0),
                                                np.count_nonzero(cts2model == 0))
            if ratio_nonzero_zero == np.inf:
                ratio_nonzero_zero = 1.0
        
        # Perform Fit
        fit = countsCSS_NEW(cts2model, total2model, seed=seed, mode=mode)
        frequency, prob = fit.fit(max_pi=max_pi, nchain=nchain, nburn=int(perc_burn * nchain))

        # Always keep the modeled abundance for downstream cutoff tuning.
        res['freq_raw'] = frequency

        # Preserve legacy thresholded abundance as a separate value.
        if (prob > min_prob) and (fit.hpd[0] <= min_hpd) and (ratio_nonzero_zero > 0.03):
            res['freq_thresholded'] = frequency
        
        res['prob'] = prob
        res['pi'] = np.round(fit.pi, 4)
        res['pi_threshold_for_presence'] = np.round(fit.pi, 4)
        res['hpd'] = fit.hpd
        res['map'] = fit.counts_MAP
        res['chain'] = fit.chain
        res['mle_cts'] = fit.counts_MLE
        res['mle_tot'] = fit.total_MLE
        res['high_alpha'] = fit.measured_alpha > 100000

    return res

# =====================================================================
#  Main Controller Class
# =====================================================================

class Classify:
    def __init__(self, path_to_bam, path_to_classifier, ref_file, path_to_frequencies,
                 level_input=False, path_to_data=False, mode='mle',
                 min_snps=10, max_pi=0.3, min_prob=0.5, min_hpd=0.1,
                 nchain=10000, perc_burn=0.1, seed=False, verbose=True):

        self.__path_to_bam_file = path_to_bam
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
        print("Reading in file(s)...")
        self.file_check()
        self.load_classifier()
        self.load_data()
    
        print("Sorting counts information...")
        self.index_counts()
        self.get_allele_counts()
        self.get_counts_alpha()

        print("Modeling counts in parallel...")
        self.calc_frequencies()
        self.save_frequencies()

    def file_check(self):
        if not os.path.exists(self.__path_to_bam_file):
            raise FileNotFoundError(f"Cannot find the file: {self.__path_to_bam_file} !")
        if not os.path.exists(self.__ref_file):
            raise FileNotFoundError(f"Cannot find the reference genome file: {self.__ref_file} !")

    def get_positions(self, output_chrpos_file):
        cat_pos = np.array([], dtype=np.int32)
        chr_starts, _, scaf_names = helper.genomestats(self.__ref_file)
        
        path_to_cfrs_ls = []
        if os.path.isdir(self.__classifier_file):
            for filename in os.listdir(self.__classifier_file):
                if filename.endswith('.classifier'):
                    path_to_cfrs_ls.append(os.path.join(self.__classifier_file, filename))
        else:
            path_to_cfrs_ls.append(self.__classifier_file)
            
        for cfr in path_to_cfrs_ls:
            with gzip.open(cfr, 'rb') as f:
                csSNPs = pickle.load(f)
                pos = csSNPs['cssnp_pos']
                cat_pos = np.concatenate([pos, cat_pos])

        allpos = np.unique(np.sort(cat_pos))
        print(f"{len(allpos)} informative positions found across {len(path_to_cfrs_ls)} classifier(s)")
        
        chr_pos = helper.p2chrpos(allpos, chr_starts)
        chr_names = np.array([scaf_names[i-1] for i in chr_pos[:,0]])
        chr_pos_final = np.vstack((chr_names, chr_pos[:,1])).T
        np.savetxt(output_chrpos_file, chr_pos_final, delimiter='\t', fmt='%s')

    def load_classifier(self):
        self.classifier = helper.PhlameClassifier.read_file(self.__classifier_file)
        if self.__levels_input:
            self.mylevel = PhyloLevel(self.__levels_input, self.classifier.clades, self.classifier.clade_names)
            self.level_cfr = self.classifier.grab_level(self.mylevel)
        else:
            self.level_cfr = self.classifier

    def load_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chrpositions_file = shlex.quote(os.path.join(temp_dir, 'chrpositions.txt'))
            pileup_file = shlex.quote(os.path.join(temp_dir, 'temp.pileup'))
            self.get_positions(chrpositions_file)
            
            print("Running samtools mpileup...")
            subprocess.run(f"samtools mpileup -q30 -x -s -O -d3000 " +
                           f"-l {chrpositions_file} " +
                           f"-f {shlex.quote(self.__ref_file)} " +
                           f"{shlex.quote(self.__path_to_bam_file)} > {pileup_file}", shell=True)

            self.countsmat = helper.CountsMat(pileup_file, self.__ref_file, self.__classifier_file)
            self.countsmat.main()

    def index_counts(self):
        informative_bool = np.nonzero(self.level_cfr.csSNPs)[0]
        self.informative_pos = self.level_cfr.csSNP_pos[informative_bool]
        self.informative_pos_idx = np.arange(0, len(self.countsmat.pos))[np.isin(self.countsmat.pos, self.informative_pos)][informative_bool]
        
        if np.sum(np.isin(self.countsmat.pos, self.informative_pos)) != len(np.unique(self.informative_pos)):
            raise Exception('Counts and classifier positions do not match.')

    def get_allele_counts(self):
        np.seterr(divide='ignore', invalid='ignore')
        counts = self.countsmat.counts
        counts_idx = self.informative_pos_idx
        alleles = self.level_cfr.alleles
        
        cts_f, cts_r = np.zeros(len(counts_idx)), np.zeros(len(counts_idx))
        tot_f, tot_r = np.zeros(len(counts_idx)), np.zeros(len(counts_idx))
        
        for i, p in enumerate(self.informative_pos):
            cts_f[i] = counts[counts_idx[i], int(alleles[i]-1)]
            cts_r[i] = counts[counts_idx[i], int(alleles[i]+3)]
            tot_f[i] = np.sum(counts[counts_idx[i], :4])
            tot_r[i] = np.sum(counts[counts_idx[i], 4:])
            
        self.allele_counts = (cts_f, cts_r, tot_f, tot_r)

    def get_counts_alpha(self):
        counts_sum = np.sum(self.countsmat.counts, axis=1)
        self.measured_alpha = np.mean(counts_sum)**2 / max(1e-6, np.var(counts_sum) - np.mean(counts_sum))
    
    def calc_frequencies(self):
        clade_names = self.level_cfr.clade_names
        nclades = len(clade_names)
        
        # Prepare parallel tasks
        tasks = []
        for c in range(nclades):
            byclade_cts = self.reshape_byclade(self.allele_counts, self.level_cfr.allele_cidx, c)
            byclade_pos = self.informative_pos[np.where(self.level_cfr.allele_cidx == c)]
            
            task_args = (
                clade_names[c], byclade_cts, self.mode, self.seed, 
                self.nchain, self.perc_burn, self.max_pi, self.min_snps, 
                self.min_prob, self.min_hpd, byclade_pos
            )
            tasks.append(task_args)

        # Execute modeling in parallel
        # Note: Using all available CPUs. You can set max_workers=X to limit cores.
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(fit_clade_worker, tasks))

        # Reassemble results
        freq_raw_list, freq_thresholded_list, prob_list, pi_list, pi_thresh_list = [], [], [], [], []
        save_cts, save_cts_pos, save_chain, save_hpd, save_cts_map = [], [], [], [], []
        save_counts_MLE = np.zeros((nclades, 2))
        save_total_MLE = np.zeros((nclades, 2))

        for i, res in enumerate(results):
            freq_raw_list.append(res['freq_raw'])
            freq_thresholded_list.append(res['freq_thresholded'])
            prob_list.append(res['prob'])
            pi_list.append(res['pi'])
            pi_thresh_list.append(res['pi_threshold_for_presence'])
            save_hpd.append(res['hpd'])
            save_cts_map.append(res['map'])
            save_chain.append(res['chain'])
            save_counts_MLE[i] = res['mle_cts']
            save_total_MLE[i] = res['mle_tot']
            save_cts.append(res['cts_data'])
            save_cts_pos.append(res['cts_pos'])

        self.frequencies = pd.DataFrame({
            'Relative abundance pre-threshold': freq_raw_list,
            'Relative abundance': freq_thresholded_list,
            'DVb': pi_list,
            'Pi threshold for presence (max_pi > this)': pi_thresh_list,
            'Probability score': prob_list
        }, index=clade_names)

        self.data = {'clade_counts': save_cts, 'clade_counts_pos': save_cts_pos}
        self.fit_info = {
            'counts_MLE': save_counts_MLE, 'total_MLE': save_total_MLE,
            'counts_MAP': save_cts_map, 'chain': save_chain,
            'prob': np.array(prob_list), 'hpd': save_hpd,
            'mode': self.mode, 'coverage': self.countsmat.coverage
        }
    
    def save_frequencies(self):
        self.frequencies.to_csv(self.__output_freqs_file, sep=',')
        if self.__output_data_file:
            with gzip.open(self.__output_data_file, 'wb') as f:
                pickle.dump([self.data, self.fit_info], f)

    @staticmethod
    def reshape_byclade(allele_counts, clade_idxs, i):
        clade_cts_f = allele_counts[0][np.where(clade_idxs==i)]
        clade_cts_r = allele_counts[1][np.where(clade_idxs==i)]
        clade_tot_f = allele_counts[2][np.where(clade_idxs==i)]
        clade_tot_r = allele_counts[3][np.where(clade_idxs==i)]
        return (clade_cts_f, clade_cts_r, clade_tot_f, clade_tot_r)

# =====================================================================
#  Statistical Modeling Class
# =====================================================================

class countsCSS_NEW:
    def __init__(self, counts, total_counts, force_alpha=False, prior=True, prior_strength=20, seed=False, mode='bayesian'):
        self.counts = counts
        self.total_counts = total_counts
        self.force_alpha = force_alpha
        self.prior_strength = prior_strength
        self.seed = seed
        self.mode = mode

        # MLE for initialization
        with contextlib.redirect_stdout(io.StringIO()):
            self.counts_MLE = self.zip_fit_mle(self.counts)
            self.total_MLE = self.zip_fit_mle(self.total_counts)
        
        if force_alpha:
            self.measured_alpha = force_alpha
        else:
            self.measured_alpha = np.mean(self.total_counts)**2 / max(1e-6, np.var(self.total_counts) - np.mean(self.total_counts))

        m = self.prior_strength
        logp = m * digamma(self.measured_alpha)
        self.params = [m, logp, 0, 0] if prior else [0, np.log(1), 0, 0]

    def fit(self, max_pi, nchain=10000, nburn=500, interval_size=0.95, subsample=True):
        if self.seed: np.random.seed(self.seed)
        
        if self.mode == 'mle':
            lambda_MLE, pi_MLE = self.ZINB_MLE()
            self.counts_MLE = (pi_MLE, lambda_MLE)
            self.hpd = np.array([-1, -1])
            self.counts_MAP = {}
            self.prob = int(pi_MLE < max_pi)
            self.frequency = (lambda_MLE / self.total_MLE[0])
            self.pi = pi_MLE
            self.chain = None

        elif self.mode == 'bayesian':
            lambda_MLE, pi_MLE = self.ZINB_MLE()
            self.counts_MLE = (pi_MLE, lambda_MLE)
            if subsample and (len(self.counts) > 1000):
                self.counts, self.total_counts = self.subsample_positions(1000)
            
            try:
                param_chains, _ = self.ZINB_gibbs_sampler(nchain, nburn)
            except RuntimeError:
                self.counts, self.total_counts = self.subsample_positions(int(len(self.counts)/2))
                param_chains, _ = self.ZINB_gibbs_sampler(nchain, nburn)

            self.chain = {'pi': param_chains[:, 0], 'a': param_chains[:, 1], 'b': param_chains[:, 2]}
            self.counts_MAP = {k: self.calc_MAP(v) for k, v in self.chain.items()}
            self.hpd = self.get_hpd(self.chain['pi'], interval_size)
            self.prob = np.sum(self.chain['pi'] < max_pi) / len(self.chain['pi'])
            self.frequency = ((self.counts_MAP['a'] / self.counts_MAP['b']) / self.total_MLE[0])
            self.pi = self.counts_MAP['pi']

        return self.frequency, self.prob

    def ZINB_MLE(self):
        init = (np.log(np.mean(self.counts)), 0)
        result = minimize(self.zinb_nloglike, init, args=(self.counts, self.measured_alpha,))
        return np.exp(result.x[0]), 1 / (1 + np.exp(-result.x[1]))

    def ZINB_gibbs_sampler(self, nchain, nburn):
        nparams = 3 
        npts = len(self.counts)    
        m, logp, v, s = self.params
        zero_idx = np.nonzero(self.counts == 0)[0]
        nonzero_idx = np.nonzero(self.counts)[0]    
        
        param_inits, latent_var_inits = self.initialize_chain()
        param_samples = np.zeros([nchain, nparams])
        param_samples[0] = param_inits
        latent_var_samples = np.zeros([nchain, 2, npts])
        latent_var_samples[0] = latent_var_inits
    
        for i in range(nchain-1):
            lambda_i = self.sample_lambda_i_conditional(param_samples[i,1], param_samples[i,2], self.counts, latent_var_samples[i,1])
            ri = self.sample_ri_conditional(npts, param_samples[i,0], lambda_i, zero_idx, nonzero_idx)
            pi = self.sample_p_conditional(ri, npts)
            b = max(1e-6, self.sample_b_conditional(param_samples[i,1], v, s, lambda_i, npts))
            a = self.sample_a_conditional_gp([lambda_i, b, v, m, logp], param_samples[i,1])[-1]
            
            param_samples[i+1] = [pi, a, b]
            latent_var_samples[i+1] = [lambda_i, ri]
        
        param_samples[:,0] = (1 - param_samples[:,0])
        return param_samples, latent_var_samples
    
    def initialize_chain(self):
        counts_mean = self.counts.mean()
        pi_init = min(1 - ((self.counts == 0).mean() - stats.poisson.pmf(0, counts_mean)), 0.99)
        a_init = self.measured_alpha
        b_init = a_init / counts_mean
        ri_init = np.int64(self.counts > 0)
        return [pi_init, a_init, b_init], [np.full(len(self.counts), counts_mean), ri_init]

    def subsample_positions(self, n):
        idx = np.random.choice(np.arange(0, len(self.counts)), n)
        return self.counts[idx], self.total_counts[idx]
        
    def zip_fit_mle(self, cts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ZeroInflatedPoisson(cts)
            fit = model.fit()
            return np.array([fit.params[1], fit.params[0]]) # lambda, pi
        
    @staticmethod
    def zinb_nloglike(params, data, alpha_prior):
        lambd = np.exp(params[0])
        pi = 1 / (1 + np.exp(-params[1]))
        p = alpha_prior / (lambd + alpha_prior)
        nb_pmf = stats.nbinom.pmf(data, alpha_prior, p)
        log_likelihood = np.sum(np.where(data == 0, np.log(pi + (1 - pi) * nb_pmf), np.log((1 - pi) * nb_pmf)))
        return -log_likelihood

    @staticmethod
    def calc_MAP(chain, bins=100):
        counts, bins = np.histogram(chain, bins=bins)
        return bins[np.argmax(counts)]
    
    @staticmethod
    def sample_lambda_i_conditional(a, b, counts, r_i):
        return stats.gamma.rvs(a=a+counts, scale=1/(r_i+b))
    
    @staticmethod
    def sample_p_conditional(ri, npts):
        return stats.beta.rvs(np.sum(ri)+1, npts - np.sum(ri) + 1)
    
    @staticmethod
    def sample_ri_conditional(npts, pi, lambda_i, zero_idx, nonzero_idx):
        ri = np.zeros(npts)
        pred_prob_zero = (1/(1+(np.var(lambda_i)/np.mean(lambda_i))))**(np.mean(lambda_i)**2/np.var(lambda_i))
        zero_bern = (pi*pred_prob_zero) / ((pi*pred_prob_zero)+(1-pi))
        if len(zero_idx) > 0: ri[zero_idx] = stats.bernoulli.rvs(zero_bern, size=len(zero_idx))
        if len(nonzero_idx) > 0: ri[nonzero_idx] = 1
        return ri
    
    @staticmethod
    def sample_a_conditional_gp(aparams, a_old):
        def log_a_conditional_dist_gp(a, params):
            li, b, v, m, lp = params
            li = li[li > 0]
            return (a*(len(li)+v)*np.log(b) + a*lp + a*np.sum(np.log(li)) - (m + len(li))*loggamma(a))
        return slice_sampler(x0=a_old, loglike=log_a_conditional_dist_gp, params=aparams, niter=50, sigma=a_old/10)
    
    @staticmethod
    def sample_b_conditional(a, v, s, data, npts):
        return stats.gamma.rvs(a=a*(npts + v), scale=1/(s + np.sum(data)))    
    
    @staticmethod
    def get_hpd(chain, interval_size=0.95):
        d = np.sort(np.copy(chain))
        n = len(chain)
        interval = np.floor(interval_size * n).astype(int)
        int_width = d[interval:] - d[:n-interval]
        min_int = np.argmin(int_width)
        return np.array([d[min_int], d[min_int+interval]])

def slice_sampler(x0, loglike, params, niter, sigma, step_out=True):
    samples = np.zeros(niter)
    xx = float(x0)
    last_llh = loglike(xx, params)
    for i in range(niter):
        llh0 = last_llh + np.log(np.random.rand())
        rr = np.random.rand()
        x_l, x_r = xx - rr * sigma, xx + (1 - rr) * sigma
        if step_out:
            while loglike(x_l, params) > llh0: x_l -= sigma
            while loglike(x_r, params) > llh0: x_r += sigma
        while True:
            xd = np.random.rand() * (x_r - x_l) + x_l
            last_llh = loglike(xd, params)
            if last_llh > llh0:
                xx = xd; break
            elif xd > xx: x_r = xd
            else: x_l = xd
        samples[i] = xx
    return samples

class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None: exog = np.zeros_like(endog)
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        return -np.log(self._zip_pmf(self.endog, pi=params[0], lambda_=params[1]))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            l_start = self.endog.mean()
            pi_start = max(0, (self.endog == 0).mean() - stats.poisson.pmf(0, l_start))
            start_params = np.array([pi_start, l_start])
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)
    
    @staticmethod
    def _zip_pmf(x, pi, lambda_):
        if pi < 0 or pi > 1 or lambda_ <= 0: return np.zeros_like(x)
        return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)

class PhyloLevel:
    def __init__(self, levelin, allclades, allclade_names):
        self.__allclades, self.__allclade_names = allclades, allclade_names
        if os.path.isfile(levelin):
            self.clades, self.clade_names, self.names = self._parse_file(levelin)
        else:
            self.clades, self.clade_names, self.names = self._parse_str(levelin)
            
    def _parse_file(self, levelin_file):
        with open(levelin_file, 'r') as f:
            line = f.readline()
        dlim = '\t' if '\t' in line else ','
        clade_ids = np.loadtxt(levelin_file, delimiter=dlim, dtype=str)
        clades, names = self._reshape(clade_ids[:,0], clade_ids[:,1], '-1')
        clade_names = self._check_file(clades, names)
        self._ancdesc(clades, clade_names)
        return clades, clade_names, names

    def _parse_str(self, levelin_str):
        clade_names = levelin_str.strip().split(',')
        clades = self._check_str(clade_names)
        self._ancdesc(clades, clade_names)
        return clades, clade_names, clade_names
    
    def _check_file(self, clades, names):
        clade_names = []
        all_c_vals = list(self.__allclades.values())
        all_c_keys = np.array(list(self.__allclades.keys()))
        for i, clade in enumerate(clades):
            match = [set(clade) == set(genomes) for genomes in all_c_vals]
            if not any(match): raise Exception(f"Clade {names[i]} not in classifier.")
            clade_names.append(str(all_c_keys[match][0]))
        return clade_names
    
    def _check_str(self, clade_names):
        clades = []
        for name in clade_names:
            if name not in self.__allclade_names: raise Exception(f"{name} invalid.")
            clades.append(self.__allclades[name])
        return clades
    
    @staticmethod
    def _ancdesc(clades, clade_names):
        for i, clade in enumerate(clades):
            cp = clades[:i] + clades[i+1:]
            if any([len(set(clade).intersection(set(g))) > 0 for g in cp]):
                raise Exception(f"{clade_names[i]} overlaps with another clade.")

    @staticmethod
    def _reshape(names_long, ids_long, uncl):
        clades, names = [], []
        for c in np.unique(ids_long):
            if c != uncl:
                names.append(str(c))
                clades.append(names_long[ids_long == c])
        return clades, names

# =====================================================================
#  Entry Point
# =====================================================================
if __name__ == "__main__":
    # Example usage:
    # classifier = Classify(...)
    # classifier.main()
    pass
