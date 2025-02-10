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
import seaborn as sns
import ete3
import skbio

import glob

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures')

import fig_helper_functions as helper

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

fmt={'fontsize':14,
     'fontname':'Helvetica'}

%matplotlib auto

#%% Functions

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

def outcome_score(accepted_genomes, output_genomes):

    if len(output_genomes) == 0:
        return 3
    elif np.all(np.in1d(output_genomes, accepted_genomes)):
        return 0
    elif np.any(np.in1d(output_genomes, accepted_genomes)):
        return 1
    else:
        return 2

def duplicate_substring(strings):
    result = []
    
    for s in strings:
        # Split the string to find the index part
        parts = s.split('_')
        
        # Identify the part that contains the index
        for i in range(len(parts)):
            if 'idx' in parts[i]:
                # Duplicate the index part
                parts[i] = parts[i] + '_' + parts[i]
        
        # Rejoin the parts into a single string
        new_string = '_'.join(parts)
        result.append(new_string)
    
    return result

def normalized_pairwise_distance(method_results, dropped_genomes,
                                 tree, true_genome,
                                 true_distance):

    if np.sum(method_results) == 0:
        return np.nan
    
    else:
        true_leaf = tree.search_nodes(name = true_genome)[0]

        pairwise_distance = []
        rabundance_ls = []
        for genome, rabundance in method_results.iteritems():

            if not np.isin(genome, dropped_genomes):

                genome_leaf = tree.search_nodes(name = genome)[0]
                pairwise_distance.append(true_leaf.get_distance(genome_leaf))
                rabundance_ls.append(rabundance)

        # Normalize to true_distance
        pairwise_distance = np.array(pairwise_distance)
        pairwise_distance_norm = pairwise_distance - true_distance

        #Normalize to relative abundance
        pairwise_distance_norm2 = pairwise_distance_norm * np.array(rabundance_ls)

        return np.sum(pairwise_distance_norm2)

def novel_diversity_score(species_name, tree, pusbs_df,
                          true_abundances,
                          strainest_out, strainge_out, strainscan_out):

    results = pd.DataFrame(columns = ['Clade','pusb', 
                                      'min_dist_sister', 'min_dist_other',
                                      'StrainEst_Score','StrainGE_Score', 'StrainScan_Score',
                                      'StrainEst_normalized_dist','StrainGE_normalized_dist', 'StrainScan_normalized_dist'])
    
    for idx, (pusb, cladename) in enumerate(zip(pusbs_df['pusb'],pusbs_df['sis1_name'])):
            
        if idx == 0:
            continue

        sample_name = f"10X_{species_name}_dropped_idx{idx}"
        sample_name_alt = f"{sample_name}_idx{idx}"

        # Get information about the clade that was dropped
        clade = tree.search_nodes(name = cladename)[0]
        sister = tree.search_nodes(name = cladename)[0].get_sisters()[0]

        dropped_genomes = sister.get_leaf_names() # Genomes that were dropped from the db
        accepted_genomes = clade.get_leaf_names() # Genomes that are "acceptable" as an answer (aka from the nearest clade)

        # Make sure that the true genome from the "metagenome" is from the dropped clade
        true_genome = true_abundances.loc[sample_name][true_abundances.loc[sample_name] > 0].index
        assert np.isin(true_genome, dropped_genomes).all()
        true_leaf = tree.search_nodes(name = true_genome)[0]

        # Get pairwise distance(s) between dropped genome and its closest sister
        pairwise_distance = []
        for genome in accepted_genomes:
            pairwise_distance.append(true_leaf.get_distance(genome))

        # Get pairwise distance(s) between dropped genome and all other genomes
        pairwise_distance_all = []
        for genome in tree.get_leaves():
            if not (np.isin(genome.name, accepted_genomes) | np.isin(genome.name, dropped_genomes)):
                pairwise_distance_all.append(true_leaf.get_distance(genome))
                # if true_leaf.get_distance(genome) < min(pairwise_distance):
                #     print(genome.name)

        assert len(pairwise_distance) + len(pairwise_distance_all) == len(tree.get_leaves()) - len(dropped_genomes)

        strainest_genomes = strainest_out.loc[sample_name][strainest_out.loc[sample_name] > 0].index
        strainge_genomes = strainge_out.loc[sample_name_alt][strainge_out.loc[sample_name_alt] > 0].index
        strainscan_genomes = strainscan_out.loc[sample_name_alt][strainscan_out.loc[sample_name_alt] > 0].index

        # Get pairwise distance(s) of each method normalized to the distance to the closest genome
        strainest_norm_dist = normalized_pairwise_distance(strainest_out.loc[sample_name], dropped_genomes,
                                                        tree, true_genome, np.min(pairwise_distance + pairwise_distance_all))
        strainge_norm_dist = normalized_pairwise_distance(strainge_out.loc[sample_name_alt], dropped_genomes,
                                                            tree, true_genome, np.min(pairwise_distance + pairwise_distance_all))
        strainscan_norm_dist = normalized_pairwise_distance(strainscan_out.loc[sample_name_alt], dropped_genomes,
                                                            tree, true_genome, np.min(pairwise_distance + pairwise_distance_all))

        results.loc[idx] = [cladename, pusb, 
                            np.min(pairwise_distance), np.min(pairwise_distance_all),
                            outcome_score(accepted_genomes, strainest_genomes),
                            outcome_score(accepted_genomes, strainge_genomes),
                            outcome_score(accepted_genomes, strainscan_genomes),
                            strainest_norm_dist, strainge_norm_dist, strainscan_norm_dist]
        
    return results


#%% Read in outputs from each method for C. acnes

# C. acnes

TRUE_DISTRIBUTIONS_FILE = 'fig1/Cacnes/true_community_abundances.csv'

path_to_samples_csv = 'fig1/Cacnes_samples.csv'
path_to_strainest_dir = 'fig1/Cacnes/StrainEst'
path_to_strainge_dir = 'fig1/Cacnes/StrainGE'
path_to_strainscan_dir = 'fig1/Cacnes/StrainScan'
path_to_clades_file = 'fig1/trees/Cacnes_megatree_GTR_isonames_cladeIDs_v2.txt'
reference_genome = 'Pacnes_C1'
detection_lim =  0.01

cacnes_sample_names = pd.read_csv(path_to_samples_csv)['Sample']
# cacnes_sample_names = duplicate_substring(cacnes_sample_names)[1:]

# Read in true frequencies
cacnes_true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

cacnes_strain_names = np.unique(pd.read_csv(path_to_clades_file, sep='\t', header=None)[0])

# Make the same format as the method outputs
cacnes_to_add = [name for name in cacnes_strain_names if name not in cacnes_true_abundances]
old_columns = set(cacnes_true_abundances.columns)
cacnes_true_abundances = cacnes_true_abundances.reindex(columns=old_columns.union(cacnes_to_add))
cacnes_true_abundances.fillna(0, inplace=True)

#StrainEst
cacnes_strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           duplicate_substring(cacnes_sample_names)[1:],
                                           cacnes_strain_names,
                                           reference_genome)
cacnes_strainest_out.fillna(0, inplace=True)
#StrainGE
cacnes_strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         duplicate_substring(cacnes_sample_names)[1:],
                                         cacnes_strain_names)
#StrainScan
cacnes_strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             duplicate_substring(cacnes_sample_names)[1:],
                                             cacnes_strain_names)


path_to_pusb_file = 'fig1/Cacnes/Cacnes_pusb.csv'
path_to_tree_file = 'fig1/trees/Cacnes_megatree_GTR_isonames_calls_v2.tre'
path_to_clades_file = 'fig1/trees/Cacnes_megatree_GTR_isonames_cladeIDs_v2.txt'

cacnes_tree = ete3.Tree(path_to_tree_file, format=1)
cacnes_cladeIDs = pd.read_csv(path_to_clades_file, sep='\t', header=None, index_col=0)
cacnes_pusbs = pd.read_csv(path_to_pusb_file, header=0, index_col=0)
cacnes_skbio_tree = skbio.TreeNode.read(path_to_tree_file, convert_underscores=False)

#%% C. acnes

cacnes_results = novel_diversity_score('Cacnes', cacnes_tree, cacnes_pusbs,
                                        cacnes_true_abundances,
                                        cacnes_strainest_out, cacnes_strainge_out, cacnes_strainscan_out)

#%% Fig 2B1: Read in outputs from each method for S. epi

# S. epi

TRUE_DISTRIBUTIONS_FILE = 'fig1/Sepi/true_community_abundances.csv'

path_to_samples_csv = 'fig1/Sepi_samples.csv'
path_to_strainest_dir = 'fig1/Sepi/StrainEst'
path_to_strainge_dir = 'fig1/Sepi/StrainGE'
path_to_strainscan_dir = 'fig1/Sepi/StrainScan'
path_to_clades_file = 'fig1/trees/Sepi_acera_norecomb_GTR_clades.txt'
reference_genome = 'SepidermidisATCC12228'
detection_lim =  0.01

sepi_sample_names = pd.read_csv(path_to_samples_csv)['Sample']

# Read in true frequencies
sepi_true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

sepi_strain_names = np.unique(pd.read_csv(path_to_clades_file, sep='\t', header=None)[0])

# Make the same format as the method outputs
sepi_to_add = [name for name in sepi_strain_names if name not in sepi_true_abundances]
old_columns = set(sepi_true_abundances.columns)
sepi_true_abundances = sepi_true_abundances.reindex(columns=old_columns.union(sepi_to_add))
sepi_true_abundances.fillna(0, inplace=True)

#StrainEst
sepi_strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           duplicate_substring(sepi_sample_names)[1:],
                                           sepi_strain_names,
                                           reference_genome)
sepi_strainest_out.fillna(0, inplace=True)
#StrainGE
sepi_strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         duplicate_substring(sepi_sample_names)[1:],
                                         sepi_strain_names)
#StrainScan
sepi_strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             duplicate_substring(sepi_sample_names)[1:],
                                             sepi_strain_names)


path_to_pusb_file = 'fig1/Sepi/Sepi_pusb.csv'
path_to_tree_file = 'fig1/trees/Sepi_acera_norecomb_GTR_calls.tre'
path_to_clades_file = 'fig1/trees/Sepi_acera_norecomb_GTR_clades.txt'

sepi_tree = ete3.Tree(path_to_tree_file, format=1)
sepi_cladeIDs = pd.read_csv(path_to_clades_file, sep='\t', header=None, index_col=0)
sepi_pusbs = pd.read_csv(path_to_pusb_file, header=0, index_col=0)
sepi_skbio_tree = skbio.TreeNode.read(path_to_tree_file, convert_underscores=False)

#%% S. epi

sepi_results = novel_diversity_score('Sepi', sepi_tree, sepi_pusbs,
                                     sepi_true_abundances,
                                     sepi_strainest_out, sepi_strainge_out, sepi_strainscan_out)

#%% E. coli

TRUE_DISTRIBUTIONS_FILE = 'fig1/Ecoli/true_community_abundances.csv'

path_to_samples_csv = 'fig1/Ecoli_samples.csv'
path_to_strainest_dir = 'fig1/Ecoli/StrainEst'
path_to_strainge_dir = 'fig1/Ecoli/StrainGE'
path_to_strainscan_dir = 'fig1/Ecoli/StrainScan'
path_to_clades_file = 'fig1/trees/Ecoli_GTR_cladeIDs.txt'
reference_genome = 'Ecoli_ASM584'
detection_lim =  0.01

Ecoli_sample_names = pd.read_csv(path_to_samples_csv)['Sample']

# Read in true frequencies
Ecoli_true_abundances = pd.read_csv(TRUE_DISTRIBUTIONS_FILE, header=0, index_col=0).T

Ecoli_strain_names = np.unique(pd.read_csv(path_to_clades_file, sep='\t', header=None)[0])

# Make the same format as the method outputs
Ecoli_to_add = [name for name in Ecoli_strain_names if name not in Ecoli_true_abundances]
old_columns = set(Ecoli_true_abundances.columns)
Ecoli_true_abundances = Ecoli_true_abundances.reindex(columns=old_columns.union(Ecoli_to_add))
Ecoli_true_abundances.fillna(0, inplace=True)

#StrainEst
Ecoli_strainest_out = helper.parse_strainest_out(path_to_strainest_dir,
                                           duplicate_substring(Ecoli_sample_names)[1:],
                                           Ecoli_strain_names,
                                           reference_genome)
Ecoli_strainest_out.fillna(0, inplace=True)
#StrainGE
Ecoli_strainge_out = helper.parse_strainge_out(path_to_strainge_dir,
                                         duplicate_substring(Ecoli_sample_names)[1:],
                                         Ecoli_strain_names)
#StrainScan
Ecoli_strainscan_out = helper.parse_strainscan_out(path_to_strainscan_dir,
                                             duplicate_substring(Ecoli_sample_names)[1:],
                                             Ecoli_strain_names)


path_to_pusb_file = 'fig1/Ecoli/Ecoli_pusb.csv'
path_to_tree_file = 'fig1/trees/Ecoli_GTR_calls.tre'
path_to_clades_file = 'fig1/trees/Ecoli_GTR_cladeIDs.txt'

Ecoli_tree = ete3.Tree(path_to_tree_file, format=1)
Ecoli_cladeIDs = pd.read_csv(path_to_clades_file, sep='\t', header=None, index_col=0)
Ecoli_pusbs = pd.read_csv(path_to_pusb_file, header=0, index_col=0)
Ecoli_skbio_tree = skbio.TreeNode.read(path_to_tree_file, convert_underscores=False)

#%%

Ecoli_results = novel_diversity_score('Ecoli', Ecoli_tree, Ecoli_pusbs,
                                        Ecoli_true_abundances,
                                        Ecoli_strainest_out, Ecoli_strainge_out, Ecoli_strainscan_out)

#%% Fig 1A: Tree vignette

# Good example is 10X_Cacnes_dropped_idx22 which is in phylogroup K
path_to_tree_file = 'fig1/tree_vignette.tre'
tree_K = ete3.Tree(path_to_tree_file, format=1)

sample_name = "10X_Cacnes_dropped_idx22"
strainest_vignette = cacnes_strainest_out.loc[sample_name][cacnes_strainest_out.loc[sample_name] > 0]
strainge_vignette = cacnes_strainge_out.loc[f"{sample_name}_idx22"][cacnes_strainge_out.loc[f"{sample_name}_idx22"] > 0]
strainscan_vignette = cacnes_strainscan_out.loc[f"{sample_name}_idx22"][cacnes_strainscan_out.loc[f"{sample_name}_idx22"] > 0]

# Add metadata to nodes

for leaf in tree_K.iter_leaves():
    
    if leaf.name in strainest_vignette.index:
        leaf.add_face(ete3.CircleFace(0.5,'orange'),
                      column=0,position = "branch-right")
    if leaf.name in strainge_vignette.index:
        leaf.add_face(ete3.CircleFace(0.5,'seagreen'),
                      column=0,position = "branch-right")
    if leaf.name in strainscan_vignette.index:
        leaf.add_face(ete3.CircleFace(0.5,'palevioletred'),
                      column=1,position = "branch-right")

#%% Plot vignette

# TreeStyle
general_ts = ete3.TreeStyle()
general_ts.show_leaf_name=False
general_ts.scale = 0.01
general_ts.scale_length = 1000

# Remove blue dot on every node
nstyle = ete3.NodeStyle()
nstyle['shape']=''; nstyle['size']=0
nstyle['vt_line_width']=0.1
nstyle['hz_line_width']=0.1

for node in tree_K.traverse():
   node.set_style(nstyle)

# Add legend
general_ts.legend.add_face(ete3.CircleFace(1, 'orange'), column=0)
general_ts.legend.add_face(ete3.TextFace("StrainEst", fsize=2), column=1)
general_ts.legend.add_face(ete3.CircleFace(1, 'seagreen'), column=0)
general_ts.legend.add_face(ete3.TextFace("StrainGE", fsize=2), column=1)
general_ts.legend.add_face(ete3.CircleFace(1, 'palevioletred'), column=0)
general_ts.legend.add_face(ete3.TextFace("StrainScan", fsize=2), column=1)

general_ts.legend_position=3

tree_K.show(tree_style=general_ts)

tree_K.render("fig1/fig1a1.pdf", tree_style=general_ts, w=20, units="in", dpi=300)


#%% Fig 1A: Plot vignette

fmt={'fontsize':14,
     'fontname':'Helvetica'}

clades = np.array(["C.2.1.1",'C.2.1.2.1','C.2.1.2.2.1.1'])

barplot_df = pd.concat([strainest_vignette, strainge_vignette, strainscan_vignette], axis=1)

ls = []
for genome in barplot_df.index:
    clade = cacnes_cladeIDs.loc[genome]
    ls.append(clades[np.in1d(clades,clade.values)][0])

barplot_df.index = ls

# Merge identical indices in df and sum values
barplot_df = barplot_df.groupby(level=0).sum()
barplot_df.columns = ['StrainEst','StrainGE','StrainScan']

fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [.3, 1]})

fig.set_size_inches(4.2, 4)

barplot_df['StrainEst'].T.plot.barh(ax=axs[0], stacked=True, edgecolor='k')
axs[0].set_ylim(.5,1.5)
axs[0].set_yticklabels(['','Truth\n(100% novel)',''], **fmt)
axs[0].xaxis.set_tick_params(labelsize=14)

barplot_df.T.plot.barh(ax=axs[1], stacked=True, edgecolor='k')
axs[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=14,
              labels=['C.1','C.2','C.3'])
axs[1].set_yticklabels(barplot_df.columns, **fmt)
axs[1].xaxis.set_tick_params(labelsize=14)
axs[1].set_xlabel('R. abundance', **fmt)

fig.tight_layout()

# fig.savefig('fig1/fig1a2.pdf', dpi=300, format='pdf')

#%%  Fig 1B: Plot the outcomes of each method        

import seaborn as sns
import scipy.stats as stats

fmt={'fontsize':14,
     'fontname':'Helvetica'}

fig, axs = plt.subplots(1,3)
fig.set_size_inches(8, 4)

for idx, (results_, species) in enumerate(zip([cacnes_results, sepi_results, Ecoli_results],
                                              ['C. acnes', 'S. epidermidis', 'E. coli'])):


    sns.stripplot(data=results_[['StrainEst_normalized_dist',
                                'StrainGE_normalized_dist',
                                'StrainScan_normalized_dist']],
                                color='k', ax=axs[idx], jitter=True, alpha=0.5, s=4)
    # Just the mean line
    sns.boxplot(data=results_[['StrainEst_normalized_dist',
                                'StrainGE_normalized_dist',
                                'StrainScan_normalized_dist']],
                                showmeans=True,
                                meanline=True,
                                meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                                medianprops={'visible': False},
                                whiskerprops={'visible': False},
                                zorder=10,
                                showfliers=False,
                                showbox=False,
                                showcaps=False,ax=axs[idx])

    axs[idx].set_ylabel('Excess distance from\nclosest reference (# SNVs)', **fmt)
    axs[idx].set_title(species, **fmt)
    axs[idx].set_yscale('symlog')
    axs[idx].set_xticklabels(['StrainEst','StrainGE','StrainScan'], **fmt, rotation=45)
    axs[idx].tick_params(axis='y', labelsize=12)  
    axs[idx].set_ylim(-0.1, 10e5)

fig.tight_layout()

# Kruskal-Wallis test between groups

for ridx_, results_ in enumerate([cacnes_results, sepi_results, Ecoli_results]):

    strainest_res = results_['StrainEst_normalized_dist'][results_['StrainEst_normalized_dist']>=0]
    strainge_res = results_['StrainGE_normalized_dist'][results_['StrainGE_normalized_dist']>=0]
    strainscan_res = results_['StrainScan_normalized_dist'][results_['StrainScan_normalized_dist']>=0]

    p = stats.kruskal(strainest_res, strainge_res, strainscan_res)

    axs[ridx_].text(1, 10e4, f"p={p.pvalue:.3f}", ha='center', va='bottom', fontsize=12)

    print(f"Median excess distance: {np.median(np.concatenate([strainest_res,strainge_res,strainscan_res]))}")
    

# How many simulations returned the exact closest reference
exact_closest_ls = []
for method in ['StrainEst_normalized_dist','StrainGE_normalized_dist','StrainScan_normalized_dist']:
    exact_closest = 0
    exact_closest += np.sum(cacnes_results[method] <= 0)
    exact_closest += np.sum(sepi_results[method] <= 0)
    exact_closest += np.sum(Ecoli_results[method] <= 0)
    exact_closest_ls.append(exact_closest)

nsims = len(sepi_results) + len(cacnes_results) + len(Ecoli_results)

print(f"% of simulations that returned exact closest reference (StrainEst): {1-(exact_closest_ls[0]/nsims):.3f}")
print(f"% of simulations that returned exact closest reference (StrainGE): {1-(exact_closest_ls[1]/nsims):.3f}")
print(f"% of simulations that returned exact closest reference (StrainScan): {1-(exact_closest_ls[2]/nsims):.3f}")

# fig.savefig('supplemental/figS1a.pdf', format='pdf', dpi=300)

#%% Compare how consistent method outputs are with each other
# Mean pairwise distance between method outputs versus true pairwise distance to closest genome

# C. acnes

cacnes_mean_pwdist = []
cacnes_min_dist_sister = []

for idx, sample_name in enumerate(cacnes_sample_names):

    if idx == 0:
        continue
    
    sample_name_alt = f"{sample_name}_idx{idx}"

    strainest_sample = cacnes_strainest_out.loc[sample_name].values
    strainge_sample = cacnes_strainge_out.loc[sample_name_alt].values
    strainscan_sample = cacnes_strainscan_out.loc[sample_name_alt].values

    strainest_present = (strainest_sample > 0).astype(int)
    strainge_present = (strainge_sample > 0).astype(int)
    strainscan_present = (strainscan_sample > 0).astype(int)
    
    # Get the unifrac distance between the genomes output by each method
    if np.sum(strainest_present) > 0 and np.sum(strainge_present) > 0 and np.sum(strainscan_present) > 0:
        x = skbio.diversity.beta.unweighted_unifrac(strainest_present, strainge_present,
                                                    cacnes_strain_names, cacnes_skbio_tree)
        y = skbio.diversity.beta.unweighted_unifrac(strainest_present, strainscan_present,
                                                    cacnes_strain_names, cacnes_skbio_tree)
        z = skbio.diversity.beta.unweighted_unifrac(strainge_present, strainscan_present,
                                                    cacnes_strain_names, cacnes_skbio_tree)

    # mean_pairwise_distance.append(np.mean([x,y,z]))
        cacnes_mean_pwdist.append(np.mean([x,y,z]))
        cacnes_min_dist_sister.append(cacnes_results.loc[idx]['min_dist_sister'])

# S. epi
sepi_mean_pwdist = []
sepi_min_dist_sister = []
for idx, sample_name in enumerate(sepi_sample_names):

    if idx == 0:
        continue
    
    sample_name_alt = f"{sample_name}_idx{idx}"

    strainest_sample = sepi_strainest_out.loc[sample_name].values
    strainge_sample = sepi_strainge_out.loc[sample_name_alt].values
    strainscan_sample = sepi_strainscan_out.loc[sample_name_alt].values

    strainest_present = (strainest_sample > 0).astype(int)
    strainge_present = (strainge_sample > 0).astype(int)
    strainscan_present = (strainscan_sample > 0).astype(int)
    
    # Get the unifrac distance between the genomes output by each method
    if np.sum(strainest_present) > 0 and np.sum(strainge_present) > 0 and np.sum(strainscan_present) > 0:
        x = skbio.diversity.beta.unweighted_unifrac(strainest_present, strainge_present,
                                                    sepi_strain_names, sepi_skbio_tree)
        y = skbio.diversity.beta.unweighted_unifrac(strainest_present, strainscan_present,
                                                    sepi_strain_names, sepi_skbio_tree)
        z = skbio.diversity.beta.unweighted_unifrac(strainge_present, strainscan_present,
                                                    sepi_strain_names, sepi_skbio_tree)

        # mean_pairwise_distance.append(np.mean([x,y,z]))
        sepi_mean_pwdist.append(np.mean([x,y,z]))
        sepi_min_dist_sister.append(cacnes_results.loc[idx]['min_dist_sister'])

# E. coli
ecoli_mean_pwdist = []
ecoli_min_dist_sister = []
for idx, sample_name in enumerate(Ecoli_sample_names):

    if idx == 0:
        continue

    sample_name_alt = f"{sample_name}_idx{idx}"

    strainest_sample = Ecoli_strainest_out.loc[sample_name].values
    strainge_sample = Ecoli_strainge_out.loc[sample_name_alt].values
    strainscan_sample = Ecoli_strainscan_out.loc[sample_name_alt].values

    strainest_present = (strainest_sample > 0).astype(int)
    strainge_present = (strainge_sample > 0).astype(int)
    strainscan_present = (strainscan_sample > 0).astype(int)

    # Get the unifrac distance between the genomes output by each method
    if np.sum(strainge_present) > 0 and np.sum(strainscan_present) > 0:
        z = skbio.diversity.beta.unweighted_unifrac(strainge_present, strainscan_present,
                                                    Ecoli_strain_names, Ecoli_skbio_tree)
        ecoli_mean_pwdist.append(z)
        ecoli_min_dist_sister.append(Ecoli_results.loc[idx]['min_dist_sister'])

#%% Plot stripplot/ boxplot

import seaborn as sns

fig, axs = plt.subplots()
fig.set_size_inches(4, 4)

box_df = pd.DataFrame(columns = ['Mean_Pw_dist','Species'])
box_df['Mean_Pw_dist'] = cacnes_mean_pwdist + sepi_mean_pwdist + ecoli_mean_pwdist
box_df['Species'] = ['C. acnes']*len(cacnes_mean_pwdist) + ['S. epidermidis']*len(sepi_mean_pwdist) + ['E. coli']*len(ecoli_mean_pwdist)

sns.stripplot(data=box_df, ax=axs, y='Species', x='Mean_Pw_dist', color='k')
# Just show mean line with boxplot
sns.boxplot(data=box_df, ax=axs, y='Species', x='Mean_Pw_dist', color='w', linewidth=2,
            meanline=True, showmeans=True, meanprops={'color':'k','ls':'-','lw':2},
            medianprops={'visible':False}, whiskerprops={'visible':False},
            showfliers=False, showcaps=False, showbox=False)
axs.set_xlabel('Mean Pairwise UniFrac\n between methods', **fmt)
axs.set_ylabel('', **fmt)
axs.set_yticklabels(labels = ['C. acnes','S. epidermidis','E. coli'], **fmt)
axs.xaxis.set_tick_params(labelsize=14)
# axs.yaxis.set_tick_params(labelsize=14, fontname='Helvetica')

fig.tight_layout()
fig.savefig('supplemental/figS1b.pdf', format='pdf', dpi=300)

print(f"Median mean pairwise distance (C. acnes): {np.median(cacnes_mean_pwdist)}")
print(f"Median mean pairwise distance (S. epidermidis): {np.median(sepi_mean_pwdist)}")
print(f"Median mean pairwise distance (E. coli): {np.median(ecoli_mean_pwdist)}")

#%% Plot unifrac against True Pairwise Distance to Closest Genome

fig, axs = plt.subplots(2)
fig.set_size_inches(5.5, 6)

axs[0].scatter(cacnes_min_dist_sister, cacnes_mean_pwdist, color='k')
# axs[0].set_xscale('log')
axs[0].set_ylim(0,.2)
axs[0].set_ylabel('Mean Pairwise UniFrac\n between methods', **fmt)
axs[0].set_title('C. acnes', **fmt)
axs[0].tick_params(axis='both', labelsize=12)
# Spearman correlation p value
r, pval = stats.spearmanr(cacnes_min_dist_sister, cacnes_mean_pwdist)

axs[0].text(.9, 0.85, f"p={pval:.3f}", transform=axs[0].transAxes, ha='center', va='bottom', fontsize=12)

axs[1].scatter(sepi_min_dist_sister, sepi_mean_pwdist, color='k')
# axs[1].set_xscale('log')
axs[1].set_ylim(0,.5)
axs[1].set_ylabel('Mean Pairwise UniFrac\n between methods', **fmt)
axs[1].set_xlabel('True Pairwise Distance to Closest Genome \n(#SNVs)', **fmt)
axs[1].set_title('S. epidermidis', **fmt)
axs[1].tick_params(axis='both', labelsize=12)
r, pval = stats.spearmanr(sepi_min_dist_sister, sepi_mean_pwdist)

axs[1].text(.9, 0.85, f"p={pval:.3f}", transform=axs[1].transAxes, ha='center', va='bottom', fontsize=12)

fig.tight_layout()
# fig.savefig('supplemental/figS1c.pdf', format='pdf', dpi=300)

#%% True pairwise distance to closest genome versus whether or not an output was given

fig, axs = plt.subplots(1,2)

fig.set_size_inches(10, 5)

strainest_df_melt = pd.DataFrame(columns=['Species','min_dist_sister','Output returned'])
strainest_df_melt['Species'] = ['C. acnes']*len(cacnes_results) + ['S. epidermidis']*len(sepi_results) + ['E. coli']*len(Ecoli_results)
strainest_df_melt['min_dist_sister'] = np.concatenate([cacnes_results['min_dist_sister'],
                                                        sepi_results['min_dist_sister'],
                                                        Ecoli_results['min_dist_sister']])
strainest_df_melt['Output returned'] = np.concatenate([(cacnes_results['StrainEst_Score'] != 3).astype(int),
                                                (sepi_results['StrainEst_Score'] != 3).astype(int),
                                                (Ecoli_results['StrainEst_Score'] != 3).astype(int)])


sns.stripplot(data=strainest_df_melt, ax=axs[0],
              x='Species', y='min_dist_sister', hue='Output returned',
              jitter=True, alpha=0.5, s=4, dodge=True)
sns.boxplot(data=strainest_df_melt, ax=axs[0],
            x='Species', y='min_dist_sister', hue='Output returned',
            showmeans=True, meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False)

axs[0].set_ylabel('True Distance to Closest Genome (#SNVs)', **fmt)
axs[0].set_xlabel('', **fmt)
axs[0].set_title('StrainEst', **fmt)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].legend([],[], frameon=False)

# p values on comparisons
cacnes_strainest_p = stats.mannwhitneyu(cacnes_results['min_dist_sister'][cacnes_results['StrainEst_Score'] == 3],
                                       cacnes_results['min_dist_sister'][cacnes_results['StrainEst_Score'] != 3])
sepi_strainest_p = stats.mannwhitneyu(sepi_results['min_dist_sister'][sepi_results['StrainEst_Score'] == 3],
                                        sepi_results['min_dist_sister'][sepi_results['StrainEst_Score'] != 3])
Ecoli_strainest_p = stats.mannwhitneyu(Ecoli_results['min_dist_sister'][Ecoli_results['StrainEst_Score'] == 3],
                                        Ecoli_results['min_dist_sister'][Ecoli_results['StrainEst_Score'] != 3])

axs[0].set_ylim(0, 8e4)
axs[0].text(0, 7.5e4, f"p={cacnes_strainest_p.pvalue:.1E}", fontsize=12, ha='center')
axs[0].text(1, 7.5e4, f"p={sepi_strainest_p.pvalue:.1E}", fontsize=12, ha='center')
axs[0].text(2, 7.5e4, f"p={Ecoli_strainest_p.pvalue:.1E}", fontsize=12, ha='center')


strainscan_df_melt = pd.DataFrame(columns=['Species','min_dist_sister','Output returned'])
strainscan_df_melt['Species'] = ['C. acnes']*len(cacnes_results) + ['S. epidermidis']*len(sepi_results) + ['E. coli']*len(Ecoli_results)
strainscan_df_melt['min_dist_sister'] = np.concatenate([cacnes_results['min_dist_sister'],
                                                        sepi_results['min_dist_sister'],
                                                        Ecoli_results['min_dist_sister']])
strainscan_df_melt['Output returned'] = np.concatenate([(cacnes_results['StrainScan_Score'] != 3).astype(int),
                                                (sepi_results['StrainScan_Score'] != 3).astype(int),
                                                (Ecoli_results['StrainScan_Score'] != 3).astype(int)])

sns.stripplot(data=strainscan_df_melt, ax=axs[1],
                x='Species', y='min_dist_sister', hue='Output returned',
                jitter=True, alpha=0.5, s=4, dodge=True)
sns.boxplot(data=strainscan_df_melt, ax=axs[1],
            x='Species', y='min_dist_sister', hue='Output returned',
            showmeans=True, meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False)

axs[1].set_ylabel('', **fmt)
axs[1].set_xlabel('', **fmt)
axs[1].set_title('StrainScan', **fmt)
axs[1].tick_params(axis='both', labelsize=12)

axs[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=12)

# p values on comparisons
cacnes_strainscan_p = stats.mannwhitneyu(cacnes_results['min_dist_sister'][cacnes_results['StrainScan_Score'] == 3],
                                        cacnes_results['min_dist_sister'][cacnes_results['StrainScan_Score'] != 3])
sepi_strainscan_p = stats.mannwhitneyu(sepi_results['min_dist_sister'][sepi_results['StrainScan_Score'] == 3],
                                        sepi_results['min_dist_sister'][sepi_results['StrainScan_Score'] != 3])
Ecoli_strainscan_p = stats.mannwhitneyu(Ecoli_results['min_dist_sister'][Ecoli_results['StrainScan_Score'] == 3],
                                        Ecoli_results['min_dist_sister'][Ecoli_results['StrainScan_Score'] != 3])

axs[1].set_ylim(0, 8e4)
axs[1].text(0, 7.5e4, f"p={cacnes_strainscan_p.pvalue:.2f}", fontsize=12, ha='center')
axs[1].text(1, 7.5e4, f"p={sepi_strainscan_p.pvalue:.2f}", fontsize=12, ha='center')
axs[1].text(2, 7.5e4, f"p={Ecoli_strainscan_p.pvalue:.2f}", fontsize=12, ha='center')

fig.tight_layout()

fig.savefig('supplemental/figS1d.pdf', format='pdf', dpi=300)

#%% What % of simulations did each method fail to return an output

# StrainEst
strainest_failrate = (np.sum(Ecoli_strainest_out.sum(1)==0) + np.sum(sepi_strainest_out.sum(1)==0) + np.sum(cacnes_strainest_out.sum(1)==0)) / (len(Ecoli_strainest_out) + len(sepi_strainest_out) + len(cacnes_strainest_out))
print(f"StrainEst fail rate: {strainest_failrate:.2f}")

# StrainGE
strainge_failrate = (np.sum(Ecoli_strainge_out.sum(1)==0) + np.sum(sepi_strainge_out.sum(1)==0) + np.sum(cacnes_strainge_out.sum(1)==0)) / (len(Ecoli_strainge_out) + len(sepi_strainge_out) + len(cacnes_strainge_out))
print(f"StrainGE fail rate: {strainge_failrate:.2f}")

# StrainScan
strainscan_failrate = (np.sum(Ecoli_strainscan_out.sum(1)==0) + np.sum(sepi_strainscan_out.sum(1)==0) + np.sum(cacnes_strainscan_out.sum(1)==0)) / (len(Ecoli_strainscan_out) + len(sepi_strainscan_out) + len(cacnes_strainscan_out))
print(f"StrainScan fail rate: {strainscan_failrate:.2f}")

