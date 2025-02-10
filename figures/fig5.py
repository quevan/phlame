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
import scipy
import scipy.stats as stats
import seaborn as sns
from matplotlib import transforms
import ete3
import glob

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures')

import fig_helper_functions as helper

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

fmt={'fontsize':15,
     'fontname':'Helvetica'}

%matplotlib auto


#%% Functions

def get_prevalence_across_md(byclade_df,
                            metadata_arr):
    
    # Number of clades to compare
    nclades = len(byclade_df.columns)
    categories, category_counts = np.unique(metadata_arr, 
                                            return_counts=True)
    ncategories = len(categories)
    
    
    # Initialize prevalence/r abundance object
    prevalence = np.zeros((nclades,ncategories))

    # Iterate through the clades
    for i, clade in enumerate(byclade_df.columns):
        
        # Get indices where sample has clade > 0
        clade_present_idx = byclade_df[clade].to_numpy().nonzero()[0]
        # Get corresponding metadata categories
        clade_present_md = pd.Series(metadata_arr).iloc[clade_present_idx]
        
        # Add 1 to each category to ensure all categories are counted 
        # even if prev = 0
        foo = pd.Series(categories)
        clade_present_md_add = pd.concat((clade_present_md, foo), axis=0)
        
        _, nums = np.unique(clade_present_md_add, return_counts=True)
        
        # -1 to correct for adding prev+1 before
        prevalence[i] = nums-1

    prevalence_df = pd.DataFrame(prevalence, 
                                 index=byclade_df.columns,
                                 columns=np.unique(metadata_arr))
    
    r_prevalence = prevalence/category_counts
    
    r_prevalence_df = pd.DataFrame(r_prevalence, 
                                     index=byclade_df.columns,
                                     columns=np.unique(metadata_arr))

    return prevalence_df, r_prevalence_df


def get_midpoint_root(tree_file, clade_IDs_file):
    
    tree = ete3.Tree(tree_file, format=1)

    midpoint_root = tree.get_midpoint_outgroup()
    tree.set_outgroup(midpoint_root)
    
    # Read in SLST labels
    subslst_md = pd.read_csv(clade_IDs_file, 
                             sep="\t", index_col=0, header=None)
    
    # Get MRCA of each clade
    dmrca=[]; mrca_names=[]; clades=[]
    dmrca_dct={}
    # For each clade,
    for clade in np.unique(subslst_md.iloc[:,0]):
        
        if clade == '-1':
            continue
        # Get tips belonging to each clade
        leaves = subslst_md.index[subslst_md.iloc[:,0] == clade].tolist()
        
        ls=[]
        if len(leaves) > 0:
            mrca = tree.get_common_ancestor(leaves)
            names=[]
            for leaf in mrca.iter_leaves():
                # Get distance from each tip to mrca
                ls.append(mrca.get_distance(leaf))
                names.append(leaf.name)
            clades.append(names)
            dmrca.append(np.array(ls).mean())
            dmrca_dct[clade] = np.array(ls).mean()
            mrca_names.append(clade)
            
    return dmrca_dct

def parse_coverage_multi(sample_names,
                         counts_dir,
                         reference_genome_ls,
                         reference_genome_names_ls):

    cov_cat = pd.DataFrame()
    
    for refgenome,refgenome_name in zip(reference_genome_ls,reference_genome_names_ls):

        cov_df = parse_coverage(counts_dir, sample_names, refgenome)

        cov_df.columns = [refgenome_name]

        cov_cat = pd.concat((cov_cat, cov_df), axis=1)

    return cov_cat

#%% Fig 5A: Phylogeny of C. acnes
# ==========================================
tree_file = "fig5/Cacnes_megatree_GTR_isonames_rooted_scaled.tre"

tree = ete3.Tree("fig5/Cacnes_megatree_GTR_isonames_rooted_scaled.tre", format=0)

# =============================================================================
# Load in metadata files
# =============================================================================

slst_md = pd.read_csv('fig5/Cacnes_megatree_phylogroupIDs.txt', sep="\t", index_col=0, header=None)


loc2shape_dct = {'China':'royalblue',
                 'Japan':'royalblue',
                 'Denmark':'gold',
                 'Netherlands':'gold',
                 'Singapore':'royalblue',
                 'USA':'red',
                 'Unknown':'gray'}

slst2shape_dct = {'A':'#1f77b4',
                  'B':'#ff7f0e',
                  'C':'#2ca02c',
                  'D':'#e377c2',
                  'E':'#9467bd',
                  'F':'#17becf',
                  'H':'#bcbd22',
                  'K':'#d62728',
                  'L':'#8c564b',
                  '-1':'#000000'}

slst2shape_dct = {'A':'gainsboro',
                  'B':'k',
                  'C':'gainsboro',
                  'D':'k',
                  'E':'gainsboro',
                  'F':'k',
                  'H':'k',
                  'K':'gainsboro',
                  'L':'k',
                  '-1':'#000000'}


#%% Fig 5A: Add metadata to nodes

# Color each leaf by phylogroup identity
for leaf in tree.iter_leaves():
    
    slst=slst_md.loc[leaf.name].values[0]
    slstshape=slst2shape_dct[slst]

    # Add SLST features
    leaf.add_face(ete3.RectFace(7,2,fgcolor=slstshape,bgcolor=slstshape),
                  column=0,position = "aligned")

# # Annotate MRCA node for SLSTs
for group in slst2shape_dct.keys():
    
    if group == '-1':
        continue
    
    leaves = slst_md.index[slst_md[1] == group].tolist()
    
    if len(leaves) > 0:
        mrca = tree.get_common_ancestor(leaves)
        mrca.name = group
        # mrca.add_face(ete3.TextFace(group,fsize=30,fgcolor=slst2shape_dct[group]), 
        #               column=0, position = "branch-bottom")


#%% Fig 5A: Plot

# TreeStyle
general_ts = ete3.TreeStyle()
general_ts.show_leaf_name=False
general_ts.scale = 0.025
# general_ts.show_branch_support=True
# general_ts.mode = "c"
# general_ts.arc_start = -90 # 0 degrees = 3 o'clock
# general_ts.arc_span = 180

# Remove blue dot on every node
nstyle = ete3.NodeStyle()
nstyle['shape']=''; nstyle['size']=0
nstyle['vt_line_width']=1
nstyle['hz_line_width']=1

for node in tree.traverse():
   node.set_style(nstyle)

general_ts.legend_position=3

tree.show(tree_style=general_ts)

#%% Fig 5A: Save tree

tree.render("fig5/fig5a.pdf", tree_style=general_ts, w=20, units="in", dpi=300)

#%% Fig 5B: Read in some input files

# Phlame MG calls
phlame_out_dir = 'fig5/6-frequencies'
counts_dir='fig5/5-counts'

metadata_file='fig5/global_skin_metadata.csv'
tree_file = 'fig5/Cacnes_megatree_GTR_isonames_rooted_scaled.tre'
sample_names_file = 'fig5/global_skin_sample_names.txt'

reference_genome='Pacnes_C1'

#%%
sample_names = np.loadtxt(sample_names_file,dtype=str)

##### Read in sample frequencies #####
phlame_frequencies = helper.read_phlame_frequencies_NEW(sample_names,
                                                        phlame_out_dir,
                                                        reference_genome)
clade_names_all = phlame_frequencies.columns

phlame_frequencies[phlame_frequencies < 0.01] = 0

##### Read in metadata #####
md = pd.read_csv( metadata_file, sep=',', index_col=0)

###### Remove nan samples if any ######
# sample_names = sample_names[~np.isnan(phlame_frequencies).any(1)]
# phlame_frequencies = phlame_frequencies.reindex(sample_names)

##### Sort metadata by column #####

md_site=md.loc[sample_names]['Site'].to_numpy()
md_state=md.loc[sample_names]['City'].to_numpy()
md_country=md.loc[sample_names]['Country'].to_numpy()
md_region=md.loc[sample_names]['Region'].to_numpy()
md_pj=md.loc[sample_names]['StudyName'].to_numpy()
md_west=md.loc[sample_names]['United States, Europe or Canada'].to_numpy()
md_age=md.loc[sample_names]['Age'].to_numpy().astype(int)
md_ageapprox=md.loc[sample_names]['ApproxAge'].to_numpy()
md_sex=md.loc[sample_names]['Sex'].to_numpy()
md_acne=md.loc[sample_names]['Acne?'].to_numpy()

coverage = helper.parse_coverage(sample_names, 
                                 counts_dir, reference_genome, method='mean')

##### Read in clade age info #####
# subphylo_dmrcas = get_midpoint_root(tree_file, 'fig5/Cacnes_megatree_subphylogroupIDs.txt')
# phylo_dmrcas = get_midpoint_root(tree_file, 'fig5/Cacnes_megatree_phylogroupIDs.txt')

# dmrcas = get_midpoint_root(tree_file, 'trees/Cacnes_megatree_GTR_isonames_cladeIDs_v2.txt')

#%% Define phylogroup / subphylogroup levels

# Read phylogroupIDs file into array
phylogroups = np.loadtxt('fig5/Cacnes_megatree_phylogroupIDs.txt',dtype=str)
subphylogroups = np.loadtxt('fig5/Cacnes_megatree_subphylogroupIDs.txt',dtype=str)

phylogroup_frequencies = phlame_frequencies[np.unique(phylogroups[:,2])]
phylogroup_frequencies.columns = [np.unique(phylogroups[np.where(phylogroups[:,2] == clade),1])[0] for clade in np.unique(phylogroups[:,2])]

# [1:] is to ignore -1
subphylogroup_frequencies = phlame_frequencies[np.unique(subphylogroups[:,2])[1:]]
subphylogroup_frequencies.columns = [np.unique(subphylogroups[np.where(subphylogroups[:,2] == clade),1])[0] for clade in np.unique(subphylogroups[:,2])[1:]]

##### Normalize samples that sum above 1 to 1 #####
norm_to_1 = phylogroup_frequencies.loc[(phylogroup_frequencies.sum(axis=1) > 1)]
phylogroup_frequencies.loc[(phylogroup_frequencies.sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)
norm_to_1 = subphylogroup_frequencies.loc[(subphylogroup_frequencies.sum(axis=1) > 1)]
subphylogroup_frequencies.loc[(subphylogroup_frequencies.sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)


# print(np.median([dmrcas[clade] for clade in np.unique(subphylogroups[:,2])[1:]]))

#%%
###### Filter samples ######
# Remove samples with acne diagnosis
acne_filter = md_acne != 'Yes'
# Remove samples ages < 18 or > 55
# -1 to include ambiguous ages for now
age_filter = ((md_age < 45) & (md_age > 18)) | (md_age == -1)
# Remove samples with less than 1x coverage
coverage_filter = (coverage>1).values.squeeze()
# pt_filter=np.array(['FO' not in sam for sam in sample_names]) # remove forehead DSM samples

# sample_filter = ( ((coverage>0.5).values.squeeze()) & pt_filter )
sample_filter = (acne_filter & age_filter & coverage_filter)

print(f"Median age of adult = {np.median(md_age[md_age > 18]):.2f}")

#%% Set colors
# =============================================================================

region_colors=['royalblue','gold','red']

#%% Fig 5B PART 1: PCOAs / A + A.1 drives geographic separation

import skbio.diversity as diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance import anosim

# Percent called filter
perc_called_bool = np.sum(phylogroup_frequencies[sample_filter],1) > 0.8

# Distance matrix for PCOA
bc_dists = diversity.beta_diversity('braycurtis',
                                    phylogroup_frequencies.loc[sample_filter][perc_called_bool], 
                                    ids=sample_names[sample_filter][perc_called_bool], 
                                    pairwise_func=scipy.spatial.distance.pdist)
# PCOA
ord_results = pcoa(bc_dists)
pcoa_df = pd.DataFrame({'PC1':ord_results.samples[['PC1']].to_numpy().flatten(),
                        'PC2':ord_results.samples[['PC2']].to_numpy().flatten(),
                        'Project':md_pj[sample_filter][perc_called_bool],
                        'Country':md_country[sample_filter][perc_called_bool],
                        'Region':md_region[sample_filter][perc_called_bool],
                        'Site':md_site[sample_filter][perc_called_bool]},
                        index=sample_names[sample_filter][perc_called_bool])

# =============================================================================
# PLOT
# =============================================================================

fig5a1, axs4a1 = plt.subplots(2, gridspec_kw={'height_ratios': [1, .3]})
fig5a1.set_size_inches(7.5,6.2)

# PCOA
sns.scatterplot(data=pcoa_df, x="PC1", y="PC2", 
                hue="Region", style="Project",  
                palette=region_colors, markers=['^','o','s','*','P','o','^','s','o','^'],
                s=40, ax=axs4a1[0], edgecolor='grey', linewidth=1)
axs4a1[0].set_xlabel(None)
axs4a1[0].set_ylabel(f'PCOA2 ({ord_results.eigvals[1]:.2f}%)', **fmt)
axs4a1[0].tick_params(axis='both', labelsize=15)
# axs4a1[0].set_xticklabels([])
axs4a1[0].legend()

handles, labels = axs4a1[0].get_legend_handles_labels()
order = [0,1,2,3,4,5,6,14,13,7,8,9,11,10,12]
axs4a1[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                 bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)

# Plot first loading (PC1)
axs4a1[1].scatter(pcoa_df['PC1'], 
                  phylogroup_frequencies.loc[sample_filter][perc_called_bool]['A'],
                  color='k', s=5, alpha=0.3)
axs4a1[1].set_ylabel("Phylogroup A \n Frequency", **fmt)
axs4a1[1].set_xlabel(f'PCOA1 ({ord_results.eigvals[0]:.2f}%)', **fmt)
axs4a1[1].tick_params(axis='both', labelsize=15)


fig5a1.subplots_adjust(hspace=0, wspace=0)
fig5a1.tight_layout()

fig5a1.savefig('fig5/fig5b.pdf', format='pdf')

# Permanova on regions
from skbio.stats.distance import permanova

res = permanova(bc_dists,
                grouping=pd.Series(md_region[sample_filter][perc_called_bool],
                                   index=sample_names[sample_filter][perc_called_bool]),
                permutations=999)

print(f"p-value between regions = {res['p-value']:.4f}")

# Correlation coefficient between PCOA1 and phylogroup A
corr = np.corrcoef(pcoa_df['PC1'], phylogroup_frequencies.loc[sample_filter][perc_called_bool]['A'])[0,1]
print(f"Correlation between PCOA1 and Phylogroup A = {corr:.4f}")

# Mean phylogroup richness on individuals
richness = np.sum(phylogroup_frequencies.loc[sample_filter][perc_called_bool] > 0, 1)
print(f"Mean phylogroup richness = {np.mean(richness):.4f}")


#%% Fig 5B, PART 2: Subphylogroup level

perc_called_bool = np.sum(subphylogroup_frequencies[sample_filter],1) > 0.8

# Distance matrix for PCOA
bc_dists = diversity.beta_diversity('braycurtis',
                                    subphylogroup_frequencies.loc[sample_filter][perc_called_bool], 
                                    ids=sample_names[sample_filter][perc_called_bool], 
                                    pairwise_func=scipy.spatial.distance.pdist)
# PCOA
ord_results = pcoa(bc_dists)
pcoa_df = pd.DataFrame({'PC1':ord_results.samples[['PC1']].to_numpy().flatten(),
                        'PC2':ord_results.samples[['PC2']].to_numpy().flatten(),
                        'Project':md_pj[sample_filter][perc_called_bool],
                        'Country':md_country[sample_filter][perc_called_bool],
                        'Region':md_region[sample_filter][perc_called_bool],
                        'Site':md_site[sample_filter][perc_called_bool]},
                        index=sample_names[sample_filter][perc_called_bool])

# =============================================================================
# PLOT
# =============================================================================

fig5a2, axs4a2 = plt.subplots(2, gridspec_kw={'height_ratios': [1, .25]})
fig5a2.set_size_inches(7,6)

# PCOA
sns.scatterplot(data=pcoa_df, x="PC1", y="PC2", 
                hue="Region", style="Project",  
                palette=region_colors, s=50, ax=axs4a2[0])
axs4a2[0].set_xlabel(None)
axs4a2[0].set_ylabel(f'PC2 ({ord_results.eigvals[1]:.2f}%)', **fmt)
axs4a2[0].tick_params(axis='y', labelsize=15)
axs4a2[0].set_xticklabels([])
axs4a2[0].legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)
axs4a2[0].grid()

# Plot first loading (PC1)
axs4a2[1].scatter(pcoa_df['PC1'], 
                  subphylogroup_frequencies.loc[sample_filter][perc_called_bool]['A.1'],
                  color='k', s=5, alpha=0.3)
                  
axs4a2[1].set_ylabel("A.1 Frequency", **fmt)
axs4a2[1].set_xlabel(f'PC1 ({ord_results.eigvals[0]:.2f}%)', **fmt)
axs4a2[1].tick_params(axis='both', labelsize=15)
axs4a2[1].xaxis.grid()

fig5a2.subplots_adjust(hspace=0)
fig5a2.tight_layout()

# fig5a2.savefig('fig5/fig5a2.pdf', format='pdf')
# fig5a2.savefig('fig5/fig5a2.png', dpi=300)

#%% Supplemental PCOAs (Batch effect)

figs5a, axs5a = plt.subplots(1,3)
figs5a.set_size_inches(20,4)

for i, region in enumerate(np.unique(md_region[sample_filter])):

    region_bool = md_region[sample_filter][perc_called_bool] == region
    
    bc_dists = diversity.beta_diversity('braycurtis',
                                        phylogroup_frequencies.loc[sample_filter][perc_called_bool][region_bool], 
                                        ids=sample_names[sample_filter][perc_called_bool][region_bool], 
                                        pairwise_func=scipy.spatial.distance.pdist)

    ord_results = pcoa(bc_dists)

    pcoa_df = pd.DataFrame({'PC1':ord_results.samples[['PC1']].to_numpy().flatten(),
                            'PC2':ord_results.samples[['PC2']].to_numpy().flatten(),
                            'Project':md_pj[sample_filter][perc_called_bool][region_bool],
                            'Country':md_country[sample_filter][perc_called_bool][region_bool],
                            'Region':md_region[sample_filter][perc_called_bool][region_bool],
                            'Site':md_site[sample_filter][perc_called_bool][region_bool]},
                            index=sample_names[sample_filter][perc_called_bool][region_bool])

    # PCOA
    sns.scatterplot(data=pcoa_df, x="PC1", y="PC2", 
                    hue="Project", style="Project",
                    palette='Set1', s=50, ax=axs5a[i])
    axs5a[i].set_xlabel(f'PCOA1 ({ord_results.eigvals[0]:.2f}%)', **fmt)
    axs5a[i].set_ylabel(f'PCOA2 ({ord_results.eigvals[1]:.2f}%)', **fmt)
    axs5a[i].tick_params(axis='y', labelsize=15)
    axs5a[i].set_xticklabels([])
    axs5a[i].legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)
    axs5a[i].grid()
    axs5a[i].set_title(region, **fmt)

    # PERMANOVA for batch effect between studies
    res = permanova(bc_dists,
                    grouping=pd.Series(md_pj[sample_filter][perc_called_bool][region_bool],
                                       index=sample_names[sample_filter][perc_called_bool][region_bool]),
                    permutations=999)
    
    print(f"p-value between studies in {region} = {res['p-value']:.4f}")
    axs5a[i].text(0.8, 0.95, f"p={res['p-value']:.4f}",
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axs5a[i].transAxes,
                  fontsize=15)

figs5a.tight_layout(h_pad=0.1)


#%% Fig 5C: Phylogroups A and F have subclades that are regionally restricted

import itertools

tree_file = "fig5/Cacnes_megatree_GTR_isonames_rooted_scaled.tre"

# Read in dMRCAs
dmrcas = get_midpoint_root(tree_file, 'fig5/Cacnes_megatree_GTR_isonames_cladeIDs_v2.txt')

clade_names = phlame_frequencies.columns
byclade_df = pd.DataFrame(phlame_frequencies[sample_filter], 
                          index=sample_names[sample_filter], 
                          columns=clade_names)
byclade_df['Country'] = md_country[sample_filter]
byclade_df['Continent'] = md_region[sample_filter]


# Pick which clades to plot

clades2plot = ['C.2.2.2.1.1.2', #A
               'C.2.2.2.1.1.2.1',
               'C.2.2.2.1.1.2.2.1',
               'C.2.2.2.1.1.2.2.2']

clade_names2plot = ['A','A.1','A.2','A.3']

# clades2plot = ['C.2.2.2.1.2.1', #F
#                'C.2.2.2.1.2.1.2',
#                'C.2.2.2.1.2.1.3',
#                'C.2.2.2.1.2.1.1']

# clade_names2plot = ['F','F.1','F.2','F.3']

# clades2plot = ['C.2.1', #K
#                'C.2.1.2.2.1.1.1',
#                'C.2.1.2.2.1.1.2',
#                'C.2.1.2.1',
#                'C.2.1.1']

# clade_names2plot = ['K','K.1.1','K.1.2','K.2','K.3']


subset_byclade_df = byclade_df.loc[:,clades2plot]

# =============================================================================
# Get prevalence across countries by clade
# =============================================================================

prevalence_df, r_prevalence_df = get_prevalence_across_md(subset_byclade_df, 
                                                          md_region[sample_filter])

r_prevalence_df.columns = [f"China (n = {np.sum(md_region[sample_filter] == 'China')})",
                           f"Europe (n = {np.sum(md_region[sample_filter] == 'Europe')})",
                           f"USA (n = {np.sum(md_region[sample_filter] == 'USA')})"]
r_prevalence_df['Clade'] = r_prevalence_df.index

r_prevalence_df_melt = pd.melt(r_prevalence_df, id_vars=['Clade'])
r_prevalence_df_melt.columns = ['Clade','Region','value']

# =============================================================================
# Relative abundance on the people who have it
# =============================================================================

subset_byclade_df['Region'] = md_region[sample_filter]

subset_byclade_df_melt = pd.melt(subset_byclade_df, id_vars=['Region'])
subset_byclade_df_nonzero = subset_byclade_df_melt.iloc[np.nonzero(subset_byclade_df_melt['value'].values)]
    
# =============================================================================
# Plot
# =============================================================================

fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
fig.set_size_inches(5.75,3.2)

## Prevalence ##
sns.barplot(data=r_prevalence_df_melt, 
            x='Clade', y='value', hue='Region', 
            palette=sns.color_palette(region_colors,3),
            linewidth=0.7,
            ax=axs[0])

axs[0].set_ylabel('Prevalence',size=14)
axs[0].set_xlabel('')
axs[0].set_xticklabels([])
# axs[0].xaxis.set_tick_params(labelsize=15)
axs[0].yaxis.set_tick_params(labelsize=12)
axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)
# axs[0].set_ylim(0,1.1)

# Add p values from chi2 test
# for idx_, subclade in enumerate(clade_names2plot[1:]):
#     axs[0].text(idx_+.75, 1.05, f"{float(pvals[pvals['Phylogroup'] == subclade]['p_adj']):.2f}", fontsize=12)
                
### dMRCA ###
dmrcas2plot = np.array([dmrcas[col] for col in subset_byclade_df.columns[:-1]])
# dmrcas2plot = np.array([phylo_dmrcas[col] for col in subset_byclade_df.columns[:-1]])

plt.rcParams['font.family'] = 'Helvetica'

table = axs[1].table(cellText=np.expand_dims(dmrcas2plot.astype(int),0),
                colLabels=clade_names2plot,
                rowLabels=['dMRCA'],
                loc='top',
                edges='open')
axs[1].axis('off')

table.set_fontsize(14)
table.scale(1, 1.25) 

fig.tight_layout(h_pad=0.1)

# fig.savefig('supplemental/figS8E.pdf',format='pdf')

#%% Fisher's exact test for A.3 and F.1 inside/outside of China

from scipy.stats import fisher_exact

region_counts = np.unique(md_region[sample_filter], return_counts=True)[1]

china_prevalence = prevalence_df.loc[['A.1','F.2'],['China']].values.flatten()
nonchina_prevalence = prevalence_df.loc[['A.1','F.2'],['Europe']].values.flatten() + \
                        prevalence_df.loc[['A.1','F.2'],['USA']].values.flatten()

# make a 2x2 table (china, not china) (present, not present)
for clade in ['A.1','F.2']:
    table = np.array([
            [prevalence_df['China'].loc[clade],region_counts[0]-prevalence_df['China'].loc[clade]],
            [prevalence_df['Europe'].loc[clade]+prevalence_df['USA'].loc[clade],region_counts[1]-(prevalence_df['Europe'].loc[clade]+prevalence_df['USA'].loc[clade])]])


    oddsratio, pval = fisher_exact(table)

    print(f"Clade {clade} has an odds ratio of {oddsratio:.2f} and p-value of {pval:.4f}")

#%%

# =============================================================================
# Fig 5C: Which subphylogroups are distributed differently than their parent clade?
# =============================================================================

# Get just the subphylogroups with a parent phylogroup
subphylogroups_w_parent = ['A.1','A.2','A.3','F.1','F.2','F.3','H.1','H.2','H.3','K.1.1','K.1.2','K.2','K.3']
parent_phylogroups = ['A','A','A','F','F','F','H','H','H','K','K','K','K']


phylo_byclade_df = pd.DataFrame(phylogroup_frequencies[sample_filter],
                                index=sample_names[sample_filter],
                                columns=phylogroup_frequencies.columns)
phylo_byclade_df['Region'] = md_region[sample_filter]

subphylo_byclade_df = pd.DataFrame(subphylogroup_frequencies[sample_filter], 
                          index=sample_names[sample_filter], 
                          columns=subphylogroup_frequencies.columns)
subphylo_byclade_df['Region'] = md_region[sample_filter]


prevalence_df, r_prevalence_df = get_prevalence_across_md(phylo_byclade_df, 
                                                          md_region[sample_filter])

subprevalence_df, r_subprevalence_df = get_prevalence_across_md(subphylo_byclade_df, 
                    md_region[sample_filter])


pvals = pd.DataFrame(columns=['Phylogroup','p_adj'])

for phylogroup, subphylogroup in zip(parent_phylogroups,
                                     subphylogroups_w_parent):

    prevalence_comparison = pd.DataFrame({'Phylogroup':prevalence_df.loc[phylogroup],
                                        'Subphylogroup':subprevalence_df.loc[subphylogroup]})

    res, pval, dof, expected_freq = stats.chi2_contingency(prevalence_comparison, correction=False)

    pvals = pvals.append({'Phylogroup':subphylogroup, 'p_adj':pval}, ignore_index=True)

    if pval*len(parent_phylogroups) < 0.05:
        print(f"Phylogroup {phylogroup} and Subphylogroup {subphylogroup} are differentially distributed")



#%% Fig 5D: Systematic identification of phylogenetic assocations with age


include_bool = (md_ageapprox != -1)

byclade_df = pd.DataFrame(phylogroup_frequencies.loc[include_bool], 
                          index=sample_names[include_bool], 
                          columns=phylogroup_frequencies.columns)

byclade_df['Sex'] = md_sex[include_bool]
byclade_df['Age'] = md_age[include_bool]
byclade_df['AgeApprox'] = md_ageapprox[include_bool]
byclade_df['Over40'] = [True if i > 40 else False for i in md_ageapprox[include_bool]]
byclade_df['Under40'] = [True if ((i <= 40) & (i > 0)) else False for i in md_ageapprox[include_bool]]
byclade_df['Region'] = md_region[include_bool]

byclade_df_melt = pd.melt(byclade_df, id_vars=['Sex','Over40','Under40','Age','Region'])
byclade_df_melt_nonzero = byclade_df_melt.iloc[np.nonzero(byclade_df_melt['value'].values)]

pvals = np.ones(len(phylogroup_frequencies.columns))
rankstats = np.zeros(len(phylogroup_frequencies.columns))

fig, axs = plt.subplots(5, 2, figsize=(12,10), 
                        gridspec_kw={'width_ratios':[1, 3]})

nhits = 0 # to count the number of hits

# Create a set for each sex, region, and clade combination
for c, clade in enumerate(phylogroup_frequencies.columns):
    
    this_clade_bool = ((byclade_df_melt_nonzero['variable']==clade))


    # Run a rank sum test if there are at least 5 points each in the over54 and under54 sets
    if ((np.sum(byclade_df_melt_nonzero['Over40'][this_clade_bool]) > 4) &
        (np.sum(byclade_df_melt_nonzero['Under40'][this_clade_bool]) > 4)):
    
        res, pval = stats.ranksums(byclade_df_melt_nonzero[this_clade_bool & (byclade_df_melt_nonzero['Over40'].values)]['value'],
                                    byclade_df_melt_nonzero[this_clade_bool & (byclade_df_melt_nonzero['Under40'].values)]['value'])
        pvals[c] = pval
        rankstats[c] = res

        # Bonferonni correction for the number of linear regressions ran
        if pval < 0.05/np.sum(pvals.flatten() < 1):
            
            # Plot the rank sum test + boxplots
            age_comparison_bool = ((byclade_df_melt_nonzero[this_clade_bool]['Over40'].values) | (byclade_df_melt_nonzero[this_clade_bool]['Under40'].values)) & (byclade_df_melt_nonzero[this_clade_bool]['variable'] == clade)
            
            sns.boxplot(data=byclade_df_melt_nonzero[this_clade_bool & age_comparison_bool], x='variable', y='value', hue='Over40',
                        boxprops=dict(alpha=.9), showfliers=False, ax=axs[nhits,0])
            sns.stripplot(data=byclade_df_melt_nonzero[this_clade_bool & age_comparison_bool], x='variable', y='value', hue='Over40',
                            dodge=True, linewidth=0.5, size=4, alpha=0.95, ax=axs[nhits,0])
            axs[nhits,0].legend([])
            axs[nhits,0].set_title(f'Rank sum p={pval:.4E}')
            axs[nhits,0].set_ylabel('Relative Abundance')
            
            # Plot continous clade abundance with age 
            axs[nhits,1].scatter(byclade_df_melt_nonzero.loc[this_clade_bool]['Age'],
                        byclade_df_melt_nonzero[this_clade_bool]['value'],
                        s=10, color='k')
            # axs[nhits,1].set_title(f"Region:{region}, Clade:{clade}")
            axs[nhits,1].set_title(f"Clade:{clade}")
            axs[nhits,1].axvline(50, color='tab:orange', ls='--', alpha=0.5)
            axs[nhits,1].axvline(30, color='tab:blue', ls='--', alpha=0.5)
            axs[nhits,1].set_ylim(0,1)
            axs[nhits,1].set_xlim(5,70)
            axs[nhits,1].set_xlabel('Age (y)')
            nhits+=1

plt.tight_layout()


#%% Fig 5D: Plot all the p-values and rank sum statistics ###


pvals_sorted = np.sort(pvals.flatten())

fig5e0, axs5e0 = plt.subplots()
fig5e0.set_size_inches(2.5,2)

axs5e0.scatter(np.arange(1,len(pvals)+1), np.log10(pvals_sorted),
            color='k', s=20)
axs5e0.set_xlabel('Rank', **fmt)
axs5e0.set_ylabel('Log10(p-value)', **fmt)
axs5e0.axhline( np.log10(0.05/len(pvals)), color='r')
axs5e0.set_xticks([])
axs5e0.tick_params(axis='both', which='major', labelsize=14)
# axs.set_xlim(-5,5)
fig5e0.tight_layout()

# fig5e0.savefig('fig5/fig5e0.pdf', format='pdf')

#%% ===========================================================================
# Fig 5D: Plot just the clade D summary information
# =============================================================================

# =============================================================================
# Calculate prevalence
# =============================================================================

# include_bool = (md_ageapprox != -1)

# Over40_prevalence_df, Over40_r_prevalence_df = get_prevalence_across_md(byclade_df, 
#                                                                         byclade_df['Over40'])

# Under40_prevalence_df, Under40_r_prevalence_df = get_prevalence_across_md(byclade_df,
#                                                                             byclade_df['Under40'])

# age_prevalence_df = pd.DataFrame({'Under40':Under40_r_prevalence_df[True],
#                                   'Over40':Over40_r_prevalence_df[True]})

# =============================================================================
# Plot
# =============================================================================

fig5e1, axs5e1 = plt.subplots()
fig5e1.set_size_inches(3,3)

# ## Prevalence ##
# axs5e1[0].bar([0,1], height=age_prevalence_df.loc['D'].values, color=['k','k'], edgecolor='k', width=0.8)

# axs5e1[0].set_ylabel('Prevalence',size=14)
# axs5e1[0].set_xlabel('Age group')
# axs5e1[0].set_xticks([0,1])
# axs5e1[0].set_xticklabels(['Under 30', 'Over 50'])
# axs5e1[0].yaxis.set_tick_params(labelsize=12)
# axs5e1[0].set_ylim(0,1); axs5e1[0].set_xlim(-0.5,1.5)

sns.boxplot(data=byclade_df_melt_nonzero[this_clade_bool & age_comparison_bool], x='variable', y='value',
            hue='Over40', palette=['w','w'], showfliers=False,  ax=axs5e1)
sns.stripplot(data=byclade_df_melt_nonzero[this_clade_bool & age_comparison_bool], x='variable', y='value',
               hue='Over40', palette=['k','k'], dodge=True, linewidth=0.5, size=3, ax=axs5e1)
axs5e1.legend([])
# axs5e1.set_title(f'p = {np.min(pvals):.2E}', **fmt)
axs5e1.set_xlabel(None)
# axs5e1.set_xticklabels(None)
# axs5e1.set_xticks(None)
axs5e1.set_ylim(0,1.1)
axs5e1.set_ylabel("Relative abundance \n (Phylogroup D)", **fmt)
axs5e1.tick_params(axis='both', which='major', labelsize=14)

fig5e1.tight_layout()

# fig5e1.savefig('fig5/fig5e1.pdf', format='pdf')
#%% ===========================================================================
# Fig 5E: Break down age differences in clade D by sex and region
# =============================================================================

byclade_df_melt = pd.melt(byclade_df, id_vars=['Sex','Over40','Under40','Age','Region'])
byclade_df_melt_nonzero = byclade_df_melt.iloc[np.nonzero(byclade_df_melt['value'].values)]

fig5e2, axs5e2 = plt.subplots(3)

fig5e2.set_size_inches(4,6)

for r, region in enumerate(['China','Europe','USA']):

    # Spearman correlation by region
    region_bool = (byclade_df_melt_nonzero['Region'] == region) & (byclade_df_melt_nonzero['variable']=='D')

    corr, pval = stats.spearmanr(byclade_df_melt_nonzero[region_bool]['Age'],
                                byclade_df_melt_nonzero[region_bool]['value'])
    
    fit = stats.linregress(byclade_df_melt_nonzero[region_bool]['Age'],
                            byclade_df_melt_nonzero[region_bool]['value'])
        

    for s, sex in enumerate(['F','M']):

        # Continuous plot of ages
        sexregion_bool = ((byclade_df_melt_nonzero['Region']==region) 
                          & (byclade_df_melt_nonzero['Sex']==sex) &
                        (byclade_df_melt_nonzero['variable']=='D'))

        if sex=='M':
            col = 'white'
        else:
            col = 'k'

        axs5e2[r].scatter(byclade_df_melt_nonzero[sexregion_bool]['Age'],
                       byclade_df_melt_nonzero[sexregion_bool]['value'],
                       s=15, color=col, edgecolors='k', linewidth=0.5, label=sex)
        
        axs5e2[r].set_title(region, **fmt)
        axs5e2[r].set_ylim(0,1)
        axs5e2[r].set_xlim(5,70)
        axs5e2[2].set_xlabel('Age (y)', **fmt)
        # axs5e2[r].set_ylabel('Frequency', **fmt)
        axs5e2[0].legend(loc = 'upper left', fontsize=12, title='Sex', title_fontsize=14)
        axs5e2[r].tick_params(axis='both', which='major', labelsize=14)

        # Add correlation coefficient and p value to plot
        axs5e2[r].text(s=f'p = {pval:.3f}', x=0.9, y=1.08, transform=axs5e2[r].transAxes, ha='center', fontsize=12)

        # Add linear fit line
        # axs5e2[r].plot([0,70], fit.intercept + fit.slope*np.array([0,70]), color='k', ls='--', lw=1)
        
        
fig5e2.tight_layout()

# fig5e2.savefig('fig5/fig5e2.pdf', format='pdf')

#%%

byclade_df_melt = pd.melt(byclade_df, id_vars=['Sex','Over40','Under40','Age','Region'])
byclade_df_melt_nonzero = byclade_df_melt.iloc[np.nonzero(byclade_df_melt['value'].values)]

fig5e2, axs5e2 = plt.subplots(1,3)

fig5e2.set_size_inches(7,3)

for r, region in enumerate(['China','Europe','USA']):
# for r, sex in enumerate(['F','M']):

    # Boxplot with age categories

    region_bool = (byclade_df_melt_nonzero['Region'] == region) & (byclade_df_melt_nonzero['variable']=='D')
    sexregion_bool = ((byclade_df_melt_nonzero['Region']==region) &
                    (byclade_df_melt_nonzero['Sex']==sex)
                    & (byclade_df_melt_nonzero['variable']=='D'))
    
    sex_bool = ((byclade_df_melt_nonzero['Sex']==sex)
                    & (byclade_df_melt_nonzero['variable']=='D'))

    has_age_bool = ((byclade_df_melt_nonzero['Age'] > 1))
    age_range_bool = ((byclade_df_melt_nonzero['Over40']) | (byclade_df_melt_nonzero['Under40']))
    
    sns.boxplot(data=byclade_df_melt_nonzero[region_bool  & age_range_bool & has_age_bool], x='variable', y='value', hue='Over40',
                palette = ['w','w'], showfliers=False, ax=axs5e2[r])
    
    sns.stripplot(data=byclade_df_melt_nonzero[region_bool  & age_range_bool & has_age_bool], x='variable', y='value', hue='Over40',
                    palette = ['k','k'], dodge=True, linewidth=0.8, edgecolors='k', size=3,
                    ax=axs5e2[r])
    
    # Hypothesis testing
    stat, pval = stats.ttest_ind(byclade_df_melt_nonzero[region_bool  & age_range_bool & has_age_bool].query('Over40 == True')['value'],
                                byclade_df_melt_nonzero[region_bool  & age_range_bool & has_age_bool].query('Over40 == False')['value'])
    
    axs5e2[r].text(s=f'p = {pval:.3f}', x=0.5, y=0.9, transform=axs5e2[r].transAxes, ha='center', fontsize=12)

    axs5e2[r].set_title(f"{region}")
    axs5e2[r].legend([],[], frameon=False)
    axs5e2[r].set_xlabel(None)
    axs5e2[r].set_ylabel(None)
    if region=='China':
        axs5e2[r].set_ylim(0,.5)
    else:
        axs5e2[r].set_ylim(0,1.2)
    # axs5e2[r,1].set_xticklabels(['Under 30', 'Over 50'])
    # axs5e2[r,1].set_xticks([0,1])
    axs5e2[r].tick_params(axis='both', which='major', labelsize=14)

fig5e2.tight_layout()

fig5e2.savefig('supplemental/figS8B.pdf', format='pdf')

#%%
# =============================================================================
# E. Search for novel clades driving true effects
# =============================================================================

# General idea of what I want to do:
# For every major clade,
# Get pis for everything that modeled to that clade,
# regardless of whether it was actually called or not.
# Filter pis for only things with a HPD
# Plot against various features

pi_maps = np.full((len(sample_names), len(phlame_frequencies.columns)),-1, dtype=np.float)
pi_hpd_lower = np.full((len(sample_names), len(phlame_frequencies.columns)),-1, dtype=np.float)
pi_hpd_upper = np.full((len(sample_names), len(phlame_frequencies.columns)),-1, dtype=np.float)

refgenome = 'Pacnes_C1'

pi_chain_HC = []
pi_chain_LC = []

for sidx, sample in enumerate(sample_names):

    path_to_sample_data = f"{phlame_out_dir}/{sample}_ref_{refgenome}_fitinfo.data"

    data = helper.FrequenciesData(path_to_sample_data)

    for cidx, chain in enumerate(data.chain):

        if len(chain) > 0:
            pi_chain = chain['pi']
        else:
            continue

        pi_map = helper.calc_MAP(pi_chain, bins = np.arange(0,1.01,0.01))
        hpd_lower, hpd_upper = helper.get_hpd(pi_chain, interval_size=0.95)

        pi_maps[sidx,cidx] = pi_map
        pi_hpd_lower[sidx,cidx] = hpd_lower
        pi_hpd_upper[sidx,cidx] = hpd_upper

        if (hpd_upper - hpd_lower < 0.2) & (pi_map > 0.3) & (pi_map < 0.7):
            pi_chain_HC.append(pi_chain)
        if (hpd_upper - hpd_lower > 0.4) & (pi_map > 0.3) & (pi_map < 0.7):
            pi_chain_LC.append(pi_chain)

hpd_mask = (pi_hpd_lower != -1) & (pi_hpd_upper != -1) & (pi_hpd_upper - pi_hpd_lower < 0.2)

#%%

# Look for associations between pi and various features
idx_to_look = np.where(np.in1d(phlame_frequencies.columns, 
                               np.unique(phylogroups[:,2])))[0]

phylogroup_names = [np.unique(phylogroups[np.where(phylogroups[:,2] == clade),1])[0] for clade in np.unique(phylogroups[:,2])]

fig, axs = plt.subplots(len(idx_to_look))
fig.set_size_inches(5,1.5*len(idx_to_look))

for idx_, clade in enumerate(idx_to_look):
    axs[idx_].hist(pi_maps[hpd_mask[:,clade],clade], bins=np.arange(0,1.01,0.01),
                color='white', edgecolor='k')
    axs[idx_].set_ylabel('# of Samples', **fmt)
    axs[idx_].tick_params(axis='both', which='major', labelsize=12)
    axs[idx_].text(0.95,0.8,phylogroup_names[idx_], ha='center', transform=axs[idx_].transAxes, fontsize=12)

axs[len(idx_to_look)-1].set_xlabel('MAP \u03C0 (High confidence)', **fmt)

fig.tight_layout()

# fig.savefig('fig6/fig6c.pdf', dpi=300, format='pdf')

#%% Difference between a high confidence and low confidence posterior

fig, axs = plt.subplots()
fig.set_size_inches(4,4)

pi_chain_HC_hist = np.histogram(pi_chain_HC[0], bins=np.arange(0,1.01,0.01))
pi_chain_LC_hist = np.histogram(pi_chain_LC[2], bins=np.arange(0,1.01,0.01))

axs.plot(pi_chain_HC_hist[1][:-1], pi_chain_HC_hist[0], color='k', label='High Confidence')
axs.plot(pi_chain_LC_hist[1][:-1], pi_chain_LC_hist[0], color='grey', label='Low Confidence')
axs.set_xlabel('\u03C0', **fmt)
axs.set_ylabel('Density', **fmt)
axs.set_yticks([])
axs.tick_params(axis='both', which='major', labelsize=12)

axs.legend()

fig.tight_layout()

#%% Supplemental 1: Novel C. acnes diversity is uncommon at the phylogroup level

perc_called_phylo = np.sum(phylogroup_frequencies,1)
perc_called_subphylo = np.sum(subphylogroup_frequencies,1)

fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, .3]})
fig.set_size_inches(7,4)
axs[0].scatter(coverage, perc_called_phylo,
                color='k', alpha=.4, s=5)
# axs[0].scatter(coverage[md_pj != 'Li, 2021'], perc_called_phylo[md_pj != 'Li, 2021'], 
#                color='k', alpha=.6, s=5, label='Other')
# axs[0].scatter(coverage[md_pj == 'Li, 2021'], perc_called_phylo[md_pj == 'Li, 2021'],
#                 color='r', alpha=.4, s=5, label='Li 2021')

axs[0].axhline(y=1, color='k', alpha=.3)
# axs[0].axvline(x=1, color='k', alpha=.3)    
axs[0].set_xscale('log')
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].set_xlabel('Mean Coverage across C. acnes genome', **fmt); axs[0].set_ylabel('Percent of sample called', **fmt)
# axs[0].legend()

axs[1].hist(perc_called_phylo[(coverage[0] > 1).values], bins=np.arange(0,1.01,0.02), color='k', alpha=0.9, orientation="horizontal")
axs[1].set_title('Mean coverage \n>1X', **fmt)
axs[1].set_yticks([])
axs[1].set_xlabel('# of Samples', **fmt)
axs[1].tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout()

print(f"% of samples above 1X coverage where >80% of the sample was classifiable: {np.sum(perc_called_phylo[(coverage[0] > 1).values] > 0.8)/np.sum(coverage[0] > 1):.2f}")
print(f"% of samples above 5X coverage where >80% of the sample was classifiable: {np.sum(perc_called_phylo[(coverage[0] > 5).values] > 0.8)/np.sum(coverage[0] > 5):.2f}")

# fig.savefig('supplemental/figS10a.pdf', dpi=300, format='pdf')

#%% Supplemental 2: 25 random stacked barplots from each region

slst2shape_dct = {'A':'#1f77b4',
                  'B':'#ff7f0e',
                  'C':'#2ca02c',
                  'D':'#e377c2',
                  'E':'#9467bd',
                  'F':'#17becf',
                  'H':'#bcbd22',
                  'K':'#d62728',
                  'L':'#8c564b',
                  '-1':'#000000'}

colors_ = [slst2shape_dct[slst] for slst in phylogroup_frequencies.columns]

import random

np.random.seed(1)

fig, axs = plt.subplots(3)
fig.set_size_inches(8,8)

for r, region in enumerate(['China','Europe','USA']):

    region_bool = md_region[sample_filter] == region

    random_sample_names = np.random.choice(list(sample_names[sample_filter][region_bool]), 25)

    random_samples = phylogroup_frequencies.loc[sample_filter].loc[random_sample_names]

    # Sort columns by letter
    random_samples = random_samples[sorted(random_samples.columns)]

    # Sort by phylogroup A
    random_samples = random_samples.loc[random_samples['A'].sort_values().index]

    colors_ = [slst2shape_dct[slst] for slst in random_samples.columns]

    if r == 1:
        random_samples.plot(kind='bar', stacked=True, ax=axs[r], edgecolor='k', linewidth=0.5, color=colors_) 
        axs[r].legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)
    else:
        random_samples.plot(kind='bar', stacked=True, ax=axs[r], edgecolor='k', linewidth=0.5, legend=False, color=colors_)

    axs[r].set_title(region, **fmt)
    axs[r].set_xticks([])
    axs[r].tick_params(axis='both', which='major', labelsize=12)
    axs[r].set_ylabel('R. abundance', **fmt)

    axs[r].set_xlabel('Subjects', **fmt)

fig.tight_layout()

fig.savefig('supplemental/figS10.pdf', dpi=300, format='pdf')

#%% Supplemental 3: Beta diversity across demographic features exceeds Beta diversity across studies (Batch effect)

import skbio.diversity as diversity
import itertools


perc_called_bool = np.sum(phylogroup_frequencies[sample_filter],1) > 0.8

bc_dists = diversity.beta_diversity('braycurtis',
                                    phylogroup_frequencies.loc[sample_filter][perc_called_bool], 
                                    ids=sample_names[sample_filter][perc_called_bool], 
                                    pairwise_func=scipy.spatial.distance.pdist)


df_melt = pd.DataFrame()

# Pairwise distances between regions
region_groups = {region: np.where(md_region[sample_filter][perc_called_bool] == region)[0] for region in np.unique(md_region)}
pairwise_distances = {
    (g1, g2): bc_dists[np.ix_(region_groups[g1], region_groups[g2])]
    for i, g1 in enumerate(region_groups) for g2 in list(region_groups)[i + 1:]
}
for (group1, group2), distances in pairwise_distances.items():
    pairwise_distances[group1, group2] = np.array(distances).flatten()

foo = list(itertools.chain.from_iterable(pairwise_distances.values()))

df_melt = pd.concat([df_melt, pd.DataFrame({'Category':['Across Regions']*len(foo), 'Distance':foo})])
                                            

# Pairwise distances between studies within each region
for region in np.unique(md_region[sample_filter]):

    studies = np.unique(md_pj[sample_filter][md_region[sample_filter] == region])

    study_groups = {study: np.where(md_pj[sample_filter][perc_called_bool] == study)[0] for study in studies}

    pairwise_distances = {
        (g1, g2): bc_dists[np.ix_(study_groups[g1], study_groups[g2])]
        for i, g1 in enumerate(study_groups) for g2 in list(study_groups)[i + 1:]
    }
    for (group1, group2), distances in pairwise_distances.items():
        pairwise_distances[group1, group2] = np.array(distances).flatten()

    foo = list(itertools.chain.from_iterable(pairwise_distances.values()))
    df_melt = pd.concat([df_melt, pd.DataFrame({'Category':[f'Within {region}']*len(foo), 'Distance':foo})])


fig, axs = plt.subplots()
fig.set_size_inches(4,4)

sns.boxplot(data=df_melt, x='Category', y='Distance', ax=axs, color='w', showfliers=False)
# sns.stripplot(data=df_melt, x='Category', y='Distance', ax=axs, color='k', alpha=0.5, jitter=0.2)

axs.set_ylabel('Bray-Curtis Distance', **fmt)
axs.set_xlabel('')

fig.tight_layout()

#%%

import numpy as np

# Example distance matrix (1000x1000)
distance_matrix = np.random.rand(1000, 1000)  # Replace with your actual distance matrix

# List of features
features = [
    'Study 1', 'Study 1', 'Study 1', 'Study 2',
    'Study 2', 'Study 2', 'Study 3', 'Study 3',
    'Study 3', 'Study 1', 'Study 2', 'Study 3'
]

# Convert features to a NumPy array
features_array = np.array(features)
# Group indices by feature categories
feature_groups = {feature: np.where(features_array == feature)[0] for feature in np.unique(features_array)}

# Extract pairwise distances for each combination of feature groups
pairwise_distances = {
    (g1, g2): distance_matrix[np.ix_(feature_groups[g1], feature_groups[g2])]
    for i, g1 in enumerate(feature_groups) for g2 in list(feature_groups)[i + 1:]
}

# Example of accessing distances
for (group1, group2), distances in pairwise_distances.items():
    pairwise_distances[group1, group2] = np.array(distances).flatten()
