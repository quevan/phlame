"""
author: evanqu
date: 2024/03/18 11:51
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

os.chdir('/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/manuscript/figures')

import fig_helper_functions as helper

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

fmt={'fontsize':15,
     'fontname':'Helvetica'}

%matplotlib auto


#%% Functions

def group_hits_by_family(TP_ls, FP_ls, 
                         isotable_family, method_ls):
    '''
    Group metagenomic hits by whether the isolate origniates from the same person,
    same family, or different family.
    '''

    nTPs = []
    nFPs = []

    for i, (TP,FP) in enumerate(zip(TP_ls,FP_ls)):

        # Subject needs to have at least one isolate
        isotable_bool = (np.sum(isotable_family > 0,1) > 0).values

        TP_sum = np.sum(TP,1)
        FP_sum = np.sum(FP,1)
        
        nTPs.append(np.sum(TP_sum[isotable_bool]))
        nFPs.append(np.sum(FP_sum[isotable_bool]))

    nFPs_same_family = []
    nFPs_diff_family = []

    for i, FP in enumerate(FP_ls):
            
        # Only consider lineages with any isolates at all
        isotable_family_bool = isotable_family > 0
        
        # :-2 to remove handler samples
        same_family_P = FP & isotable_family_bool
        
        nFPs_sf = np.sum(same_family_P,1)
        nFPs_df = np.sum(FP,1) - nFPs_sf
        
        assert ((nFPs_sf + nFPs_df) == np.sum(FP,1)).all()

        nFPs_same_family.append(np.sum(nFPs_sf))
        nFPs_diff_family.append(np.sum(nFPs_df))

    df = pd.DataFrame({'Same Person':nTPs,
                        'Same Family':nFPs_same_family,
                        'Different Family':nFPs_diff_family})
    df.index = method_ls

    return df

#%% Fig 4B: Read in metadata and isolate information for C. acnes

path_to_MG_metadata = 'fig4/MG_metadata_CAT_forbenchmarking.csv'

path_to_cacnes_samples = 'fig4/Cacnes_samples2190.csv'
path_to_cacnes_assemblies_all = 'fig4/Cacnes_acera_assemblies.csv'
path_to_cacnes_assemblies_rep = 'fig4/Cacnes_rep_assemblies.csv'
cacnes_isolate2lineage_file = 'fig4/Cacnes_lineage2assembly.txt'

path_to_strainest_output = 'fig4/StrainEst'
path_to_strainge_output = 'fig4/StrainGE'
path_to_strainscan_output = 'fig4/StrainScan'
path_to_phlame_output = 'fig4/PHLAME'
path_to_counts = 'fig4/5-concat-counts_withL_31JUL'

path_to_crossfamily_sharing = 'fig4/Cacnes_crossfamily_sharing.csv'

reference_genome = 'Pacnes_C1'

# Read in subject metadata
MG_metadata = pd.read_csv(path_to_MG_metadata, header=0, 
                              index_col=False, dtype=str)
sample_names = MG_metadata['SAMPLE_NAME']

subj_IDs = np.unique(MG_metadata['SID'].to_numpy().astype(str))
subjfamily_IDs = np.array([subj[:-2] for subj in subj_IDs])

# Read in assemblies information
cacnes_assemblies_all = np.loadtxt(path_to_cacnes_assemblies_all, dtype=str, delimiter=',')
cacnes_assemblies_rep = np.loadtxt(path_to_cacnes_assemblies_rep, dtype=str, delimiter=',')

iso2lineage_dct = {}
with open(cacnes_isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = int(lineinfo[1])

# Read in isolate ground truth
cacnes_isotable = helper.samplescsv_to_isotable(path_to_cacnes_samples, subj_IDs, 
                                         'CacnesClusters_new')
cacnes_isotable_family = helper.group_by_family(cacnes_isotable, subjfamily_IDs)

# Read in cross-family sharing
cacnes_crossfamily_sharing = helper.samplescsv_to_isotable(path_to_crossfamily_sharing, subj_IDs,
                                                    'CacnesClusters_new')
cacnes_crossfamily_sharing_byfamily = helper.group_by_family(cacnes_crossfamily_sharing, subjfamily_IDs)

# Make crossfamily sharing table same size as isolate table
cacnes_crossfamily_sharing_byfamily = cacnes_crossfamily_sharing_byfamily.reindex(cacnes_isotable.index, fill_value=0)
cacnes_crossfamily_sharing_byfamily = cacnes_crossfamily_sharing_byfamily.reindex(cacnes_isotable.columns, axis=1, fill_value=0)

# Convert isolate table to phylogroup table
cacnes_isotable_phylogroup = helper.isotable_phylogroup_convert(cacnes_isotable,
                                                                'fig4/Cacnes_lineage2phylogroups.txt')

# Read in coverage
coverage = helper.parse_coverage(sample_names,
                                 path_to_counts,
                                 'Pacnes_C1')

print(f"Coverage range: {np.min(coverage[0]):.2f} - {np.max(coverage[0]):.2f}")
print(f"Median coverage: {np.median(coverage[0]):.2f}")

#%% Fig 3B: Read in strainest, strainge, and strainscan output

cacnes_strainest_out = helper.parse_strainest_out(path_to_strainest_output + '/Cacnes_classify_all',
                                           sample_names,
                                           cacnes_assemblies_all[:,0],
                                           reference_genome)
cacnes_strainge_out = helper.parse_strainge_out(path_to_strainge_output + '/Cacnes_classify_all',
                                            sample_names,
                                            cacnes_assemblies_all[:,0])

cacnes_strainest_rep_out = helper.parse_strainest_out(path_to_strainest_output + '/Cacnes_classify_rep',
                                                  sample_names,
                                                  cacnes_assemblies_rep[:,0],
                                                  reference_genome)                                        
cacnes_strainge_rep_out = helper.parse_strainge_out(path_to_strainge_output + '/Cacnes_classify_rep',
                                                sample_names,
                                                cacnes_assemblies_rep[:,0])
cacnes_strainscan_rep_out = helper.parse_strainscan_out(path_to_strainscan_output + '/Cacnes',
                                                sample_names,
                                                cacnes_assemblies_all[:,0])

# Merge frequencies by lineage
cacnes_strainest_freqs = helper.merge_frequencies_by_lineage(cacnes_strainest_out, iso2lineage_dct)
cacnes_strainge_freqs = helper.merge_frequencies_by_lineage(cacnes_strainge_out, iso2lineage_dct)
cacnes_strainest_rep_freqs = helper.merge_frequencies_by_lineage(cacnes_strainest_rep_out, iso2lineage_dct)
cacnes_strainge_rep_freqs = helper.merge_frequencies_by_lineage(cacnes_strainge_rep_out, iso2lineage_dct)
cacnes_strainscan_freqs = helper.merge_frequencies_by_lineage(cacnes_strainscan_rep_out, iso2lineage_dct)

# Read in phlame frequencies
cacnes_phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                         path_to_phlame_output,
                                                         'Pacnes_C1')
cacnes_phlame_freqs = cacnes_phlame_freqs.reindex(sorted(cacnes_phlame_freqs.columns), axis=1)

# Remove low frequency calls
cacnes_strainge_freqs[cacnes_strainge_freqs < 0.01] = 0
cacnes_strainest_freqs[cacnes_strainest_freqs < 0.01] = 0
cacnes_strainest_rep_freqs[cacnes_strainest_rep_freqs < 0.01] = 0
cacnes_strainge_rep_freqs[cacnes_strainge_rep_freqs < 0.01] = 0
cacnes_strainscan_freqs[cacnes_strainscan_freqs < 0.01] = 0
cacnes_phlame_freqs[cacnes_phlame_freqs < 0.01] = 0

assert helper.check_equal(cacnes_isotable.columns,
                          cacnes_phlame_freqs.columns,
                          cacnes_strainest_freqs.columns,
                          cacnes_strainge_freqs.columns,
                          cacnes_strainscan_freqs.columns)

# From Baker et al.
# C. acnes lineages 43 and 116 and S. epidermidis lineages 6 and 45 were used in high-biomass mock communities ... and identified in many samples. We therefore conservatively removed their abundances.

cacnes_strainest_freqs = cacnes_strainest_freqs.drop([43,116], axis=1)
cacnes_strainge_freqs = cacnes_strainge_freqs.drop([43,116], axis=1)
cacnes_strainest_rep_freqs = cacnes_strainest_rep_freqs.drop([43,116], axis=1)
cacnes_strainge_rep_freqs = cacnes_strainge_rep_freqs.drop([43,116], axis=1)
cacnes_strainscan_freqs = cacnes_strainscan_freqs.drop([43,116], axis=1)
cacnes_phlame_freqs = cacnes_phlame_freqs.drop([43,116], axis=1)

cacnes_isotable = cacnes_isotable.drop([43,116], axis=1)
cacnes_isotable_family = cacnes_isotable_family.drop([43,116], axis=1)

cacnes_crossfamily_sharing_byfamily = cacnes_crossfamily_sharing_byfamily.drop([43,116], axis=1)

# Don't consider families where the entire family has <10 isolates
nisolates_bool = (np.sum(cacnes_isotable_family,1) > 10).values

cacnes_phlame_freqs = cacnes_phlame_freqs.loc[nisolates_bool]
cacnes_strainge_freqs = cacnes_strainge_freqs.loc[nisolates_bool]
cacnes_strainest_freqs = cacnes_strainest_freqs.loc[nisolates_bool]
cacnes_strainest_rep_freqs = cacnes_strainest_rep_freqs.loc[nisolates_bool]
cacnes_strainge_rep_freqs = cacnes_strainge_rep_freqs.loc[nisolates_bool]
cacnes_strainscan_freqs = cacnes_strainscan_freqs.loc[nisolates_bool]

cacnes_isotable = cacnes_isotable.loc[nisolates_bool]
cacnes_isotable_family = cacnes_isotable_family.loc[nisolates_bool]

cacnes_crossfamily_sharing_byfamily = cacnes_crossfamily_sharing_byfamily.loc[nisolates_bool]

subj_IDs = subj_IDs[nisolates_bool]
MG_metadata = MG_metadata.loc[nisolates_bool]
subjfamily_IDs = subjfamily_IDs[nisolates_bool]

#%% Fig 3B: Calculate recall, precision, F1

cacnes_TP_ls = []; cacnes_FP_ls = []; cacnes_FN_ls = []
cacnes_recall_ls = []; cacnes_precision_ls = []; cacnes_f1_ls = []

color_ls = ['k','g','g','b','b','r']
method_ls = ['PHLAME','StrainEst','StrainEst (Derep.)','StrainGE','StrainGE (Derep.)','StrainScan']

# Need this temporarily because need to fix some samples for some tools 
# bool_include_ls = [phlame_bool_include, strainest_bool_include, strainge_bool_include,]

for freqs in [cacnes_phlame_freqs, cacnes_strainest_freqs, cacnes_strainest_rep_freqs, 
              cacnes_strainge_freqs, cacnes_strainge_rep_freqs, cacnes_strainscan_freqs]:
    
    # if i==0:
    bool_ = [True]*len(freqs)#     freqs = freqs[bool_]
        
    # Calc True and False positives
    TPs, FPs, FNs = helper.get_tpfps(freqs.to_numpy(),
                                     MG_metadata,
                                     cacnes_isotable, subj_IDs,
                                     bool_)
    cacnes_TP_ls.append(TPs)
    cacnes_FP_ls.append(FPs)
    cacnes_FN_ls.append(FNs)
    
    # Calc precision and recall
    recall, precision, f1 = helper.calc_recall_precision(TPs, FPs, FNs)
    
    cacnes_recall_ls.append(recall)
    cacnes_precision_ls.append(precision)
    cacnes_f1_ls.append(f1)

#%% Fig 3B2: Read in metadata and isolate information for S. epidermidis

path_to_MG_metadata = 'fig4/MG_metadata_CAT_forbenchmarking.csv'

path_to_sepi_samples = 'fig4/Sepi_acera_samples2025_nodoubletons.csv'
path_to_sepi_assemblies_all = 'fig4/Sepi_assemblies.csv'
path_to_sepi_assemblies_rep = 'fig4/Sepi_rep_assemblies.csv'
sepi_isolate2lineage_file = 'fig4/Sepi_lineage2assembly.txt'

path_to_strainest_output = 'fig4/StrainEst'
path_to_strainge_output = 'fig4/StrainGE'
path_to_strainscan_output = 'fig4/StrainScan'
path_to_phlame_output = 'fig4/PHLAME'
path_to_counts = 'fig4/5-concat-counts_withL_31JUL'

path_to_crossfamily_sharing = 'fig4/Sepi_crossfamily_sharing.csv'

# Read in subject metadata
MG_metadata = pd.read_csv(path_to_MG_metadata, header=0, 
                              index_col=False, dtype=str)
sample_names = MG_metadata['SAMPLE_NAME']

subj_IDs = np.unique(MG_metadata['SID'].to_numpy().astype(str))
subjfamily_IDs = np.array([subj[:-2] for subj in subj_IDs])

reference_genome = 'SepidermidisATCC12228'


# Read in assemblies information
sepi_assemblies_all = np.loadtxt(path_to_sepi_assemblies_all, dtype=str, delimiter=',')
sepi_assemblies_rep = np.loadtxt(path_to_sepi_assemblies_rep, dtype=str, delimiter=',')

iso2lineage_dct = {}
with open(sepi_isolate2lineage_file,'r') as f:
    for line in f:
        lineinfo = line.rstrip('\n').split('\t')
        iso2lineage_dct[lineinfo[0]] = int(lineinfo[1])

# Read in isolate ground truth
sepi_isotable = helper.samplescsv_to_isotable(path_to_sepi_samples, subj_IDs, 
                                         'SepiClusters_new')
sepi_isotable_family = helper.group_by_family(sepi_isotable, subjfamily_IDs)

# Read in cross-family sharing
sepi_crossfamily_sharing = helper.samplescsv_to_isotable(path_to_crossfamily_sharing, subj_IDs,
                                                    'SepiClusters_new')
sepi_crossfamily_sharing_byfamily = helper.group_by_family(sepi_crossfamily_sharing, subjfamily_IDs)

# Make crossfamily sharing table same size as isolate table
sepi_crossfamily_sharing_byfamily = sepi_crossfamily_sharing_byfamily.reindex(sepi_isotable.index, fill_value=0)
sepi_crossfamily_sharing_byfamily = sepi_crossfamily_sharing_byfamily.reindex(sepi_isotable.columns, axis=1, fill_value=0)


coverage = helper.parse_coverage(sample_names,
                                 path_to_counts,
                                 'SepidermidisATCC12228')

print(f"Coverage range: {np.min(coverage[0]):.2f} - {np.max(coverage[0]):.2f}")
print(f"Median coverage: {np.median(coverage[0]):.2f}")

#%% Fig 3B2: Read in strainest, strainge, and strainscan output


sepi_strainest_out = helper.parse_strainest_out(path_to_strainest_output + '/Sepi_classify_all',
                                           sample_names,
                                           sepi_assemblies_all[:,0],
                                           reference_genome)
sepi_strainge_out = helper.parse_strainge_out(path_to_strainge_output + '/Sepi_classify_all',
                                         sample_names,
                                         sepi_assemblies_all[:,0])

sepi_strainest_rep_out = helper.parse_strainest_out(path_to_strainest_output + '/Sepi_classify_rep',
                                               sample_names,
                                               sepi_assemblies_rep[:,0],
                                               reference_genome)

sepi_strainge_rep_out = helper.parse_strainge_out(path_to_strainge_output + '/Sepi_classify_rep',
                                             sample_names,
                                             sepi_assemblies_rep[:,0])

sepi_strainscan_out = helper.parse_strainscan_out(path_to_strainscan_output + '/Sepi_lowcov',
                                             sample_names,
                                             sepi_assemblies_all[:,0])
# Add in any samples with missing files
sepi_strainscan_out = sepi_strainscan_out.reindex(sepi_strainscan_out.index.union(sepi_strainscan_out.index)).fillna(0)

# Merge frequencies by lineage
sepi_strainest_freqs = helper.merge_frequencies_by_lineage(sepi_strainest_out, iso2lineage_dct)
sepi_strainge_freqs = helper.merge_frequencies_by_lineage(sepi_strainge_out, iso2lineage_dct)
sepi_strainest_rep_freqs = helper.merge_frequencies_by_lineage(sepi_strainest_rep_out, iso2lineage_dct)
sepi_strainge_rep_freqs = helper.merge_frequencies_by_lineage(sepi_strainge_rep_out, iso2lineage_dct)
sepi_strainscan_freqs = helper.merge_frequencies_by_lineage(sepi_strainscan_out, iso2lineage_dct)

# Read in phlame frequencies
sepi_phlame_freqs = helper.read_phlame_frequencies_NEW(sample_names,
                                                       path_to_phlame_output,
                                                       'SepidermidisATCC12228')
sepi_phlame_freqs = sepi_phlame_freqs.reindex(sorted(sepi_phlame_freqs.columns), axis=1)

# Remove low frequency calls
sepi_strainge_freqs[sepi_strainge_freqs < 0.01] = 0
sepi_strainest_freqs[sepi_strainest_freqs < 0.01] = 0
sepi_strainest_rep_freqs[sepi_strainest_rep_freqs < 0.01] = 0
sepi_strainge_rep_freqs[sepi_strainge_rep_freqs < 0.01] = 0
sepi_strainscan_freqs[sepi_strainscan_freqs < 0.01] = 0
sepi_phlame_freqs[sepi_phlame_freqs<0.01] = 0

assert helper.check_equal(sepi_isotable.columns,
                          sepi_phlame_freqs.columns,
                          sepi_strainest_freqs.columns,
                          sepi_strainge_freqs.columns,
                          sepi_strainscan_freqs.columns)

# C. acnes lineages 43 and 116 and S. epidermidis lineages 6 and 45 were used in high-biomass mock communities ... and identified in many samples. We therefore conservatively removed their abundances.

sepi_strainge_freqs = sepi_strainge_freqs.drop([6,45], axis=1)
sepi_strainest_freqs = sepi_strainest_freqs.drop([6,45], axis=1)
sepi_strainest_rep_freqs = sepi_strainest_rep_freqs.drop([6,45], axis=1)
sepi_strainge_rep_freqs = sepi_strainge_rep_freqs.drop([6,45], axis=1)
sepi_strainscan_freqs = sepi_strainscan_freqs.drop([6,45], axis=1)
sepi_phlame_freqs = sepi_phlame_freqs.drop([6,45], axis=1)

sepi_isotable = sepi_isotable.drop([6,45], axis=1)
sepi_isotable_family = sepi_isotable_family.drop([6,45], axis=1)

sepi_crossfamily_sharing_byfamily = sepi_crossfamily_sharing_byfamily.drop([6,45], axis=1)

# Don't consider families where the entire family has <10 isolates
sepi_nisolates_bool = (np.sum(sepi_isotable_family,1) > 10).values

sepi_phlame_freqs = sepi_phlame_freqs.loc[sepi_nisolates_bool]
sepi_strainge_freqs = sepi_strainge_freqs.loc[sepi_nisolates_bool]
sepi_strainest_freqs = sepi_strainest_freqs.loc[sepi_nisolates_bool]
sepi_strainest_rep_freqs = sepi_strainest_rep_freqs.loc[sepi_nisolates_bool]
sepi_strainge_rep_freqs = sepi_strainge_rep_freqs.loc[sepi_nisolates_bool]
sepi_strainscan_freqs = sepi_strainscan_freqs.loc[sepi_nisolates_bool]

sepi_isotable = sepi_isotable.loc[sepi_nisolates_bool]
sepi_isotable_family = sepi_isotable_family.loc[sepi_nisolates_bool]

sepi_crossfamily_sharing_byfamily = sepi_crossfamily_sharing_byfamily.loc[sepi_nisolates_bool]

subj_IDs = subj_IDs[sepi_nisolates_bool]
MG_metadata = MG_metadata.loc[sepi_nisolates_bool]
subjfamily_IDs = subjfamily_IDs[sepi_nisolates_bool]

#%% Fig 3B2: Calculate recall, precision, F1

sepi_TP_ls = []; sepi_FP_ls = []; sepi_FN_ls = []
sepi_recall_ls = []; sepi_precision_ls = []; sepi_f1_ls = []

color_ls = ['k','g','g','b','b','r']
method_ls = ['PHLAME','StrainEst','StrainEst (Derep.)','StrainGE','StrainGE (Derep.)','StrainScan']

for freqs in [sepi_phlame_freqs, sepi_strainest_freqs, sepi_strainest_rep_freqs,
              sepi_strainge_freqs, sepi_strainge_rep_freqs, sepi_strainscan_freqs]:
    
    # if i==0:
    bool_ = [True]*len(freqs)#     freqs = freqs[bool_]
        
    # Calc True and False positives
    TPs, FPs, FNs = helper.get_tpfps(freqs.to_numpy(),
                                     MG_metadata,
                                     sepi_isotable, subj_IDs,
                                     bool_)
    sepi_TP_ls.append(TPs)
    sepi_FP_ls.append(FPs)
    sepi_FN_ls.append(FNs)
    
    # Calc precision and recall
    recall, precision, f1 = helper.calc_recall_precision(TPs, FPs, FNs)
    
    sepi_recall_ls.append(recall)
    sepi_precision_ls.append(precision)
    sepi_f1_ls.append(f1)

#%% Fig 3B: Precision curve (lineages) - how many of the lineages are not isolated from the same individual?

fig4b, ax4b = plt.subplots(2)
fig4b.set_size_inches(5,6.5)

# Decide whether or not to count cross-family sharing
cacnes_isotable_family = cacnes_isotable_family + cacnes_crossfamily_sharing_byfamily
sepi_isotable_family = sepi_isotable_family + sepi_crossfamily_sharing_byfamily

cacnes_df = group_hits_by_family(cacnes_TP_ls, cacnes_FP_ls, 
                                 cacnes_isotable_family, method_ls)
sepi_df = group_hits_by_family(sepi_TP_ls, sepi_FP_ls, 
                               sepi_isotable_family, method_ls)

cacnes_df.iloc[::-1].plot(kind='barh', stacked=True, color=['w', 'k','red'], rot=0, alpha=0.8,
        edgecolor='black', width=0.8, ax=ax4b[0], legend=False)

sepi_df.iloc[::-1].plot(kind='barh', stacked=True, color=['w', 'k','red'], rot=0, alpha=0.8,
        edgecolor='black', width=0.8, ax=ax4b[1])

ax4b[1].set_xlabel('Total lineage detections\nacross individuals', **fmt)
ax4b[0].tick_params(axis='x', labelsize=14); ax4b[0].tick_params(axis='y', labelsize=14)
ax4b[1].tick_params(axis='x', labelsize=14); ax4b[1].tick_params(axis='y', labelsize=14)
ax4b[0].set_title('C. acnes', **fmt)
ax4b[1].set_title('S. epidermidis', **fmt)
ax4b[1].legend(fontsize=12, title='Lineage isolated from:')
fig4b.tight_layout()

# fig4b.savefig('fig4/fig4b.pdf',format='pdf')
# fig4b.savefig('fig4/fig4b.jpg',format='jpg')

#%% Stats for text


print('C. acnes')
for method in method_ls:
    print(f"Number of hits from same person/family ({method}): {(cacnes_df['Same Person'][method] + cacnes_df['Same Family'][method])}")
for method in method_ls:
    print(f"% of hits from different family ({method}): {cacnes_df['Different Family'][method]/np.sum(cacnes_df.loc[method])*100:.2f}")

print('S. epidermidis')
for method in method_ls:
    print(f"Number of hits from same person/family ({method}): {(sepi_df['Same Person'][method] + sepi_df['Same Family'][method])}")

for method in method_ls:
        print(f"% of hits from different family ({method}): {sepi_df['Different Family'][method]/np.sum(sepi_df.loc[method])*100:.2f}")



#%% Fig 4C: Go into tubes analysis

path_to_frequencies = 'fig4/gointotubes'
path_to_cacnes_samples = 'fig4/Cacnes_samples2190.csv'
path_to_sepi_samples = 'fig4/Sepi_acera_samples2025_nodoubletons.csv'
path_to_MG_metadata = 'fig4/MG_metadata_CAT_forbenchmarking.csv'

path_to_counts = 'fig4/5-concat-counts_withL_31JUL'

cacnes_coverage = helper.parse_coverage(sample_names,
                                        path_to_counts,
                                        'Pacnes_C1')

sepi_coverage = helper.parse_coverage(sample_names,
                                        path_to_counts,
                                        'SepidermidisATCC12228')

MG_metadata = pd.read_csv(path_to_MG_metadata, header=0, 
                              index_col=False, dtype=str)

sample_names = MG_metadata['SAMPLE_NAME']

subj_IDs = np.unique(MG_metadata['SID'].to_numpy().astype(str))

parent_bool = helper.contains_substring(sample_names, ['PA','PB'])

sample_names_parent = sample_names[parent_bool]

cacnes_frequencies = helper.read_phlame_frequencies_NEW(sample_names_parent,
                                                        path_to_frequencies,
                                                        'Pacnes_C1')
sepi_frequencies = helper.read_phlame_frequencies_NEW(sample_names_parent,
                                                  path_to_frequencies,
                                                  'SepidermidisATCC12228')

cacnes_isotable = helper.samplescsv_to_isotable(path_to_cacnes_samples, subj_IDs, 
                                               'CacnesClusters_new')
sepi_isotable = helper.samplescsv_to_isotable(path_to_sepi_samples, subj_IDs,
                                             'SepiClusters_new')

manual_lineages_to_drop = [['43','116'], ['6','45']]

coverage_bool_ls = [cacnes_coverage[parent_bool][0] > 1, sepi_coverage[parent_bool][0] > 1]

novel_isolates_df = pd.DataFrame(columns=['C. acnes','S. epidermidis'])

for i, (isotable, to_drop, bool_) in enumerate(zip([cacnes_isotable, sepi_isotable],
                                                    manual_lineages_to_drop,
                                                    coverage_bool_ls)):

    isotable_parent = isotable[parent_bool]
    isotable_child = isotable[~parent_bool][:-2] # -2 is to remove JSB and TCL
    
    lineages_adult_only = isotable_parent.columns[isotable_parent.sum(axis=0) > 0]
    lineages_nochild = isotable_child.columns[isotable_child.sum(axis=0) == 0]
    
    lineages_in_adults = [lin for lin in lineages_adult_only if lin not in to_drop]
    lineages_not_in_children = [lin for lin in lineages_nochild if lin not in to_drop]

    # Get the fraction of isolates in parents but not also in children
    # Note that value can be NA if no isolates were taken from that child
    frac_not_in_children = np.sum(isotable_parent[lineages_not_in_children],1) / np.sum(isotable_parent[lineages_in_adults],1)
    novel_isolates_df.iloc[:,i] = frac_not_in_children

percent_called_cacnes = np.sum(cacnes_frequencies, 1)
percent_called_sepi = np.sum(sepi_frequencies, 1)

# Get correlation coefficients for C. acnes and S. epidermidis
from scipy.stats import pearsonr

cacnes_coverage_bool = (cacnes_coverage[parent_bool][0] > 1).values
cacnes_isna_bool = ~pd.isna(novel_isolates_df['C. acnes'][cacnes_coverage_bool]).values
correlation_cacnes = pearsonr(percent_called_cacnes[cacnes_coverage_bool][cacnes_isna_bool],
                                novel_isolates_df['C. acnes'][cacnes_coverage_bool][cacnes_isna_bool])

sepi_coverage_bool = (sepi_coverage[parent_bool][0] > 1).values
sepi_isna_bool = ~pd.isna(novel_isolates_df['S. epidermidis'][sepi_coverage_bool]).values
correlation_sepi = pearsonr(percent_called_sepi[sepi_coverage_bool][sepi_isna_bool],
                                novel_isolates_df['S. epidermidis'][sepi_coverage_bool][sepi_isna_bool])

fig4c, axs4c = plt.subplots()
fig4c.set_size_inches(3.5,5)

# C. acnes
axs4c.scatter(percent_called_cacnes[(cacnes_coverage[parent_bool][0] > 1).values],
            ls[0][(cacnes_coverage[parent_bool][0] > 1).values],
            color='k', edgecolors='black', label=f'C. acnes (r = {correlation_cacnes[0]:.2f})')

# S. epidermidis
axs4c.scatter(percent_called_sepi[(sepi_coverage[parent_bool][0] > 1).values],
             ls[1][(sepi_coverage[parent_bool][0] > 1).values], color='white', edgecolors='black', 
             label=f'S. epidermidis (r = {correlation_sepi[0]:.2f})')

axs4c.plot([0,1],[1,0], 'k', alpha=0.7)
axs4c.set_xlim(-0.05,1.05); axs4c.set_ylim(-0.05,1.05)
axs4c.set_xlabel('% of sample classified', **fmt)
axs4c.set_ylabel('% of novel isolates\n (unshared with parents)', **fmt)
axs4c.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1,1.5))
axs4c.tick_params(axis='both', labelsize=12)
fig4c.tight_layout()

fig4c.savefig('fig4/fig4c.pdf',format='pdf')


from colorcet.plotting import swatch, swatches, candy_buttons
import holoviews as hv
from matplotlib.cm import get_cmap

path_to_counts = '/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2022_03_Jacob MGs_NEW/Cacnes_CAT/5-concat-counts_withL_31JUL'
path_to_phlame_output = '/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/phlame_project/results/2022_03_Jacob MGs_NEW/Cacnes_CAT/6-lineage_frequencies'
# path_to_MG_metadata = 'fig4/MG_metadata_CAT.csv'
path_to_MG_metadata = 'fig4/MG_metadata_CAT_forbenchmarking.csv'
path_to_samples = 'fig4/Cacnes_samples2190.csv'
# path_to_samples = 'fig4/Sepi_acera_samples2025_nodoubletons.csv'

MG_metadata = pd.read_csv(path_to_MG_metadata, header=0, 
                              index_col=False, dtype=str)

sample_names = MG_metadata['SAMPLE_NAME']

subj_IDs = MG_metadata['SID']
TPs = MG_metadata['TP']

subj_tp_IDs = np.unique([subj_+tp_ for subj_, tp_ in zip(subj_IDs,TPs)])


def samplescsv_to_isotable_bytp(path_to_samples_csv,
                                subj_tp_IDs, cluster_col):
    '''
    Convert samples_csv file into isolate table.
    '''
    
    samples_df = pd.read_csv(path_to_samples_csv) 
    sample_names = samples_df['SampleName']
    cluster = samples_df[cluster_col]
    
    isotable = pd.DataFrame( np.zeros((len(subj_tp_IDs),
                                       len(np.unique(cluster)))),
                             index = subj_tp_IDs,
                             columns = np.unique(cluster) )
    
    for subj_idx, subj in enumerate(subj_tp_IDs):
        
        if subj == 'nan':
            continue
        
        this_subj_bool = np.array([subj in str(name)[:4] for name in sample_names])
        
        this_subj_clusters = cluster[this_subj_bool]
        
        cluster_cts = np.unique(this_subj_clusters, return_counts=True)
        
        for clu, cts in zip(cluster_cts[0], cluster_cts[1]):
            
            isotable[clu].iloc[subj_idx] = cts 
            
    return isotable


cacnes_isotable_tp = samplescsv_to_isotable_bytp(path_to_samples, subj_tp_IDs, 
                                                'CacnesClusters_new')

cacnes_isotable = helper.samplescsv_to_isotable(path_to_samples, subj_IDs,
                                                'CacnesClusters_new')

#%% Figure for tami

colors = get_cmap("cet_glasbey")

# subj_tp = ['1AA1','1AA2','1AA4']
subj_tp = ['1AA']
samples_1AA = ['ConcatenatedReads_1AA_TP_1_all',
                'ConcatenatedReads_1AA_TP_3_all',
                'ConcatenatedReads_1AA_TP_4_all',
                'ConcatenatedReads_1AA_TP_5_all']


phlame_frequencies = helper.read_phlame_frequencies(samples_1AA,
                                                    path_to_phlame_output,
                                                    'Pacnes_C1')

# Width ratio is sort of arbitrary to for MG and Iso columns, respectively

phlame_frequencies.columns = phlame_frequencies.columns.astype(int)
phlame_frequencies_sorted = phlame_frequencies.sort_index(1)

this_subj_isolates = cacnes_isotable.loc[subj_tp]
iso_rename_columns = pd.Series(this_subj_isolates.columns)

# Rename columns to hide lineages that don't appear ever
iso_rename_columns[(np.sum(this_subj_isolates).values) == 0] = '_'
this_subj_isolates = pd.DataFrame(this_subj_isolates)
this_subj_isolates.columns = iso_rename_columns
MG_rename_columns = pd.Series(phlame_frequencies_sorted.columns)
MG_rename_columns[((phlame_frequencies_sorted.sum(0)) == 0).values] = '_'
phlame_frequencies_sorted.columns = MG_rename_columns


#Plot MG and isolate barplots    
# (this_subj_isolates/np.tile(this_subj_isolates.sum(1),
#                             np.size(this_subj_isolates,1))).plot.bar(stacked=True, colormap=colors, ax=axs4a[0],legend=None)
fig4a, axs4a = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]})
fig4a.set_size_inches(6,3)

#Normalize this_subj_isolates by row
this_subj_isolates_norm = this_subj_isolates.div(this_subj_isolates.sum(axis=1), axis=0)

this_subj_isolates_norm.plot.bar(stacked=True, colormap=colors, ax=axs4a[0], legend=False)
phlame_frequencies_sorted.plot.bar(stacked=True, colormap=colors, ax=axs4a[1])

axs4a[0].set_ylabel('Frequency', **fmt)
# axs4a[0].set_title('Isolates', **fmt)
# axs4a[1].set_title('Metagenomics', **fmt)

axs4a[0].set_xlabel('Isolates', **fmt)
axs4a[1].set_xlabel('Metagenomics', **fmt)
axs4a[1].set_ylim(0,1.1); axs4a[0].set_ylim(0,1.1)

axs4a[0].spines.top.set_visible(False)
axs4a[1].spines.top.set_visible(False)
axs4a[0].spines.right.set_visible(False)
axs4a[1].spines.right.set_visible(False)
axs4a[1].tick_params(axis='both', labelsize=12)
axs4a[0].tick_params(axis='both', labelsize=12)
axs4a[0].set_xticklabels([], rotation=0)
axs4a[1].set_xticklabels([], rotation=0)
# axs4a[0].legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize='medium',ncol=1)
axs4a[1].legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize='medium',ncol=1, title='Lineage')

fig4a.tight_layout(w_pad=1.5)

# fig4a.savefig('unused_panels/fig4a.pdf',format='pdf')


# coverage = helper.parse_coverage(samples_1AA,
#                                  path_to_counts,
#                                  'Pacnes_C1')
