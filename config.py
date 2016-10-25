"""
config.py

@author: wronk

Define parameters used in the connectivity estimation
"""
import numpy as np

# TODO: check labels for each subject, not all have the same number of
# divisions
# XXX: Could test both left and right hemispheres instead of just one

# 019 had bounced 1 triggers in original experiment.
# 020 had bad pupillometry, and the first run had a TTL issue
# 007 had bad pupillometry; All epochs rejected
# 022, 004 had droopy pupillometry
subj_nums = [15, 17, 19, 23, 31, 32, 34, 38]
inv_lambda = 1 / 9.

# Divisions on fsaverage
# 7 div1, ind0: MPFC (Medial frontal region)
# 7 div2, ind1: STS (Superior temporal sulcus)
# 7 div3, ind2: PCC, pCun (Posterior cing. cort., preCuneus)
# 7 div4, ind3: TPJ, (temporoparietal junction) or angular gyrus(?)
# 7 div5, ind4: XXX Name? Small region on medial/ventral side of temporal lobe

# 7 div1, ind5: MT, SPL7A, etc. (Large region)
# 7 div2, ind6: FEF, (frontal eye field)

# Right half
# 7 div1, ind7: MPFC (Medial frontal region)
# 7 div2, ind8: STS (Superior temporal sulcus)
# 7 div3, ind9: PCC, pCun (Posterior cing. cort., preCuneus)
# 7 div4, ind10: TPJ, (temporoparietal junction) or angular gyrus(?)
# 7 div5, ind11: XXX Name? Small region on medial/ventral side of temporal lobe

# 7 div1, ind12: MT, SPL7A, etc. (Large region)
# 7 div2, ind13: FEF, (frontal eye field)

rsn_labels_lh = ['lh.7Networks_7_div%i.label' % i for i in range(1, 6)] + \
    ['lh.7Networks_3_div%i.label' % k for k in range(1, 3)]
rsn_labels_rh = ['rh.7Networks_7_div%i.label' % i for i in range(1, 6)] + \
    ['rh.7Networks_3_div%i.label' % k for k in range(1, 3)]
rsn_labels = rsn_labels_lh + rsn_labels_rh

# Define processing parameters
# In preprocessing, frequency cutoff at 55 Hz
common_params = dict(mode='cwt_morlet',
                     cwt_frequencies=np.array([10, 12, 16, 20, 24]),
                     n_cycles=3,
                     corr_wind=0.5)  # seconds

config_conn_methods = [dict(method='windowed_power_corr')]

for adict in config_conn_methods:
    adict.update(common_params)

# conn_pairs should index into RSN divisions on fsaverage
config_conn_params = dict(conn_pairs=(np.array([3, 3, 3, 5, 10, 10, 10, 12]),
                                      np.array([0, 1, 2, 6, 7, 8, 9, 13])),
                          conn_methods=config_conn_methods,
                          rsn_labels=rsn_labels,
                          mean_mode='mean_flip',
                          n_jobs=6,
                          verbose=True)
