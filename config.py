"""
config.py

@author: wronk

Define parameters used in the connectivity estimation
"""
import numpy as np

# TODO: check labels for each subject, not all have the same number of
# divisions
# TODO: Double check that bounced 1 triggers for 019 were fixed

# 019 had bounced 1 triggers in original experiment.
# 020 had bad pupillometry, and the first run had a TTL issue
# 007 had bad pupillometry; All epochs rejected
# 022, 004 had droopy pupillometry
# 017, 019, 032, 034, 036 were excluded for behavioral performance in original
#      vocoder experiment

#subj_nums = [15, 17, 19, 23, 31, 32, 34, 38]  # 26, 36, 37

###############################################################################
# Human Connectome Project
hcp_path = '/media/Toshiba/Code/hcp_data'

hcp_subj_no_restin = [104012, 125525, 151526, 182840, 200109, 500222]
hcp_subj_no_motor = [100307, 102816, 111514, 112920, 116524, 146129, 149741,
                     154532, 158136, 166438, 172029, 174841, 175540, 179245,
                     181232, 182840, 187547, 195041, 214524, 223929, 233326,
                     248339, 352132, 352738, 433839, 512835, 555348, 665254,
                     715950, 825048, 872764, 877168, 917255, 990366]
hcp_subj_no_wm = [116524, 153732, 154532, 174841, 179245, 181232, 187547,
                  221319, 233326, 287248, 352132, 559053]
hcp_subj_no_storyM = [116524, 125525, 154532, 174841, 179245, 181232, 189349,
                      191841, 250427, 352132, 352738, 500222, 912447]
hcp_subj_remove = list(set(hcp_subj_no_restin + hcp_subj_no_motor +
                           hcp_subj_no_wm + hcp_subj_no_storyM))

subj_nums_hcp = [105923, 106521, 108323, 109123, 113922,
                 116726, 133019, 140117, 156334, 162935,
                 164636, 169040, 175237, 177746, 185442,
                 191033, 191437, 192641, 198653, 204521]
subj_nums_hcp = sorted(list(set(subj_nums_hcp) - set(hcp_subj_remove)))
subj_nums_hcp = [106521]  # Override for testing

inv_lambda = 1 / 25.

# Example from mne-hcp
#task_params_working_memory = dict(tmin=-1.5, tmax=2.5, decim=4,
#                                  event_id=dict(face=1), baseline=(-0.5, 0),
#                                  runs=range(2))

# XXX Double check decim
# Event ids pulled from data info
# Cue on from 0-0.15 sec, movement follows cue
motor_params = dict(tmin=0.15, tmax=1.15, decim=4, runs=range(2),
                    event_id=dict(LH=1, LF=2, RH=4, RF=5, fixate=6),
                    time_samp_col=3, stim_code_col=1,
                    baseline=(None, None))

# Event ids arbitrarily set to 1, events constructed during epo construction
rest_params = dict(tmin=0., tmax=1., decim=4, event_id=dict(rest=1),
                   runs=range(2), baseline=(None, None))

filt_params = dict(lp=0.1, hp=60., method='fir', phase='linear',
                   filter_length='2s')
                   #iir_params=dict(order=4, ftype='butter', output='sos'))
epo_reject = dict(mag=4e-12)

###############################################################################
# In Yeo, 2011 (fc-fMRI), strongest DMN connections seem to be PCC-TPJ,
#    and PCC-MPFC. Then TPJ-STS, and PCC-STS

# In Yeo, 2011 (fc-fMRI), strongest DAN connections seem to be
#    FEF-IPS/SPL7A, FEF-aMT+, aMT+/SPL7A,

# Divisions on fsaverage
# 7 div1, ind0: MPFC (Medial frontal region)
# 7 div2, ind1: STS (Superior temporal sulcus)
# 7 div3, ind2: PCC, pCun (Posterior cing. cort., preCuneus)
# 7 div4, ind3: TPJ, (temporoparietal junction) or angular gyrus(?)
# 7 div5, ind4: PHC, Parahippocampal cortex

# 7 div1, ind5: aMT+, SPL7A, etc. (Large region)
# 7 div2, ind6: FEF, (frontal eye field)

# Right half
# 7 div1, ind7: MPFC (Medial frontal region)
# 7 div2, ind8: STS (Superior temporal sulcus)
# 7 div3, ind9: PCC, pCun (Posterior cing. cort., preCuneus)
# 7 div4, ind10: TPJ, (temporoparietal junction) or angular gyrus(?)
# 7 div5, ind11: PHC, Parahippocampal cortex


# 7 div1, ind12: aMT+, SPL7A, etc. (Large region)
# 7 div2, ind13: FEF, (frontal eye field)

rsn_labels_lh = ['lh.7Networks_7_div%i.label' % i for i in range(1, 6)] + \
    ['lh.7Networks_3_div%i.label' % k for k in range(1, 3)]
rsn_labels_rh = ['rh.7Networks_7_div%i.label' % i for i in range(1, 6)] + \
    ['rh.7Networks_3_div%i.label' % k for k in range(1, 3)]
rsn_labels = rsn_labels_lh + rsn_labels_rh

###############################################################################
# Define processing parameters
# In preprocessing, frequency cutoff at 55 Hz
common_params = dict(mode='cwt_morlet',
                     cwt_frequencies=np.array([10, 12, 16, 20, 24, 30]),
                     #cwt_frequencies=np.array([12]),
                     n_cycles=5,
                     corr_wind=0.5,  # seconds
                     post_decim=10,
                     n_jobs=6)

config_proc_methods = [dict(method='windowed_power_corr')]

for adict in config_proc_methods:
    adict.update(common_params)

# conn_pairs should index into RSN divisions on fsaverage
config_conn_params = dict(conn_pairs=(np.array([3, 3, 3, 5, 10, 10, 10, 12]),
                                      np.array([0, 1, 2, 6, 7, 8, 9, 13])),
                          proc_methods=config_proc_methods,
                          rsn_labels=rsn_labels,
                          mean_mode='mean_flip',
                          n_jobs=6,
                          verbose=True)

# Support Vector Machine parameters
SVM_PARAMS = dict(C_range=[10. ** x for x in range(-4, 1)],
                  g_range=[10. ** x for x in range(-6, 1)],
#                  g_range=[10. ** x for x in [-7, -6, -5, -4, -3, -2]],
                  kernel='linear',  # Take out 'LinearModel' if not using linear kernel anymore
                  n_folds=5,
                  n_repeats=3,
                  cache_size=8192)


# Random forest parameters
#RF_PARAMS = dict(n_est_range=[50, 100, 1000, 2000, 5000, 10000],
RF_PARAMS = dict(n_est_range=[50, 100, 1000, 2000],
                 max_feat_range=10 ** np.arange(1, 6),
                 n_folds=5,
                 n_repeats=3,
                 n_jobs=-1)
