"""
preprocess_raw.py

@author: wronk

Script to apply preprocessing methods, generate forward and inverse
"""
import os
from os import path as op
import numpy as np

import mnefun

# use niprove for handling events if possible
try:
    from niprov.mnefunsupport import handler

except ImportError:
    handler = None

data_dir = op.join(os.environ['CODE_ROOT'], 'rsn_data')
subj_nums = ['004', '007', '015', '017', '019', '020', '023', '031', '032',
             '034', '038']
subj_nums = ['017']

#######################################
# Construct subject information
#######################################

'''
# Create a cropped version of data just for faster debugging
fname_raw = '/media/Toshiba/Code/rsn_data/wronk_resting_017/raw_fif/wronk_resting_017_01_raw.fif'
fname_raw_save = '/media/Toshiba/Code/rsn_data/wronk_resting_017/raw_fif/wronk_resting_017_01_short_raw.fif'
fname_erm = '/media/Toshiba/Code/rsn_data/wronk_resting_017/raw_fif/wronk_resting_017_erm_raw.fif'
fname_erm_save = '/media/Toshiba/Code/rsn_data/wronk_resting_017/raw_fif/wronk_resting_017_erm_short_raw.fif'
fname_raw_eve_save = '/media/Toshiba/Code/rsn_data/wronk_resting_017/lists/ALL_wronk_resting_017_01-eve.lst'
fname_erm_eve_save = '/media/Toshiba/Code/rsn_data/wronk_resting_017/lists/ALL_wronk_resting_017_erm_short-eve.lst'

for fname, sname, eve_name in zip([fname_raw, fname_erm],
                                  [fname_raw_save, fname_erm_save],
                                  [fname_raw_eve_save, fname_erm_eve_save]):
    raw = mne.io.Raw(fname, preload=True, verbose=True, allow_maxshield=True)
    raw.crop(tmax=120, copy=False)
    raw.save(sname, overwrite=True)
    eve = mne.make_fixed_length_events(raw, 1)
    mne.write_events(eve_name, eve)
    '''

# Setup cuda and set `n_jobs_resampe='cuda'`
# TODO check t_adjust (trigger delay)
params = mnefun.Params(tmin=0., tmax=1., n_jobs=6, lp_cut=55, decim=6, proj_sfreq=200,
                       n_jobs_mkl=6, n_jobs_fir=6, n_jobs_resample=6,
                       filter_length='5s', hp_cut=1)

#TODO: check 1) filter_length, 2) hpcut, 3) cov_method 4) baselining

params.subjects = ['wronk_resting_%s' % num for num in subj_nums]
params.structurals = ['AKCLEE_1%s' % num[-2:] for num in subj_nums]  # Struct dirs
params.work_dir = data_dir
params.subject_indices = np.arange(len(subj_nums))

params.baseline = None  # Don't apply baselineing
#params.cov_method = 'emperical'
params.cov_method = 'shrunk'  # How to calc covariance. 'emperical', 'shrunk',
params.epochs_type = 'fif'

#params.on_process = handler  # Fails trying to find *raw_sss.fif
params.on_process = None

# Run/save information
params.analyses = ['All']
params.in_names = ['start']     # Input event names
params.in_numbers = [1]         # Input event numbers
params.out_names = [['start']]  # Output event names
params.out_numbers = [[1]]      # Output types (all 0)
#params.run_names = ['%s_01']    # Run names (all _01 here)
#params.runs_empty = ['%s_erm']  # Name of empty room data
params.run_names = ['%s_01_short']    # Run names (all _01 here)
params.runs_empty = ['%s_erm_short']  # Name of empty room data
params.get_projs_from = np.arange(len(params.run_names))
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 2],  # EOG; XXX [1, 1, 2] fails for ERM data
                    [0, 0, 0]]  # Continuous (from ERM)

# Fwd/Inv
params.inv_runs = [np.arange(len(params.run_names))]
params.inv_names = ['%s']

# SSS
params.sss_type = 'python'  # Run python implementation of SSS
# TODO: change back to 60 when done validating on 30s test data
params.tsss_dur = 60.       # Default, tSSS windown duration

params.freeze()
mnefun.do_processing(params,
                     do_score=True,
                     do_sss=True,        # Apply SSS filtering (locally)
                     do_ch_fix=True,     # Fix channel ordering
                     gen_ssp=True,       # Gen/Apply signal-space separation for artifacts
                     apply_ssp=True,     # ^^^
                     write_epochs=True,  # Save epochs objects
                     gen_covs=True,      # Generate covariances from baseline (should be using ERM)
                     gen_fwd=True,       # Generate forward matrix
                     gen_inv=True,       # Generate inverse matrix
                     gen_report=True,    # Write html of results
                     print_status=True)
