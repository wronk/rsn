"""
preprocess_raw.py

@author: wronk

Script to apply preprocessing methods, generate forward and inverse
"""
import os
from os import path as op
import numpy as np

import mne
import mnefun

# use niprove for handling events if possible
try:
    from niprov.mnefunsupport import handler

except ImportError:
    handler = None

data_dir = op.join(os.environ['CODE_ROOT'], 'rsn_data')
# 019 had bounced 1 triggers in original experiment.
# 020 had bad pupillometry, and the first run had a TTL issue
# 007 had bad pupillometry; All epochs rejected
# 022, 004 had droopy pupillometry
subj_nums = [15, 17, 19, 23, 31, 32, 34, 38]
subj_nums = [17]
epo_len = 1
debug = False

#######################################
# Construct subject information
#######################################

data_dir = '/media/Toshiba/Code/rsn_data/'

# Create a cropped version of data just for faster debugging
for subj_num in subj_nums:
    full_name = 'wronk_resting_%03d' % subj_num
    fname_raw = op.join(data_dir, full_name, 'raw_fif', full_name +
                        '_01_raw.fif')
    fname_raw_save = op.join(data_dir, full_name, 'raw_fif', full_name +
                             '_01_short_raw.fif')
    fname_erm = op.join(data_dir, full_name, 'raw_fif', full_name +
                        '_erm_raw.fif')
    fname_erm_save = op.join(data_dir, full_name, 'raw_fif', full_name +
                             '_erm_short_raw.fif')
    fname_raw_eve_save = op.join(data_dir, full_name, 'lists', 'ALL_' +
                                 full_name + '_01-eve.lst')
    fname_erm_eve_save = op.join(data_dir, full_name, 'lists', 'ALL_' +
                                 full_name + '_erm-eve.lst')

    for fname, sname, eve_name in zip([fname_raw, fname_erm],
                                      [fname_raw_save, fname_erm_save],
                                      [fname_raw_eve_save, fname_erm_eve_save]):
        raw = mne.io.Raw(fname, preload=True, verbose=True,
                         allow_maxshield=True)
        if debug:
            raw.crop(tmax=120, copy=False)
            raw.save(sname, overwrite=True)
        eve = mne.make_fixed_length_events(raw, 1, duration=epo_len)
        mne.write_events(eve_name, eve)

# TODO: Setup cuda and set `n_jobs_resampe='cuda'`
# TODO check t_adjust (trigger delay)
params = mnefun.Params(tmin=0., tmax=epo_len, n_jobs=6, n_jobs_mkl=1,
                       n_jobs_fir=6, n_jobs_resample=6,
                       decim=5, proj_sfreq=200, filter_length='10s',
                       cov_method='shrunk')

params.subjects = subj_nums
params.convert_subjects('wronk_resting_%03d', 'AKCLEE_1%02d')
params.dates = [None] * len(params.subjects)  # don't fill in with actual date
params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)), [])
params.ssp_ecg_reject = params.reject

#TODO: check 1) filter_length, 2) hpcut, 3) cov_method 4) baselining

#params.subjects = ['wronk_resting_%s' % num for num in subj_nums]
#params.structurals = ['AKCLEE_1%s' % num[-2:] for num in subj_nums]  # Struct dirs
params.work_dir = data_dir
#params.subject_indices = np.arange(len(subj_nums))

params.sws_ssh = 'itwronk@kasga.ilabs.uw.edu'
params.sws_dir = '/data06/itwronk/sss_work_dir'
params.work_dir = data_dir

params.baseline = None  # Don't apply baselineing
params.epochs_type = 'fif'

#params.on_process = handler  # Fails trying to find *raw_sss.fif
params.on_process = None
params.plot_drop_logs = False

# Run/save information
params.analyses = ['All']
params.in_names = ['rest']      # Input event names
params.in_numbers = [1]         # Input event numbers
params.out_names = [['rest']]   # Output event names
params.out_numbers = [[1]]      # Output types (all 0)
params.run_names = ['%s_01']    # Run names (all _01 here)
params.runs_empty = ['%s_erm']  # Name of empty room data
#params.run_names = ['%s_01_short']    # Run names of shortened data (all _01 here)
#params.runs_empty = ['%s_erm_short']  # Name of shortened empty room data
params.get_projs_from = np.arange(len(params.run_names))
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 2],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)

# Fwd/Inv
params.inv_runs = [np.arange(len(params.run_names))]
#params.inv_names = ['%s']

# SSS
params.sss_type = 'python'  # Run python implementation of SSS
params.tsss_dur = 10.       # Default, tSSS windown duration

params.freeze()
mnefun.do_processing(params,
                     do_score=False,      # F.
                     do_sss=False,        # T. Apply SSS filtering (locally)
                     do_ch_fix=False,     # ? Fix channel ordering
                     gen_ssp=False,       # T. for voc_meg. Reuse pre-generated SSPs for resting data
                     apply_ssp=True,      # T.
                     write_epochs=True,   # T. Save epochs objects
                     gen_covs=False,      # F. Generate covariances from baseline (should be using ERM)
                     gen_fwd=False,       # T. Generate forward matrix
                     gen_inv=False,       # T. Generate inverse matrix
                     gen_report=True,     # T. Write html of results
                     print_status=False)
