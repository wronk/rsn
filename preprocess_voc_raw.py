"""
preprocess_voc_raw.py

@author: wronk, larsoner

Script to apply preprocessing methods, generate forward and inverse
for vocoder data
"""

import os
from os import path as op
import mnefun
import numpy as np
import config

data_dir = op.join(os.environ['CODE_ROOT'], 'voc_data')
subj_nums = config.subj_nums

params = mnefun.Params(tmin=-0.2, tmax=7.5, n_jobs=6, n_jobs_mkl=1,
                       n_jobs_fir=6, n_jobs_resample=6,
                       decim=5, proj_sfreq=200, filter_length='10s',
                       cov_method='shrunk')

params.subjects = subj_nums

params.convert_subjects('eric_voc_%03d', 'AKCLEE_1%02d')
params.dates = [None] * len(params.subjects)  # don't fill in with actual date
#params.score = score  # scoring function to use
params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)), [])
params.ssp_ecg_reject = params.reject

params.sws_ssh = 'itwronk@kasga.ilabs.uw.edu'
params.sws_dir = '/data06/itwronk/sss_work_dir'
params.work_dir = data_dir

params.run_names = ['%%s_%02d' % (x + 1) for x in range(8)]
params.get_projs_from = np.arange(len(params.run_names))
params.inv_names = ['%s']
params.inv_runs = [np.arange(len(params.run_names))]
params.runs_empty = ['%s_erm']
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 2],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)

params.in_names = ['M10_', 'S10_', 'M20_', 'S20_', 'V10_', 'V20_', 'V2N_']
params.in_numbers = [31, 41, 32, 42, 51, 52, 62]

###############
# Added
params.baseline = None  # Don't apply baselineing
params.epochs_type = 'fif'

params.on_process = None
params.plot_drop_logs = False

###############

#params.sss_type = 'python'  # Run python implementation of SSS
params.tsss_dur = 10.

params.analyses = [
    'All',
    'MSxNC',
]
params.out_names = [
    ['All'],
    ['M10', 'M20', 'S10', 'S20', 'V10', 'V20'],
]
params.out_numbers = [
    [1, 1, 1, 1, -1, -1, -1],
    [1, 2, 3, 4, 5, 6, -1],
]
params.must_match = [
    [],
    np.arange(6),
]

params.freeze()

mnefun.do_processing(
    params,
    fetch_raw=False,  # Fetch raw recording files from acq machine
    do_score=False,  # do scoring
    push_raw=True,  # Push raw files and SSS script to SSS workstation
    do_sss=True,  # Run SSS locally
    fetch_sss=True,  # Fetch SSSed files
    do_ch_fix=True,  # Fix channel ordering
    gen_ssp=True,  # Generate SSP vectors
    apply_ssp=True,  # Apply SSP vectors and filtering
    write_epochs=True,  # Write epochs to disk
    gen_covs=True,  # Generate covariances
    gen_fwd=True,  # Generate forward solutions (and source space if needed)
    gen_inv=True,  # Generate inverses
    gen_report=True,    # Write html of results
    print_status=False,
)
