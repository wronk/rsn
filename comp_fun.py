"""
comp_fun.py

@author: wronk

Various extra helper functions
"""
import os
import os.path as op
import numpy as np

from mne import read_epochs, concatenate_epochs
from mne.time_frequency.tfr import _compute_tfr

import config as cf
from config import motor_params, rest_params, hcp_path


def tfr_split(data, processing_params):
    """Helper to calculate wavelet power in batches instead of all at once

    Needed for memory issues.
    """
    '''
    batch_size = 50  # Number trials to do at once
    batch_list = []

    power_arr = np.zeros((data.shape[0], data.shape[1],
                          len(processing_params['cwt_frequencies']),
                          data.shape[2]))

    batch_inds = range(0, data.shape[0], batch_size)
    batch_inds.append(data.shape[0])

    for bi1, bi2 in zip(batch_inds[:-1], batch_inds[1:]):

        batch = _compute_tfr(data[bi1:bi2],
                             frequencies=processing_params['cwt_frequencies'],
                             sfreq=processing_params['sfreq'],
                             n_cycles=processing_params['n_cycles'],
                             decim=processing_params['post_decim'],
                             n_jobs=6, output='power')
        power_arr[bi1:bi2, :, :, :] = batch
    '''

    batch = _compute_tfr(data,
                         frequencies=processing_params['cwt_frequencies'],
                         sfreq=processing_params['sfreq'],
                         n_cycles=processing_params['n_cycles'],
                         decim=processing_params['post_decim'],
                         n_jobs=processing_params['n_jobs'], output='power')

    return batch


def check_and_create_dir(fold):
    """If directory doesn't exist, create it"""
    if not os.path.exists(fold):
        os.makedirs(fold)


def get_concat_epos(subject, exp_type):
    """Load all epochs for one experiment and return concatenated object

    Parameters
    ==========
    subject: str
        Subject directory in string form
    exp_type: str
        "motor" or "rest"

    Returns
    =======
    epochs: Epochs
    """
    if exp_type is 'motor':
        runs = cf.motor_params['runs']
    elif exp_type is 'rest':
        runs = cf.rest_params['runs']
    else:
        raise RuntimeError('Incorrect trial designation: %s' % exp_type)

    epo_list = []
    for run_i in runs:
        epo = read_epochs(op.join(hcp_path, '%s' % subject, 'epochs',
                          '%s_%s_run%i-epo.fif' % (subject, exp_type, run_i)))
        epo_list.append(epo)

    print '\nConcatenating %i epoch files' % len(epo_list)
    return concatenate_epochs(epo_list)
