"""
comp_fun.py

@author: wronk

Various extra helper functions
"""
import os
import numpy as np

from mne.time_frequency.tfr import _compute_tfr


def tfr_split(data, processing_params):
    """Helper to calculate wavelet power in batches instead of all at once

    Needed for memory issues.
    """
    batch_size = 2
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
                             n_jobs=1, output='power')
        power_arr[bi1:bi2, :, :, :] = batch

    return power_arr


def check_and_create_dir(fold):
    """If directory doesn't exist, create it"""
    if not os.path.exists(fold):
        os.makedirs(fold)
