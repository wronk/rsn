"""
comp_dynamics.py

@author: wronk

Script to load synchronization data and use nonlinear dynamics to analyze
"""

import os
import os.path as op
import numpy as np
import cPickle
from copy import deepcopy

import mne
from mne import (read_epochs, read_source_spaces,
                 extract_label_time_course as eltc)
from mne.minimum_norm import (apply_inverse_epochs as apply_inv,
                              read_inverse_operator as read_inv)
from mne.connectivity import spectral_connectivity as conn
from mne.label import read_label

from matplotlib import pyplot as plt


################
# Define globals
################

data_dir = op.join(os.environ['CODE_ROOT'], 'rsn_data')
save_dir = op.join(os.environ['CODE_ROOT'], 'rsn_results')
struct_dir = os.environ['SUBJECTS_DIR']

subj_nums = ['004', '007', '015', '017', '019', '020', '023', '031', '032',
             '034', '038']
subj_nums = ['017']


def plot_auto_corr(ax, data, freq):
    ax.plot(auto_corr)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Time lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Freq: %s Hz' % freq)

# Load data for subject(s) of interest
for subj in subj_nums:
    subj_name = 'wronk_resting_%s' % subj
    # Save results as pkl file
    load_file = op.join(data_dir, subj_name, 'connectivity',
                        'conn_results.pkl')
    with open(load_file, 'rb') as pkl_obj:
        conn = cPickle.load(pkl_obj)
    methods = [conn['conn_methods'][ii]['method']
               for ii in range(len(conn['conn_methods']))]

    # Iterate through all sychronization metrics
    for cmi, conn_method_dict in enumerate(conn['conn_methods']):
        conn_method = conn_method_dict['method']

        # Iterate through synchronization pairs
        for si, sync_pair in enumerate(conn['conn_pairs']):
            # Get frequencies of interest
            freqs = conn['conn_methods'][cmi]['cwt_frequencies']

            # Set up plotting
            fig_ac, axes_ac = plt.subplots(nrows=len(freqs),
                                           figsize=(2 * len(freqs), 8),
                                           sharex=True)
            fname_save_ac = op.join(save_dir, subj_name, 'autocorrelation',
                                    '%s_%s.png' % (conn_method, str(sync_pair)))

            # Iterate through all frequencies
            for fi, freq in enumerate(freqs):
                sync_data = conn['conn_data'][cmi][si, fi, :]

                # Compute autocorrelation
                auto_corr = np.correlate(sync_data, sync_data, mode='full')
                auto_corr /= np.max(auto_corr)
                auto_corr = auto_corr[len(auto_corr) / 2:]
                plot_auto_corr(axes_ac[fi], auto_corr, freq)

                # Plot and save phase plot

            fig_ac.savefig(fname_save_ac)


def get_all_time_embed(signal, tau=1, dim=3):
    """Helper to get all time delay embedded vectors"""

    last_ind = len(signal) - (len(signal) % (tau * dim))
    full_embedding = np.zeros((len(signal) - last_ind, dim))

    for ind in np.arange(0, last_ind):
        full_embedding[ind, :] = time_embed(signal, tau, dim, start=ind)


def time_embed(signal, tau=1, dim=3, start=0):
    """Function to time embed a signal
    Parameters:
    ==========
    signal: np.array; shape = (n_sig,)

    tau: int
        Delay in samples
    dim: int
        Dimension of embedded vector
    Returns:
    =======
    sig: array
        Time embedded signal.

    """
    last_ind = start + tau * dim
    assert len(signal) > last_ind, 'Can\'t get data point beyond end of signal'

    # XXX Could pass inds or bool mask instead
    inds = np.arange(start, last_ind, tau)
    return signal[inds]


# Compute autocorrelation

###############################
# Compute time embedded vectors
###############################

# Autocorrelation, and time delayed mutual information should help guide the
# parameter choices here
# Some plot just PCA components

# Compute Lyapunov exponents
