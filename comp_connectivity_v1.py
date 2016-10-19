"""
comp_connectivity_v1.py

@author: wronk

Compute connectivity metrics between RSN areas

Handling voc_meg data: This code takes all auditory trials and stitchs them
together into one long time-series before computing correlations. This isn't
ideal as the time series will often jump to the start of the next trial.

Output data is saved as (n_ROIs, n_freqs, n_times).
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
from mne.time_frequency.tfr import _compute_tfr
from pandas import rolling_corr

from config import config_conn_methods, config_conn_params

################
# Define globals
################
save_data = True  # Whether or not to pickle data

struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])

# Choose to process resting data or task data
#exp_heading = 'wronk_resting'
exp_heading = 'eric_voc'


if exp_heading == 'wronk_resting':
    data_dir = op.join(data_head, 'rsn_data')
else:
    data_dir = op.join(data_head, 'voc_data')
    trial_types = ['M10_', 'M20_', 'S10_', 'S20_']

subj_nums = ['04', '07', '15', '17', '19', '20', '23', '31', '32', '34', '38']
subj_nums = ['15']

conn_methods = config_conn_methods
conn_params = config_conn_params
conn_params.update(dict(subj_nums=['0%s' % s_num for s_num in subj_nums]))

inv_lam = 1. / 9


##################
# Helper functions
##################
def get_lab_list(conn_params, s_name, src):
    """Helper to get list of labels and all vertices"""

    assert('rsn_labels' in conn_params.keys())
    print 'Loading %i labels for subject: %s' % \
        (len(conn_params['rsn_labels']), s_name)

    lab_list = []
    lab_vert_list = []

    for label_name in conn_params['rsn_labels']:
        # Load label, get vertices
        fname_label = op.join(struct_dir, s_name, 'label', label_name)
        temp_label = read_label(fname_label, subject=s_name)

        lab_list.append(temp_label)
        lab_vert_list.extend(list(lab_list[-1].vertices))

    return lab_list, lab_vert_list


def tfr_split(label_activity):
    """Helper to calculate wavelet power in batches instead of all at once

    Needed for memory issues.
    """
    batch_size = 2
    batch_list = []

    power_arr = np.zeros((label_activity.shape[0], label_activity.shape[1],
                          len(conn_dict['cwt_frequencies']),
                          label_activity.shape[2]))

    batch_inds = range(0, label_activity.shape[0], batch_size)
    batch_inds.append(label_activity.shape[0])

    for bi1, bi2 in zip(batch_inds[:-1], batch_inds[1:]):

        batch = _compute_tfr(label_activity[bi1:bi2],
                             frequencies=conn_dict['cwt_frequencies'],
                             sfreq=epo.info['sfreq'],
                             n_cycles=conn_dict['n_cycles'],
                             n_jobs=1, output='power')
        power_arr[bi1:bi2, :, :, :] = batch

    return power_arr

###########################################################
# Loop through each subject; load info, comp connectivity
###########################################################

for s_num in subj_nums:
    num_exp = '0%s' % s_num
    s_name = 'AKCLEE_1' + s_num
    print '\nProcessing: ' + s_name

    print '\n\tLoading epo, inv, src, labels'
    # Load Epochs
    epo = read_epochs(op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                              'epochs', 'All_55-sss_%s_%s-epo.fif' %
                              (exp_heading, num_exp)))

    # Subselect only vocoder trials (and throw away visual control trials)
    if exp_heading == 'voc_meg':
        epo = epo[trial_types]
        #epo.crop(None, 5.2)  # Crop to end at last stimulus letter

    # Generate source activity restricted to pre-defined RSN areas
    # XXX Using MEG only ERM inverse
    fname_inv = op.join(data_head, 'voc_data', 'eric_voc_%s' % num_exp,
                        'inverse', 'eric_voc_%s-55-sss-meg-erm-fixed-inv.fif' %
                        num_exp)
    inv = read_inv(fname_inv)

    # XXX: 'src' object in structs_dir doesn't match src file in `struct_dir`
    src = inv['src']

    print '\n\tLoading RSN label info'
    # Get list of labels and vertices
    lab_list, lab_verts = get_lab_list(conn_params, s_name, src)
    lab_verts_restricted = np.intersect1d(lab_verts, src[0]['vertno'])
    lab_verts_restricted.sort()

    summed_label = mne.Label(lab_verts_restricted, hemi='lh',
                             name='RSNs_summed', subject=s_name)

    print '\n\tComputing STCs'
    stc_list = apply_inv(epo, inv, lambda2=inv_lam, method='MNE')
    # XXX downstream problems when trying to use only labels of interest eltc
    # requires that n_vertices match between src and stc
    #stc_list = apply_inv(epo, inv, lambda2=inv_lam, label=summed_label,
    #                     method='MNE')

    # Extract label time course (n_stcs x n_label x n_times)
    label_activity = np.array(eltc(stc_list, lab_list, src,
                                   mode=conn_params['mean_mode'],
                                   verbose=False))

    import ipdb; ipdb.set_trace()
    ###########################################################
    # Compute connectivity between RSN regions for each subject
    ###########################################################
    print '\tComputing connectivity'
    conn_data = []
    power_data = []
    for conn_dict in conn_params['conn_methods']:
        # Sliding correlation of BLP
        # Compute TFR of power bands
            # wavelet_ts is (n_stc, n_lab, n_freq, n_times)
        wavelet_ts = tfr_split(label_activity)

        # Reshape to concatenate data from all epochs into one long time trace
        wavelet_ts = wavelet_ts.reshape(1, label_activity.shape[1],
                                        len(conn_dict['cwt_frequencies']),
                                        (label_activity.shape[0] *
                                            label_activity.shape[2]))
        power_data.append(wavelet_ts)

        # Calculate sliding correlation window using pandas.rolling_corr
        corr_len = int(conn_dict['corr_wind'] * epo.info['sfreq'])
        blp_corr = np.zeros((len(conn_params['conn_pairs'][0]),
                             wavelet_ts.shape[2],
                             wavelet_ts.shape[3]))

        for match_i, (li_1, li_2) in enumerate(zip(conn_params['conn_pairs'][0],
                                                   conn_params['conn_pairs'][1])):
            for bp_i in range(wavelet_ts.shape[2]):
                blp_corr[match_i, bp_i] = \
                    rolling_corr(wavelet_ts[0, li_1, bp_i, :],
                                 wavelet_ts[0, li_2, bp_i, :], window=corr_len)

        conn_data.append(blp_corr)  # Store only connectivity traces

    ##############
    # Save Results
    ##############
    print '\tSaving connectivity data'
    if save_data:

        # Check if save directory exists (and create it if necessary)
        save_dir = op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                           'connectivity')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Save results as pkl file
        save_file = op.join(save_dir, 'conn_results_%s.pkl' % exp_heading)
        with open(save_file, 'wb') as pkl_obj:
            results_to_save = deepcopy(conn_params)
            results_to_save['conn_data'] = conn_data
            #results_to_save['power_data'] = power_data
            results_to_save['conn_data_shape'] = \
                'n_label_pairs, n_freqs, n_times'
            results_to_save['sfreq'] = epo.info['sfreq']

            cPickle.dump(results_to_save, pkl_obj)
