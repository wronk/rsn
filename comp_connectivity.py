"""
comp_connectivity.py

@author: wronk

Compute connectivity metrics between RSN areas

Handling eric_voc data: This code computes connectivity on trials individually
and does not splice them together prior to computing connectivity. It also
crops trials to only the task portion of the epochs.
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

import config
from config import config_proc_methods, config_conn_params
from comp_fun import tfr_split

################
# Define globals
################
save_data = False  # Whether or not to pickle data
shuffle_data = False  # Whether or not to shuffle data in time (as a control)

struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])

# Choose to process resting data or task data
#exp_heading = 'wronk_resting'
exp_heading = 'eric_voc'

subj_nums = config.subj_nums
inv_lam = config.inv_lambda

proc_methods = config_proc_methods
conn_params = config_conn_params
conn_params.update(dict(subj_nums=['%03i' % s_num for s_num in subj_nums]))


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


'''
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
'''


def calc_corr(wavelet_ts, conn_params, mode):
    """Helper to calculate correlation between power bands"""

    blp_corr = np.zeros((wavelet_ts.shape[0], len(conn_params['conn_pairs'][0]),
                         wavelet_ts.shape[2], wavelet_ts.shape[3]))

    if mode == 'wronk_resting':
        # Loop over each label pair
        for match_i, (li_1, li_2) in enumerate(zip(conn_params['conn_pairs'][0],
                                                   conn_params['conn_pairs'][1])):
            # Calculate sliding correlation
            for bp_i in range(wavelet_ts.shape[2]):
                blp_corr[0, match_i, bp_i, :] = \
                    rolling_corr(wavelet_ts[0, li_1, bp_i, :],
                                 wavelet_ts[0, li_2, bp_i, :], window=corr_len)

    elif mode == 'eric_voc':
        blp_corr = np.zeros((wavelet_ts.shape[0], len(conn_params['conn_pairs'][0]),
                             wavelet_ts.shape[2], wavelet_ts.shape[3]))
        # Loop over each trial
        for ti in range(wavelet_ts.shape[0]):
            # Loop over each label pair
            for match_i, (li_1, li_2) in enumerate(zip(conn_params['conn_pairs'][0],
                                                       conn_params['conn_pairs'][1])):
                # Loop over each power band
                for bp_i in range(wavelet_ts.shape[2]):
                    # Calculate sliding correlation
                    blp_corr[ti, match_i, bp_i, :] = \
                        rolling_corr(wavelet_ts[ti, li_1, bp_i, :],
                                     wavelet_ts[ti, li_2, bp_i, :],
                                     window=corr_len)
    else:
        raise RuntimeError('`mode` incorrect')

    return blp_corr

########################
# Set up a few variables
########################

if exp_heading == 'wronk_resting':
    data_dir = op.join(data_head, 'rsn_data')
elif exp_heading == 'eric_voc':
    data_dir = op.join(data_head, 'voc_data')
else:
    raise RuntimeError('Incorrect experimental heading')

###########################################################
# Loop through each subject; load info, comp connectivity
###########################################################
print '\nShuffle data: ' + str(shuffle_data)
print 'Subjects: ' + str(subj_nums) + '\n'

for s_num in subj_nums:
    num_exp = '%03i' % s_num
    s_name = 'AKCLEE_1%02i' % s_num

    print '\n%s\nProcessing: %s\n' % ('=' * 40, s_name)
    print '\tLoading epo, inv, src, labels'

    # Load Epochs
    epo = read_epochs(op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                              'epochs', 'All_55-sss_%s_%s-epo.fif' %
                              (exp_heading, num_exp)))
    epo.pick_types(meg=True, eeg=True)

    # Generate source activity restricted to pre-defined RSN areas
    # XXX Using MEG only ERM inverse
    fname_inv = op.join(data_head, 'voc_data', 'eric_voc_%s' % num_exp,
                        'inverse', 'eric_voc_%s-55-sss-meg-erm-fixed-inv.fif' %
                        num_exp)
    inv = read_inv(fname_inv)

    # XXX: 'src' object in structs_dir doesn't match src file in `struct_dir`
    src = inv['src']

    # Subselect only vocoder trials (and throw away visual control trials)
    if exp_heading == 'eric_voc':
        epo.crop(None, 5.2)  # Crop to end at last stimulus letter

    # Randomly (and independently) shuffle time axis of each epoch data trial
    if shuffle_data:
        print '\tShuffling data'
        for t_i in range(epo._data.shape[0]):
            for c_i in range(epo._data.shape[1]):
                epo._data[t_i, c_i, :] = epo._data[t_i, c_i, np.random.permutation(epo._data.shape[2])]

    print '\n\tLoading RSN label info'
    # Get list of labels and vertices
    lab_list, lab_verts = get_lab_list(conn_params, s_name, src)

    print '\n\tComputing STCs'
    stc_list = apply_inv(epo, inv, lambda2=inv_lam, method='MNE')
    #lab_verts_restricted = np.intersect1d(lab_verts, src[0]['vertno'])
    #lab_verts_restricted.sort()
    # XXX downstream problems when trying to use only labels of interest eltc
    #summed_label = mne.Label(lab_verts_restricted, hemi='lh',
    #                         name='RSNs_summed', subject=s_name)
    # requires that n_vertices match between src and stc
    #stc_list = apply_inv(epo, inv, lambda2=inv_lam, label=summed_label,
    #                     method='MNE')

    # Extract label time course (n_stcs x n_label x n_times)
    label_activity = np.array(eltc(stc_list, lab_list, src,
                                   mode=conn_params['mean_mode'],
                                   verbose=False))

    ###########################################################
    # Compute connectivity between RSN regions for each subject
    ###########################################################
    print '\tComputing connectivity'
    conn_data = []
    power_data = []
    for conn_dict in conn_params['proc_methods']:
        # Sliding correlation of BLP
        # Compute TFR of power bands
            # wavelet_ts is (n_stc, n_lab, n_freq, n_times)
        conn_dict.update(sfreq=epo.info['sfreq'])
        wavelet_ts = tfr_split(label_activity, conn_dict)

        corr_len = int(conn_dict['corr_wind'] * epo.info['sfreq'])

        if exp_heading == 'wronk_resting':
            # Reshape all stc ROI data into long time traces
            wavelet_ts = np.rollaxis(wavelet_ts, 0, 3)
            wavelet_ts = wavelet_ts.reshape(1, wavelet_ts.shape[0],
                                            wavelet_ts.shape[1],
                                            (wavelet_ts.shape[2] *
                                             wavelet_ts.shape[3]))
            blp_corr = calc_corr(wavelet_ts, conn_params, exp_heading)

        elif exp_heading == 'eric_voc':
            blp_corr = calc_corr(wavelet_ts, conn_params, exp_heading)

        #power_data.append(wavelet_ts)  # Store raw power data
        conn_data.append(blp_corr)  # Store connectivity traces

    ##############
    # Save Results
    ##############
    if save_data:
        print '\tSaving connectivity data'

        # Check if save directory exists (and create it if necessary)
        save_dir = op.join(data_dir, '%s_connectivity' % (exp_heading))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Save results as pkl file
        shuf_addition = 'shuffled_' if shuffle_data else ''
        save_file = op.join(save_dir, 'conn_results_%s%s_%s.pkl' % (
            shuf_addition, exp_heading, num_exp))
        with open(save_file, 'wb') as pkl_obj:
            results_to_save = deepcopy(conn_params)
            results_to_save['conn_data'] = conn_data
            #results_to_save['power_data'] = power_data
            results_to_save['conn_data_shape'] = \
                'n_trials, n_label_pairs, n_freqs, n_times'
            results_to_save['sfreq'] = epo.info['sfreq']
            results_to_save['event_id'] = epo.event_id
            results_to_save['events'] = epo.events

            cPickle.dump(results_to_save, pkl_obj)
