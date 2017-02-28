"""
comp_fun.py

@author: wronk

Various extra helper functions
"""
import os
import os.path as op
import numpy as np

from pandas import rolling_corr

import mne
from mne import (read_epochs, read_proj, read_label, concatenate_epochs,
                 write_proj)
from mne.time_frequency.tfr import _compute_tfr
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import hcp
from hcp import preprocessing as preproc
from hcp.preprocessing import interpolate_missing as interp_missing

import config as cf
from config import (motor_params, rest_params, hcp_path, filt_params)


def shuffle_epo(epo):
    # Randomly (and independently) shuffle time axis of each epoch data trial
    print('\nSHUFFLING DATA\n')
    for t_i in range(epo._data.shape[0]):
        for c_i in range(epo._data.shape[1]):
            epo._data[t_i, c_i, :] = epo._data[t_i, c_i, np.random.permutation(epo._data.shape[2])]

    return epo


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


def get_lab_list(conn_params, s_name):
    """Helper to load a list of labels and all vertices for a subject

    Parameters
    ==========
    conn_params: dict
        Contains `rsn_labels` which is a list of labels to load
    s_name: str
        Subject name as it exists in subjects_dir

    Returns
    ========
    lab_list: list of Labels
        List of mne.Label
    """

    assert('rsn_labels' in conn_params.keys())
    print 'Loading %i labels for subject: %s' % \
        (len(conn_params['rsn_labels']), s_name)

    lab_list = []
    #lab_vert_list = []

    for label_name in conn_params['rsn_labels']:
        fname_label = op.join(os.environ['SUBJECTS_DIR'], s_name, 'label',
                              label_name)
        lab_list.append(read_label(fname_label, subject=s_name))

        #lab_vert_list.extend(list(lab_list[-1].vertices))

    # TODO: need to separate based on hemi if this is used
    #lab_vert_arr = np.array(lab_vert_list)

    return lab_list


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

    # XXX: check if proj should be false
    epo_list = []
    for run_i in runs:
        epo_fname = op.join(hcp_path, '%s' % subject, 'epochs',
                            '%s_%s_run%i-epo.fif' % (subject, exp_type, run_i))
        epo = read_epochs(epo_fname, proj=False)
        epo_list.append(epo)

    print '\nConcatenating %i epoch files' % len(epo_list)
    return concatenate_epochs(epo_list)


def preproc_annot_filter(subj_fold, raw, hcp_params):
    """Helper to annotate bad segments and apply Butterworth freq filtering"""

    # apply ref channel correction
    preproc.apply_ref_correction(raw)

    # construct MNE annotations
    annots = hcp.read_annot(**hcp_params)
    bad_seg = (annots['segments']['all']) / raw.info['sfreq']
    annotations = mne.Annotations(
        bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
        description='bad')

    raw.annotations = annotations

    # Read all bad channels (concatenated from all runs)
    bad_ch_file = op.join(hcp_path, subj_fold, 'unprocessed', 'MEG',
                          'prebad_chans.txt')
    with open(bad_ch_file, 'rb') as prebad_file:
        prebad_chs = prebad_file.readlines()
    raw.info['bads'].extend([ch.strip() for ch in prebad_chs])  # remove '\n'

    #raw.info['bads'].extend(annots['channels']['all'])
    #print('Bad channels added: %s' % annots['channels']['all'])

    # Band-pass filter (by default, only operates on MEG/EEG channels)
    # Note: MNE complains on Python 2.7
    raw.filter(filt_params['lp'], filt_params['hp'],
               method=filt_params['method'],
               filter_length=filt_params['filter_length'],
               l_trans_bandwidth='auto',
               h_trans_bandwidth='auto',
               phase=filt_params['phase'],
               n_jobs=-1)

    return raw, annots


def preproc_gen_ssp(subj_fold, raw, hcp_params, annots):
    """Helper to apply artifact removal on raw data

    NOTE: Only use this on one run of data to create projectors. Subsequent
    runs should load and apply the same projectors. This is required to make
    sure machine learning algorithm isn't classifying differences in projectors
    (like the ICA projectors shipped with HCP) instead of differences in
    activity."""

    proj_dir = op.join(hcp_path, subj_fold, 'ssp_pca_fif')
    check_and_create_dir(proj_dir)

    # Compute EOG and ECG projectors
    # XXX Note: do not add these to raw obj, as all raw files need to use the
    #     same projectors. Instead, save and then reapply later
    preproc.set_eog_ecg_channels(raw)

    proj_eog1, eog_events1 = compute_proj_eog(raw, ch_name='HEOG', n_jobs=-1)
    proj_eog2, eog_events2 = compute_proj_eog(raw, ch_name='VEOG', n_jobs=-1)
    proj_ecg, ecg_events = compute_proj_ecg(raw, ch_name='ECG', n_jobs=-1)

    # Save to disk so these can be used in future processing
    all_projs = proj_eog1 + proj_eog2 + proj_ecg
    write_proj(op.join(proj_dir, 'preproc_all-proj.fif'), all_projs)


def preproc_epoch(subj_fold, raw, run_index, events, exp_type):
    """Helper to convert raw data into epochs"""

    if exp_type == 'rest':
        epo_params = rest_params
    elif exp_type == 'task_motor':
        epo_params = motor_params
    else:
        raise RuntimeError('exp_type must be `rest` or `task_motor`, got %s'
                           % exp_type)

    # Read projectors and add them to raw
    all_projs = read_proj(op.join(hcp_path, subj_fold, 'ssp_pca_fif',
                                  'preproc_all-proj.fif'))
    raw.add_proj(all_projs)

    # Create and save epochs
    #     Baseline?, XXX update rejects
    raw.pick_types(meg=True, ref_meg=False)

    events = np.sort(events, 0)
    epochs = mne.Epochs(raw, events=events,
                        event_id=epo_params['event_id'],
                        tmin=epo_params['tmin'],
                        tmax=epo_params['tmax'],
                        reject=cf.epo_reject, baseline=epo_params['baseline'],
                        decim=epo_params['decim'], preload=True)

    '''
    # Add back out channels for comparison across runs
    # XXX: Note that interp_missing loads `info` object shipped with HCP and
    #    seems to replace it. Avoid using as this will overwrite SSP projectors
    epochs = interp_missing(epochs, subject=subj_fold,
                            data_type=exp_type, run_index=run_index,
                            hcp_path=hcp_path, mode='accurate')
                            '''
    return epochs


def calc_corr(power_arr, conn_params):
    """Helper to calculate correlation between power bands"""

    blp_corr = np.zeros((power_arr.shape[0], len(conn_params['conn_pairs'][0]),
                         power_arr.shape[2], power_arr.shape[3]))
    lab_pairs = zip(conn_params['conn_pairs'][0], conn_params['conn_pairs'][1])
    # Loop over each trial
    for ti in range(power_arr.shape[0]):
        # Loop over each label pair
        for match_i, (li_1, li_2) in enumerate(lab_pairs):
            # Loop over each power band
            for bp_i in range(power_arr.shape[2]):
                # Calculate sliding correlation
                blp_corr[ti, match_i, bp_i, :] = \
                    rolling_corr(power_arr[ti, li_1, bp_i, :],
                                 power_arr[ti, li_2, bp_i, :],
                                 window=corr_len)

    return blp_corr
