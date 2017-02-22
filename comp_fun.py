"""
comp_fun.py

@author: wronk

Various extra helper functions
"""
import os
import os.path as op
import numpy as np

import mne
from mne import read_epochs, concatenate_epochs
from mne.time_frequency.tfr import _compute_tfr
import hcp
from hcp import preprocessing as preproc
from hcp.preprocessing import interpolate_missing as interp_missing

import config as cf
from config import (motor_params, rest_params, hcp_path, filt_params)


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

    # XXX: check if proj should be false
    epo_list = []
    for run_i in runs:
        epo_fname = op.join(hcp_path, '%s' % subject, 'epochs',
                            '%s_%s_run%i-epo.fif' % (subject, exp_type, run_i))
        epo = read_epochs(epo_fname, proj=False)
        epo_list.append(epo)

    print '\nConcatenating %i epoch files' % len(epo_list)
    return concatenate_epochs(epo_list)


def preproc_annot_filter(raw, hcp_params):
    """Helper to annotate bad segments and apply Butterworth freq filtering"""

    # apply ref channel correction and drop ref channels
    preproc.apply_ref_correction(raw)

    # construct MNE annotations
    annots = hcp.read_annot(**hcp_params)
    bad_seg = (annots['segments']['all']) / raw.info['sfreq']
    annotations = mne.Annotations(
        bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
        description='bad')

    raw.annotations = annotations
    raw.info['bads'].extend(annots['channels']['all'])
    raw.pick_types(meg=True, ref_meg=False)

    # Note: MNE complains on Python 2.7
    raw.filter(filt_params['lp'], filt_params['hp'],
               method=filt_params['method'],
               iir_params=filt_params['iir_params'], n_jobs=-1)

    return raw, annots


def preproc_artifacts(raw, hcp_params, annots):
    """Helper to apply artifact removal on raw data"""
    # Read ICA and remove EOG ECG
    # Note that the HCP ICA assumes that bad channels have been removed
    ica_mat = hcp.read_ica(**hcp_params)

    # We will select the brain ICs only
    #exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
    #           if ii not in annots['ica']['brain_ic_vs']]
    exclude = annots['ica']['ecg_eog_ic']

    preproc.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)

    return raw


def preproc_epoch(subj_fold, run_index, raw, events, exp_type):
    """Helper to convert raw data into epochs"""

    if exp_type == 'rest':
        epo_params = rest_params
    elif exp_type == 'task_motor':
        epo_params = motor_params
    else:
        raise RuntimeError('exp_type must be `rest` or `task_motor`, got %s'
                           % exp_type)

    # Create and save epochs
    #     Baseline?, XXX update rejects
    events = np.sort(events, 0)
    epochs = mne.Epochs(raw, events=events,
                        event_id=epo_params['event_id'],
                        tmin=epo_params['tmin'],
                        tmax=epo_params['tmax'],
                        reject=None, baseline=epo_params['baseline'],
                        decim=epo_params['decim'], preload=True)

    # Add back out channels for comparison across runs
    epochs = interp_missing(epochs, subject=subj_fold,
                            data_type=exp_type, run_index=run_index,
                            hcp_path=hcp_path, mode='accurate')
    return epochs
