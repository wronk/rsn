"""
gen_motor_epochs.py

@author: wronk

Script to loop over HCP subjects and compute motor epochs
"""

from os import path as op
import numpy as np

import hcp

from rsn import config as cf
from rsn.config import hcp_path, motor_params
from rsn.comp_fun import (check_and_create_dir, preproc_annot_filter,
                          preproc_artifacts, preproc_epoch)

stored_subjects_dir = '/media/Toshiba/MRI_Data/structurals'
new_subjects_dir = op.join(hcp_path, 'anatomy')
head_trans_dir = op.join(hcp_path, 'hcp-meg')

n_jobs = 6
exp_type = 'motor'

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

hcp_params = dict(hcp_path=hcp_path, data_type='task_' + exp_type)

for subj_fold in dirs:
    hcp_params['subject'] = subj_fold

    # Construct folder for saving epochs
    epo_fold = op.join(hcp_path, subj_fold, 'epochs')
    check_and_create_dir(epo_fold)

    # Construct trial info for each run
    # Note: need unprocessed data and preprocessed data (to get trial info
    #       information in this step)
    trial_infos = []
    for run_index in motor_params['runs']:
        hcp_params['run_index'] = run_index
        trial_info = hcp.read_trial_info(**hcp_params)
        trial_infos.append(trial_info)

    print('\n' + 'Data info:' + '\n')
    print(trial_info['stim']['comments'][:10])  # which column? # 3
    #print(set(trial_info['stim']['codes'][:, 3]))  # check values

    all_events = []
    for trial_info in trial_infos:
        events = np.c_[
            trial_info['stim']['codes'][:, motor_params['time_samp_col']] - 1,
            np.zeros(len(trial_info['stim']['codes'])),
            trial_info['stim']['codes'][:, motor_params['stim_code_col']]
        ].astype(int)

        # Unfortunately, HCP data the time events may not always be unique
        unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]
        events = events[unique_subset]  # use diff to find first unique events

        all_events.append(events)

    ###########################################################################
    # Construct epochs

    for run_index, events in zip(motor_params['runs'], all_events):

        hcp_params['run_index'] = run_index

        raw = hcp.read_raw(**hcp_params)
        raw.load_data()

        #################
        # Preprocess data
        raw, annots = preproc_annot_filter(raw, hcp_params)

        # Use ICA components to remove EOG ECG (Note: assumes bad chans removed
        raw = preproc_artifacts(raw, hcp_params, annots)

        #################
        # Epoch data
        epochs = preproc_epoch(subj_fold, run_index, raw, events,
                               exp_type='task_' + exp_type)

        #################
        # Save epochs
        epo_fname = op.join(epo_fold, '%s_%s_run%i-epo.fif' %
                            (subj_fold, exp_type, run_index))
        epochs.save(epo_fname)

        del raw, epochs
