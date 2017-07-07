"""
gen_motor_epochs.py

Script to loop over HCP subjects and compute motor epochs
"""

from os import path as op
import numpy as np

import hcp

from rsn import config as cf
from rsn.config import hcp_path, motor_params
from rsn.comp_fun import (check_and_create_dir, preproc_annot_filter,
                          preproc_gen_ssp, preproc_epoch, scale_epo_data)

n_jobs = 6
exp_type = 'task_motor'
trial_types = ['LH', 'RH']#, 'LF', 'RF']
#trial_types = ['fixate']

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]

hcp_params = dict(hcp_path=hcp_path, data_type=exp_type)

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
        raw, annots = preproc_annot_filter(subj_fold, raw, hcp_params,
                                           apply_ref_correction= motor_params['ref_correction'])

        epochs, unit_scaler = preproc_epoch(subj_fold, raw, run_index, events,
                                            exp_type, trial_types)
        if motor_params['scale_epo']:
            epochs = scale_epo_data(epochs, unit_scaler)

        #############################
        # Save epochs and unit scaler
        if trial_types[0] == 'fixate':
            save_fname = op.join(epo_fold, '%s_%s_run%i-epo' %
                                 (subj_fold, 'rest', run_index))
            plot_type = 'rest'
        else:
            save_fname = op.join(epo_fold, '%s_%s_run%i-epo' %
                                 (subj_fold, 'motor', run_index))
            plot_type = 'motor'

        epochs.save(save_fname + '.fif')
        np.save(save_fname + '_scale', unit_scaler)

        hcp.preprocessing.map_ch_coords_to_mne(epochs)
        #fig = epochs.plot_psd_topomap(vmin=-250, vmax=-230, cmap='viridis', n_jobs=6, show=False)
        fig = epochs.plot_psd_topomap(vmin=0, vmax=21, cmap='viridis', n_jobs=6, show=False)
        fig.savefig('/media/Toshiba/Code/hcp_data/%s/%s_epo_%i.png' % (subj_fold, plot_type, run_index))

        fig = epochs.plot_psd(fmin=1, fmax=55, proj=False, n_jobs=6, show=False)
        fig.gca().set_ylim([295, 315])
        fig.savefig('/media/Toshiba/Code/hcp_data/%s/z_%s_epo_%i.png' % (subj_fold, plot_type, run_index))

        del raw, epochs
