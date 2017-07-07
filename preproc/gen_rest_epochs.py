"""
gen_rest_epochs.py

@author: wronk

Script to loop over HCP subjects, compute rest epochs
"""

from os import path as op
import numpy as np

from mne import make_fixed_length_events as make_events
import hcp

from rsn import config as cf
from rsn.config import hcp_path, rest_params
from rsn.comp_fun import (check_and_create_dir, preproc_annot_filter,
                          preproc_epoch, scale_epo_data)

n_jobs = 6
exp_type = 'rest'
trial_types = ['rest']

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]

hcp_params = dict(hcp_path=hcp_path, data_type=exp_type)

for subj_fold in dirs:
    hcp_params['subject'] = subj_fold

    # Construct folder for saving epochs
    epo_fold = op.join(hcp_path, subj_fold, 'epochs')
    check_and_create_dir(epo_fold)

    ###########################################################################
    # Construct epochs

    for run_index in rest_params['runs']:

        hcp_params['run_index'] = run_index

        raw = hcp.read_raw(**hcp_params)
        raw.load_data()

        # Make 1 second events; assign event code from params, shouldn't
        #     overlap with the motor task 'fixate' events
        events = make_events(raw, id=rest_params['event_id']['rest'],
                             duration=1.)

        #################
        # Preprocess data
        raw, annots = preproc_annot_filter(subj_fold, raw, hcp_params,
                                           apply_ref_correction=rest_params['ref_correction'])

        #################
        # Epoch data

        epochs, unit_scaler = preproc_epoch(subj_fold, raw, run_index, events,
                                            exp_type, trial_types)
        if rest_params['scale_epo']:
            epochs = scale_epo_data(epochs, unit_scaler)

        #############################
        # Save epochs and unit scaler
        save_fname = op.join(epo_fold, '%s_%s_run%i-epo' %
                             (subj_fold, exp_type, run_index))

        epochs.save(save_fname + '.fif')
        np.save(save_fname + '_scale', unit_scaler)

        hcp.preprocessing.map_ch_coords_to_mne(epochs)
        fig = epochs.plot_psd_topomap(vmin=0, vmax=21, cmap='viridis', n_jobs=6, show=False)
        fig.savefig('/media/Toshiba/Code/hcp_data/%s/rest_epo_%i.png' % (subj_fold, run_index))

        fig = epochs.plot_psd(fmin=1, fmax=55, proj=False, n_jobs=6,
                              show=False)
        fig.gca().set_ylim([295, 315])
        fig.savefig('/media/Toshiba/Code/hcp_data/%s/z_rest_epo_%i.png' %
                    (subj_fold, run_index))

        del raw, epochs
