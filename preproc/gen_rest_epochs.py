"""
gen_rest_epochs.py

@author: wronk

Script to loop over HCP subjects, compute rest epochs
"""

from os import path as op

import mne
import hcp

from rsn import config as cf
from rsn.config import hcp_path, rest_params
from rsn.comp_fun import (check_and_create_dir, preproc_annot_filter,
                          preproc_gen_ssp, preproc_epoch)

n_jobs = 6
exp_type = 'rest'

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

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

        # Make 1 second events
        events = mne.make_fixed_length_events(raw, id=1, duration=1.)

        #################
        # Preprocess data
        raw, annots = preproc_annot_filter(subj_fold, raw, hcp_params)

        #################
        # Epoch data
        epochs = preproc_epoch(subj_fold, raw, run_index, events,
                               exp_type=exp_type)

        #################
        # Save epochs
        epo_fname = op.join(epo_fold, '%s_%s_run%i-epo.fif' %
                            (subj_fold, exp_type, run_index))
        epochs.save(epo_fname)

        del raw, epochs
