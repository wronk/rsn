"""
comp_src_power.py

@author: wronk

Compute src power in ROIs for HCP data
"""
import os
import os.path as op
import cPickle
import numpy as np
from copy import deepcopy
from itertools import product

from mne import read_source_spaces
from mne.minimum_norm import read_inverse_operator as read_inv
from mne.minimum_norm import prepare_inverse_operator as prep_inv
from mne.minimum_norm import apply_inverse_epochs as apply_inv

import config as cf
from config import hcp_path, common_params, config_conn_params
from comp_fun import (get_concat_epos, check_and_create_dir, tfr_split,
                      shuffle_epo, get_lab_list)

save_data = True

n_jobs = 6
shuffles = [False, True]  # Whether or not to shuffle data in time (as a control)
exp_types = ['motor', 'rest']

stored_subjects_dir = os.environ['SUBJECTS_DIR']

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

###########################################################################
# Loop over all subjects; estimate source activity in pre-defined RSN areas
###########################################################################
for (s_num, exp_type, shuffle_data) in product(dirs, exp_types, shuffles):
    print('\n%s\nSubj: %s, Exp_type: %s, Shuffle:%s\n' %
          ('=' * 40, s_num, exp_type, shuffle_data))

    src_fold = op.join(stored_subjects_dir, s_num, 'src')
    # fsaverage shipped with HCP
    fs_avg_src = read_source_spaces(op.join(src_fold, 'fsaverage-src.fif'))

    # Get fsaverage labels
    lab_list = get_lab_list(config_conn_params, s_num)

    # XXX Using MEG only ERM inverse
    fname_inv = op.join(hcp_path, s_num, 'inverse', '%s_motor-inv.fif' % s_num)
    inv = read_inv(fname_inv)
    inv_prepped = prep_inv(inv, nave=1, lambda2=cf.inv_lambda, method='MNE')

    # Get all epochs
    epo = get_concat_epos(s_num, exp_type)

    if shuffle_data:
        epo = shuffle_epo(epo)

    common_params.update(sfreq=epo.info['sfreq'])

    # shape = (n_trials x n_labels x n_freqs x n_times)
    avg_lab_pow = np.zeros((len(epo.events), len(lab_list),
                            len(common_params['cwt_frequencies']),
                            int(np.ceil(float(len(epo.times)) /
                                        common_params['post_decim']))))

    print('Computing power at label vertices.')
    for li, lab in enumerate(lab_list):
        print('%s\t%s' % (40 * '-', lab.name))
        # Compute inverse estimates
        stcs = apply_inv(epo, inv_prepped, cf.inv_lambda, method='MNE',
                         label=lab, prepared=True)
        # Morph back to fsaverage
        morphed_stcs = [temp_stc.to_original_src(fs_avg_src)
                        for temp_stc in stcs]

        # Extract/store power; shape: (n_stcs x n_verts x n_freqs x n_times)
        stc_data = np.array([temp_stc.data for temp_stc in morphed_stcs])
        stc_pow = tfr_split(stc_data, common_params)
        avg_lab_pow[:, li, :, :] = stc_pow.mean(axis=1)

    ##############
    # Save Results
    ##############
    if save_data:
        print('Saving source power data:')

        # Check if save directory exists (and create it if necessary)
        save_dir = op.join(hcp_path, s_num, 'source_power')
        check_and_create_dir(save_dir)

        # Save results as pkl and npy files
        shuf_addition = '_shuffled' if shuffle_data else ''
        save_file = op.join(save_dir, 'src_power_%s%s' %
                            (exp_type, shuf_addition))
        print('\t' + save_file)

        with open(save_file + '.pkl', 'wb') as pkl_obj:
            results_to_save = deepcopy(common_params)
            results_to_save['power_data_shape'] = \
                'n_trials, n_labels, n_freqs, n_times'
            results_to_save['event_id'] = epo.event_id
            results_to_save['events'] = epo.events

            cPickle.dump(results_to_save, pkl_obj)

        with open(save_file + '.npy', 'wb') as arr_obj:
            np.save(arr_obj, avg_lab_pow)

        print('Data saved.')
