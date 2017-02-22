"""
comp_sens_power.py

@author: wronk

Compute power of sensor activity in HCP data
"""

import os
import os.path as op
from copy import deepcopy
from itertools import product
import cPickle
import numpy as np

import config as cf
from config import hcp_path, common_params
from comp_fun import tfr_split, get_concat_epos, check_and_create_dir

################
# Define globals
################
save_data = True  # Whether or not to pickle data

shuffles = [True, False]  # Whether or not to shuffle data in time (as a control)
exp_types = ['motor', 'rest']

struct_dir = os.environ['SUBJECTS_DIR']
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

###########################################################
# Loop through each subject; load info, comp power
###########################################################
print('Subjects: ' + str(dirs) + '\n')

for (s_num, exp_type, shuffle_data) in product(dirs, exp_types, shuffles):
    print('\n%s\nSubj: %s, Exp_type: %s, Shuffle:%s\n' %
          ('=' * 40, s_num, exp_type, shuffle_data))

    # Load Epochs
    epo = get_concat_epos(s_num, exp_type)
    epo.pick_types(meg=True, eeg=False)

    # Randomly (and independently) shuffle time axis of each epoch data trial
    if shuffle_data:
        print('\tShuffling data')
        for t_i in range(epo._data.shape[0]):
            for c_i in range(epo._data.shape[1]):
                epo._data[t_i, c_i, :] = epo._data[t_i, c_i, np.random.permutation(epo._data.shape[2])]

    ###########################################################
    # Compute power at each sensor
    ###########################################################
    print('Computing power in pre-specified bands')

    # Compute TFR of power bands
    #     wavelet_ts is (n_trials, n_chan, n_freq, n_times)

    common_params.update(sfreq=epo.info['sfreq'])
    wavelet_ts = tfr_split(epo._data, common_params)
    print('Wavelet shape: %s' % str(wavelet_ts.shape))

    ##############
    # Save Results
    ##############
    if save_data:
        print('Saving power data:')

        # Check if save directory exists (and create it if necessary)
        save_dir = op.join(hcp_path, s_num, 'sensor_power')
        check_and_create_dir(save_dir)

        # Save results as pkl and npy files
        shuf_addition = '_shuffled' if shuffle_data else ''
        save_file = op.join(save_dir, 'sens_power_%s%s' % (
            exp_type, shuf_addition))
        print('\t' + save_file)

        with open(save_file + '.pkl', 'wb') as pkl_obj:
            results_to_save = deepcopy(common_params)
            results_to_save['power_data_shape'] = \
                'n_trials, n_chan, n_freqs, n_times'
            results_to_save['sfreq'] = epo.info['sfreq']
            results_to_save['event_id'] = epo.event_id
            results_to_save['events'] = epo.events

            cPickle.dump(results_to_save, pkl_obj)

        with open(save_file + '.npy', 'wb') as arr_obj:
            np.save(arr_obj, wavelet_ts)

        print('Data saved.')
