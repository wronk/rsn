"""
comp_sens_power.py

@author: wronk

Compute power of sensor activity

Handling eric_voc data: This code computes power on trials individually
and does not splice them together prior to computing connectivity. It also
crops trials to only the task portion of the epochs.
"""

import os
import os.path as op
import numpy as np
import cPickle
from copy import deepcopy

from mne import read_epochs

import config
from config import config_proc_methods, config_conn_params
from comp_fun import tfr_split

################
# Define globals
################
save_data = True  # Whether or not to pickle data
shuffle_data = False  # Whether or not to shuffle data in time (as a control)

struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])

# Choose to process resting data or task data
exp_heading = 'wronk_resting'
#exp_heading = 'eric_voc'

subj_nums = config.subj_nums

proc_methods = config_proc_methods
proc_params = config_conn_params
proc_params.update(dict(subj_nums=['%03i' % s_num for s_num in subj_nums]))

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

    # Load Epochs
    epo = read_epochs(op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                              'epochs', 'All_55-sss_%s_%s-epo.fif' %
                              (exp_heading, num_exp)))
    epo.pick_types(meg=True, eeg=False)

    # Subselect only vocoder trials (and throw away visual control trials)
    if exp_heading == 'eric_voc':
        epo.crop(None, 5.2)  # Crop to end at last stimulus letter

    # Randomly (and independently) shuffle time axis of each epoch data trial
    if shuffle_data:
        print '\tShuffling data'
        for t_i in range(epo._data.shape[0]):
            for c_i in range(epo._data.shape[1]):
                epo._data[t_i, c_i, :] = epo._data[t_i, c_i, np.random.permutation(epo._data.shape[2])]

    ###########################################################
    # Compute connectivity between RSN regions for each subject
    ###########################################################
    print '\tComputing power in pre-specified bands'
    power_data = []

    # Compute TFR of power bands
        # wavelet_ts is (n_stc, n_lab, n_freq, n_times)
    for proc_method in proc_params['proc_methods']:
        proc_method.update(sfreq=epo.info['sfreq'])
        wavelet_ts = tfr_split(epo._data, proc_method)

        power_data.append(wavelet_ts)  # Store raw power data

    ##############
    # Save Results
    ##############
    if save_data:
        print '\tSaving power data'

        # Check if save directory exists (and create it if necessary)
        save_dir = op.join(data_dir, '%s_sensor_power' % (exp_heading))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Save results as pkl file
        shuf_addition = 'shuffled_' if shuffle_data else ''
        save_file = op.join(save_dir, 'sens_pwr_results_%s%s_%s.pkl' % (
            shuf_addition, exp_heading, num_exp))
        with open(save_file, 'wb') as pkl_obj:
            results_to_save = deepcopy(proc_params)
            results_to_save['power_data'] = power_data
            results_to_save['power_data_shape'] = \
                'n_trials, n_label_pairs, n_freqs, n_times'
            results_to_save['sfreq'] = epo.info['sfreq']
            results_to_save['event_id'] = epo.event_id
            results_to_save['events'] = epo.events

            cPickle.dump(results_to_save, pkl_obj)
