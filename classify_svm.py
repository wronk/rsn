"""
classify_svm.py

@author: wronk

Classify whether subject was resting or doing task using support vector
machines and sklearn
"""

import os
import os.path as op
import cPickle
import csv
from itertools import product
import numpy as np
from numpy.random import choice
from time import strftime
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

import config
from config import config_conn_methods, config_conn_params
import cPickle

################
# Define globals
################
use_shuffled = False  # Whether or not to load randomly shuffled data
save_data = True  # Whether or not to pickle classification results

data_head = op.join(os.environ['RSN_DATA_DIR'])

trial_len = 241
n_features = 40 * 241

SVM_P = config.SVM_PARAMS

trial_start_samp = [120, 434]  # Rest, task trial start times in samples
#trial_start_samp = [120, 120]  # Rest, task trial start times in samples

subj_nums = config.subj_nums
#subj_nums = [17]

hyper_param_desc = '(Subjects), C_vals, gamma_vals, range(n_repeats), range(n_folds)'
hyper_params = [SVM_P['C_range'], SVM_P['g_range'], range(SVM_P['n_repeats'])]
hyper_param_shape = [len(temp_list) for temp_list in hyper_params]
hyper_param_inds = [range(si) for si in hyper_param_shape]

class_scores = np.zeros([len(subj_nums)] + hyper_param_shape +
                        [SVM_P['n_folds']])

print 'Started processing @ ' + strftime("%d/%m %H:%M:%S")
print 'Hyper params:\n' + str(hyper_params)
print 'Shuffled: ' + str(use_shuffled)


def load_data(num_exp):
    """Load and return both voc_meg and resting state data"""

    # Construct directory paths to two datasets
    data_dirs = [op.join(data_head, exp) for exp in ['rsn_data', 'voc_data']]
    exp_headings = ['wronk_resting', 'eric_voc']

    # Load both connectivity datasets
    data_list = []
    for data_dir, exp_heading in zip(data_dirs, exp_headings):

        # Change filename slightly if using shuffled data
        shuffle_add = 'shuffled_' if use_shuffled else ''

        load_dir = op.join(data_dir, '%s_connectivity' % exp_heading)
        load_file = op.join(load_dir, 'conn_results_%s%s_%s.pkl' % (
            shuffle_add, exp_heading, num_exp))

        with open(load_file, 'rb') as pkl_obj:
            data_list.append(cPickle.load(pkl_obj))

    assert len(data_list) is 2, "Both datasets not loaded"

    ######################
    # Create trials
    ######################
    data_rest, n_feat_rest = clean_data(data_list[0]['conn_data'][0], 'wronk_resting')
    data_task, n_feat_task = clean_data(data_list[1]['conn_data'][0], 'eric_voc')
    assert n_feat_rest == n_feat_task, "n_features doesn't match in rest/task"

    return data_rest, data_task


def clean_data(data, mode):
    """Helper to reshape and normalize data"""

    # XXX Looked good when double checking data reshaping
    if mode == 'wronk_resting':
        trial_images = cut_trials_rest(data)
    elif mode == 'eric_voc':
        trial_images = cut_trials_task(data)
    else:
        raise RuntimeError

    n_pairs = trial_images.shape[1]
    n_freqs = trial_images.shape[2]
    n_feat = n_pairs * n_freqs * trial_len

    # Roll axis so frequencies will be grouped together after reshape
    trial_vecs = np.rollaxis(trial_images, 2, 1)
    # Reshape trials into 1D vectors and normalize
    trial_vecs = trial_vecs.reshape(trial_vecs.shape[0], -1,
                                    trial_vecs.shape[3])
    trial_vecs = trial_vecs + np.abs(np.min(trial_vecs))
    trial_vecs = trial_vecs / np.max(trial_vecs)

    return trial_vecs, n_feat


def cut_trials_rest(data):
    """Cut data into segments of `trial_len`

    Returns
    -------
    trial_data: size (n_trials, n_conn_pairs, n_freqs, n_times)"""

    # TODO: compute cutoff time automatically

    # Exclude first 0.5 sec (120 time points) as cross corr function
    # (w/ 0.5 sec window) evaluates to nans here
    data_no_nans = data[:, :, :, trial_start_samp[0]:]

    n_trials = data_no_nans.shape[3] // trial_len

    # Reshape last two dimensions to be n_trials x n_time points
    trial_data = data_no_nans[:, :, :, :n_trials * trial_len].reshape(
        (data.shape[1], data.shape[2], n_trials, trial_len))

    # Roll so that first index is now trial number
    trial_data = np.rollaxis(trial_data, 2, 0)

    return trial_data


def cut_trials_task(data):
    """Cut existing exp. trials into classification segments of `trial_len`

    Returns
    -------
    trial_data: size (n_trials, n_conn_pairs, n_freqs, n_times)"""

    # Exclude first 1.8 sec (-0.2-1.6; 434 time points) as actual task where
    # subject must start listening listen for 6 target letters starts at 1.6
    # seconds
    data_no_nans = data[:, :, :, trial_start_samp[1]:]

    # Compute number of segments for classification (w/ len trial_len) from
    # experimental trials.
    # Ex. Three 1 sec segments from 3.6 seconds trials of experimental data
    segs_per_trial = data_no_nans.shape[3] // trial_len
    n_trials = segs_per_trial * data.shape[0]
    trial_data = np.zeros((n_trials, data.shape[1], data.shape[2],
                           trial_len))

    for t_ind, trial in enumerate(data):
        for seg in range(segs_per_trial):
            t1 = seg * trial_len
            t2 = (seg + 1) * trial_len

            trial_data[t_ind * segs_per_trial + seg, :, :, :] = \
                data_no_nans[t_ind, :, :, t1:t2]

    return trial_data


def get_equalized_data(trials_rest, trials_task):
    """Equalize rest/task data and return labels"""

    # Equalize test trial count
    min_trials = min([trials_rest.shape[0], trials_task.shape[0]])

    inds_rest = choice(range(trials_rest.shape[0]), min_trials, replace=False)
    inds_task = choice(range(trials_task.shape[0]), min_trials, replace=False)

    trials_eq = np.concatenate((
        trials_rest[inds_rest, :, :].reshape(min_trials, -1),
        trials_task[inds_task, :, :].reshape(min_trials, -1)))

    # Create matching labels
    y_labels = np.concatenate((np.zeros(min_trials), np.ones(min_trials)))

    return trials_eq, y_labels


###########################################################
# Loop through each subject; load connectivity; predict
###########################################################

print '\nLoading pickled data'
for si, s_num in enumerate(subj_nums):
    s_num_exp = '0%02i' % s_num
    s_name = 'AKCLEE_1%02i' % s_num
    print '\nProcessing: %s\n' % s_name

    ##########################
    # Load and preprocess data
    ##########################
    all_data_rest, all_data_task = load_data(s_num_exp)

    for hpi, hp_set in zip(list(product(*hyper_param_inds)),
                           list(product(*hyper_params))):

        print 'Subj: %i/%i, Hyper param set: %s' % (si, len(subj_nums) - 1,
                                                    str(hpi))

        # Randomize and equalize subject data
        trial_data, y_labels = get_equalized_data(all_data_rest, all_data_task)

        # Check for any nans in the new data slices
        if np.any(np.isnan(trial_data)) or np.any(np.isnan(y_labels)):
            raise ValueError('NaNs in training and/or test data')

        #########################
        # Train and predict
        #########################
        cv_obj = StratifiedKFold(y_labels, n_folds=SVM_P['n_folds'],
                                 shuffle=True)

        for ti, (train_idx, test_idx) in enumerate(cv_obj):

            # Initialize classifier with params
            clf = SVC(kernel=SVM_P['kernel'], C=hp_set[0], gamma=hp_set[1],
                      cache_size=SVM_P['cache_size'])

            # Train and test classifier, save score
            clf.fit(trial_data[train_idx], y_labels[train_idx])
            temp_score = clf.score(trial_data[test_idx], y_labels[test_idx])
            class_scores[si, hpi[0], hpi[1], hpi[2], ti] = temp_score

            print 'Test accuracy on CV %i: %0.2f' % (ti, temp_score)

print 'Finished processing @ ' + strftime("%m/%d %H:%M:%S")

##############
# Save results
##############

if save_data:
    date_str = strftime('%Y_%m_%d__%H_%M')
    shuffled_add = '_shuffled' if use_shuffled else ''

    save_dict = dict(class_scores=class_scores,
                     hyper_params=hyper_params,
                     hyper_param_desc=hyper_param_desc,
                     subj_nums=subj_nums,
                     trial_len=trial_len,
                     n_features=n_features,
                     time_finished=date_str)

    fname_scores_pkl = op.join(data_head, 'rsn_results', 'svm_scores_' +
                               date_str + shuffled_add + '.pkl')
    with open(fname_scores_pkl, 'wb') as pkl_file:
        cPickle.dump(save_dict, pkl_file)
