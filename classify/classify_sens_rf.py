"""
classify_sens_rf.py

@author: wronk

Classify whether subject was resting or doing task using sensor space data and
random forests
"""

import os
import os.path as op
import cPickle
import csv
from itertools import product
from time import strftime

import cPickle
import numpy as np
from numpy.random import choice
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import StratifiedKFold

from rsn import config as cf
from rsn.config import hcp_path
from rsn.comp_fun import check_and_create_dir

################
# Define globals
################
use_shuffled = False  # Whether or not to load randomly shuffled data
save_data = True  # Whether or not to pickle classification results

CLS_P = cf.RF_PARAMS
shuffled_add = '_shuffled' if use_shuffled else ''

# Define trial types wanted
#event_ids = dict(rest=['rest'],
#                 motor=['RH', 'LH', 'RF', 'LF'])

subj_nums = [str(temp_subj) for temp_subj in cf.subj_nums_hcp[0:1]]

hyper_param_desc = '(Subjects), n_est, max_feat, range(n_repeats), range(n_folds)'
hyper_params = [CLS_P['n_est_range'], CLS_P['max_feat_range'], range(CLS_P['n_repeats'])]
hyper_param_shape = [len(temp_list) for temp_list in hyper_params]
hyper_param_inds = [range(si) for si in hyper_param_shape]


print 'Started processing @ ' + strftime("%d/%m %H:%M:%S")
print 'Hyper params:\n' + str(hyper_params)
print 'Shuffled: ' + str(use_shuffled)


def load_data(s_num):
    """Load and return both voc_meg and resting state data"""

    # Construct directory paths to two datasets
    data_dir = op.join(hcp_path, s_num, 'sensor_power')
    exp_headings = ['rest', 'motor']

    # Load both connectivity datasets
    data_list = []
    for exp_heading in exp_headings:

        load_file = op.join(data_dir, 'sens_power_%s%s' % (
            exp_heading, shuffled_add))
        print('Loading pkl/npy data from: %s' % load_file)

        with open(load_file + '.pkl', 'rb') as pkl_obj:
            meta_data = cPickle.load(pkl_obj)
        with open(load_file + '.npy', 'rb') as np_obj:
            # Load file, subselect trials desired (from `event_ids`)
            temp_data = np.load(np_obj)
            '''
            if exp_heading == 'eric_voc':
                event_nums = [temp_obj['event_id'][e_id]
                              for e_id in event_ids[exp_heading]]
                desired_trials = [row_i for row_i in range(len(temp_obj['events']))
                                  if temp_obj['events'][row_i, -1] in event_nums]

                temp_obj['conn_data'][0] = temp_obj['conn_data'][0][desired_trials]
                temp_obj['choosen_trial_ids'] = [temp_obj['events'][row_i, -1]
                                                 for row_i in desired_trials]
            '''

            # Swap channel/frequency dimensions and reshape
            #import ipdb; ipdb.set_trace()

            reshaped_data = np.mean(temp_data, axis=(-1)).reshape(temp_data.shape[0], -1)

            #rolled_data = np.rollaxis(temp_data, 2, 1)
            #reshaped_data = rolled_data.reshape(rolled_data.shape[0], -1)

            data_list.append(reshaped_data)

    assert len(data_list) is 2, "Both datasets not loaded"
    data_rest, data_task = data_list
    assert data_rest.shape[-1] == data_task.shape[-1], "n_features doesn't match in rest/task"

    return data_rest, data_task


def get_equalized_data(trials_rest, trials_task):
    """Equalize rest/task data and return labels"""

    # Equalize test trial count
    min_trials = min([trials_rest.shape[0], trials_task.shape[0]])
    min_trials = 200

    inds_rest = choice(range(trials_rest.shape[0]), min_trials, replace=False)
    inds_task = choice(range(trials_task.shape[0]), min_trials, replace=False)

    trials_eq = np.concatenate((trials_rest[inds_rest, :],
                                trials_task[inds_task, :]))

    # Create matching labels
    y_labels = np.concatenate((np.zeros(min_trials), np.ones(min_trials)))

    return trials_eq, y_labels


###########################################################
# Loop through each subject; load sensor data; predict
###########################################################
ss = StandardScaler()
for si, s_num in enumerate(subj_nums):
    print('\nProcessing: %s\n%s\n' % (s_num, 40 * '='))

    class_scores = np.zeros(hyper_param_shape + [CLS_P['n_folds']])
    ##########################
    # Load and preprocess data
    ##########################
    all_data_rest, all_data_task = load_data(s_num)
    print('\nData loaded.')

    for hpi, hp_set in zip(list(product(*hyper_param_inds)),
                           list(product(*hyper_params))):
        print('Subj: %i of %i, Hyper param set: %s' %
              (si + 1, len(subj_nums), str(hpi)))

        # Equalize subject data, and standardize to unit variance/zero mean
        trial_data, y_labels = get_equalized_data(all_data_rest, all_data_task)

        # Check for any nans in the new data slices
        if np.any(np.isnan(trial_data)) or np.any(np.isnan(y_labels)):
            raise ValueError('NaNs in training and/or test data')

        #########################
        # Train and predict
        #########################
        cv_obj = StratifiedKFold(y_labels, n_folds=CLS_P['n_folds'],
                                 shuffle=True)

        for ti, (train_idx, test_idx) in enumerate(cv_obj):

            X_train = ss.fit_transform(trial_data[train_idx])
            X_test = ss.transform(trial_data[test_idx])

            # Initialize classifier with params
            clf = RFC(n_estimators=hp_set[0], max_depth=hp_set[1],
                      n_jobs=CLS_P['n_jobs'])

            # Train and test classifier, save score
            clf.fit(X_train, y_labels[train_idx])
            temp_score = clf.score(X_test, y_labels[test_idx])
            class_scores[hpi[0], hpi[1], hpi[2], ti] = temp_score

            print 'Test accuracy on CV %i: %0.2f' % (ti, temp_score)

    # Save results
    if save_data:
        date_str = strftime('%Y_%m_%d__%H_%M')
        save_dict = dict(class_scores=class_scores, hyper_params=hyper_params,
                         hyper_param_desc=hyper_param_desc,
                         subj=s_num, time_finished=date_str)

        save_dir = op.join(hcp_path, 'classification_results', 'sens_rf%s' %
                           shuffled_add)
        fname_scores_pkl = op.join(save_dir, s_num + '.pkl')
        check_and_create_dir(save_dir)

        with open(fname_scores_pkl, 'wb') as pkl_file:
            cPickle.dump(save_dict, pkl_file)

print 'Finished processing @ ' + strftime("%m/%d %H:%M:%S")
print '\nFinal Scores:\n' + str(class_scores.mean((-1, -2)))
