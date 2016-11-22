"""
classify_lr.py

@author: wronk

Classify whether subject was resting or doing task using logistic regression
"""

import os
import os.path as op
import cPickle
import csv
from itertools import product
import numpy as np
from numpy.random import choice
from time import strftime

import tensorflow as tf

import config
from config import config_conn_methods, config_conn_params
import cPickle

tf.logging.set_verbosity(tf.logging.INFO)
################
# Define globals
################
use_shuffled = False  # Whether or not to load randomly shuffled data
save_data = True  # Whether or not to pickle classification results

data_head = op.join(os.environ['RSN_DATA_DIR'])

seed = None
trial_len = 241
n_features = 40 * 241

l1_weights = [10 ** x for x in range(-7, 0)]
l2_weights = [10 ** x for x in range(-5, 2)]

# Training params
n_classes = 2
testing_prop = 0.2  # Proportion of data saved for testing
batch_sizes = [50]
n_training_batches = 1500
n_repeats = 20
test_freq = 100  # Evaluate test data every n batches

trial_start_samp = [120, 434]  # Rest, task trial start times in samples
#trial_start_samp = [120, 120]  # Rest, task trial start times in samples

subj_nums = config.subj_nums
#subj_nums = [17]

hyper_param_desc = '(Subjects), batch_sizes, l1_weights, l2_weights, range(n_repeats)'
hyper_params = [batch_sizes, l1_weights, l2_weights, range(n_repeats)]
hyper_param_shape = [len(temp_list) for temp_list in hyper_params]
hyper_param_inds = [range(si) for si in hyper_param_shape]

class_scores = np.zeros([len(subj_nums)] + hyper_param_shape)

print 'Started processing @ ' + strftime("%d/%m %H:%M:%S")
print 'Hyper params:\n' + str(hyper_params)
print 'Shuffled: ' + str(use_shuffled)


def weight_variable(shape, name):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name))


def bias_variable(shape, name):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name))


def l1_norm(tensor):
    return tf.reduce_sum(tf.abs(tensor))


def l2_norm(tensor):
    #return tf.sqrt(tf.reduce_sum(tf.square(tensor)))  # True L2 norm
    return tf.nn.l2_loss(tensor)  # TF L2 is sum of elements squared divided by 2


def create_model(n_features):
    """Function to create regression model (logistic part added via loss func)

    n_features:
        number of feature variables per sample

    Returns
    -------
    x_data:
        training data
    y_out:
        Output tf variable for model
    """

    x_data = tf.placeholder(tf.float32, shape=[None, n_features], name='x_data')
    W = weight_variable([n_features, n_classes], name='W')
    b = bias_variable([n_classes], name='b')

    y_pred = tf.matmul(x_data, W) + b

    return x_data, y_pred, W, b


def get_training_batch(data, n_trials, seed=None):
    """Helper to get batch of training data from original data"""

    np.random.seed(seed)
    # Random sampling without replacement
    #rand_inds = choice(int(train_prop * data.shape[0]), size=(n_trials,),
    #                             replace=False)
    rand_inds = choice(data.shape[0], size=n_trials, replace=False)

    return data[rand_inds, :, :].reshape((n_trials, -1))


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


def get_training_test_data(trials_rest_im, trials_task_im):
    """Load data, split into training and test sets"""
    ##################
    # Create test data
    ##################
    # Equalize test trial count
    min_trials = int(np.min([np.ceil(testing_prop * trials_rest_im.shape[0]),
                             np.ceil(testing_prop * trials_task_im.shape[0])]))

    # Create training and testing trial indices
    np.random.seed(seed)
    test_inds_rest = choice(range(trials_rest_im.shape[0]), min_trials,
                            replace=False)
    test_inds_task = choice(range(trials_task_im.shape[0]), min_trials,
                            replace=False)

    train_inds_rest = [x for x in range(trials_rest_im.shape[0]) if x not in test_inds_rest]
    train_inds_task = [x for x in range(trials_task_im.shape[0]) if x not in test_inds_task]

    # Seperate training and test trials
    train_rest = trials_rest_im[train_inds_rest]
    train_task = trials_task_im[train_inds_task]

    test_rest = trials_rest_im[test_inds_rest]
    test_task = trials_task_im[test_inds_task]

    test_x = np.concatenate((test_rest, test_task))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = np.concatenate((np.zeros(test_rest.shape[0], dtype=int),
                             np.ones(test_task.shape[0], dtype=int)))

    return train_rest, train_task, test_x, test_y


###########################################################
# Loop through each subject; load connectivity; predict
###########################################################

x_data, y_pred, weights, biases = create_model(n_features)
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')
l1l2_weights = tf.placeholder(tf.float32, shape=[2], name='l1l2_weights')

# Create loss function, training step, etc.
pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, y_))
l1_loss = (l1_norm(weights) + l1_norm(biases))
l2_loss = (l2_norm(weights) + l2_norm(biases))
l1l2_loss = l1l2_weights[0] * l1_loss + l1l2_weights[1] * l2_loss

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(pred_loss + l1l2_loss)
correct_prediction = tf.equal(tf.arg_max(y_pred, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Attach summaries
#tf.scalar_summary('prediction loss', pred_loss)
#tf.scalar_summary('l1l2_loss', l1l2_loss)
#tf.scalar_summary('accuracy', accuracy)
#merged_summaries = tf.merge_all_summaries()

init_op = tf.initialize_all_variables()


print '\nLoading pickled data'
for si, s_num in enumerate(subj_nums):
    s_num_exp = '0%02i' % s_num
    s_name = 'AKCLEE_1%02i' % s_num
    print '\nProcessing: %s\n' % s_name

    ##########################
    # Load and preprocess data
    ##########################
    all_data_rest, all_data_task = load_data(s_num_exp)

    ##################
    # Create TF model
    ##################

    #saver = tf.train.Saver()  # create saver for saving network weights
    #train_writer = tf.train.SummaryWriter('./train_summaries_%s' % s_num_exp,
    #                                      sess.graph)

    for hpi, hp_set in zip(list(product(*hyper_param_inds)),
                           list(product(*hyper_params))):
        sess = tf.Session()
        sess.run(init_op)

        print 'Subj: %i/%i, Hyper param set: %s' % (si, len(subj_nums) - 1,
                                                    str(hpi))
        batch_size = hp_set[0]
        test_accuracies = []

        # Get new data fold
        train_x_rest, train_x_task, test_x, test_y = \
            get_training_test_data(all_data_rest, all_data_task)

        # Check for any nans in the new data slices
        if np.any(np.isnan(train_x_rest)) or np.any(np.isnan(train_x_task)) \
           or np.any(np.isnan(test_x)) or np.any(np.isnan(test_y)):
            raise ValueError('NaNs in training and/or test data')

        #########################
        # Train and predict
        #########################
        for ind in range(n_training_batches):
            batch_rest = get_training_batch(train_x_rest, batch_size / 2)
            batch_task = get_training_batch(train_x_task, batch_size / 2)
            batch_x = np.concatenate((batch_rest, batch_task))

            batch_y = np.concatenate((np.zeros(batch_size / 2, dtype=int),
                                      np.ones(batch_size / 2, dtype=int)))

            feed_dict = {x_data: batch_x, y_: batch_y,
                         l1l2_weights: [hp_set[1], hp_set[2]]}

            # Save summaries for tensorboard every batch
            _, acc = sess.run([train_op, accuracy], feed_dict)

            if (ind % test_freq == 0 and ind > 500) or ind == n_training_batches - 1:
                test_feed_dict = {x_data: test_x, y_: test_y}
                test_accuracies.append(accuracy.eval(session=sess,
                                                     feed_dict=test_feed_dict))
                print 'Test accuracy on loop %04i: %0.2f' % (ind, test_accuracies[-1])

        test_accuracies.sort(reverse=True)
        class_scores[si, hpi[0], hpi[1], hpi[2], hpi[3]] = \
            np.mean(test_accuracies)

        #print '  Top 5 accuracies = %s' % str(test_accuracies[:5])

        sess.close()

print 'Finished processing @ ' + strftime("%m/%d %H:%M:%S")

# Save results
if save_data:
    date_str = strftime('%Y_%m_%d__%H_%M')
    shuffled_add = '_shuffled' if use_shuffled else ''

    save_dict = dict(class_scores=class_scores,
                     hyper_params=hyper_params,
                     hyper_param_desc=hyper_param_desc,
                     subj_nums=subj_nums,
                     testing_proportion=testing_prop,
                     n_training_batchs=n_training_batches,
                     trial_len=trial_len,
                     n_features=n_features,
                     time_finished=date_str)

    fname_scores_pkl = op.join(data_head, 'rsn_results', 'lr_scores_' +
                               date_str + shuffled_add + '.pkl')
    with open(fname_scores_pkl, 'wb') as pkl_file:
        cPickle.dump(save_dict, pkl_file)
