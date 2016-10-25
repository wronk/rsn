"""
classify_lr.py

@author: wronk

Classify whether subject was resting or doing task using logistic regression
"""

import os
import os.path as op
import cPickle
import csv
import numpy as np
from numpy.random import choice

import tensorflow as tf

import matplotlib as mpl
from matplotlib import pyplot as plt

import config
from config import config_conn_methods, config_conn_params

################
# Define globals
################
save_data = True  # Whether or not to pickle data

struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])

seed = None
trial_len = 241
n_features = 40 * 241

l1_w, l2_w = 0.001, 0.02
#l1_w, l2_w = 0.005, 0.01

# Training params
n_classes = 2
training_prop = 0.75
batch_size = 100
n_training_batches = 1500

subj_nums = config.subj_nums
subj_nums = ['15', '17']

test_accuracies = []


def weight_variable(shape, name):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name))


def bias_variable(shape, name):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name))


def l1_norm(tensor):
    return tf.reduce_sum(tf.abs(tensor))


def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))


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


def load_rest_exp_data(num_exp):
    """Load and return both voc_meg and resting state data"""

    # Construct directory paths to two datasets
    data_dirs = [op.join(data_head, exp) for exp in ['voc_data', 'rsn_data']]
    exp_headings = ['eric_voc', 'wronk_resting']

    # Load both connectivity datasets
    data_list = []
    for data_dir, exp_heading in zip(data_dirs, exp_headings):

        load_dir = op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                           'connectivity')
        load_file = op.join(load_dir, 'conn_results_%s.pkl' % exp_heading)

        with open(load_file, 'rb') as pkl_obj:
            data_list.append(cPickle.load(pkl_obj))

    assert len(data_list) is 2, "Both datasets not loaded"
    return (data_list[0], data_list[1])


def preprocess_data(data, mode):
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
    # Reshape trials into 1D vectors and normalize pixels
    trial_vecs = trial_vecs.reshape(trial_vecs.shape[0], -1,
                                    trial_vecs.shape[3])
    #trial_vecs = trial_images.reshape(trial_images.shape[0], -1,
    #                                  trial_images.shape[3])
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
    data_no_nans = data[:, :, :, 120:]

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
    #data_no_nans = data[:, :, :, 120:]
    data_no_nans = data[:, :, :, 434:]

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


def get_training_test_data(subj_num):
    """Load data, split into training and test sets"""
    conn_task, conn_rest = load_rest_exp_data(subj_num)

    ######################
    # Create training data
    ######################
    # TODO: Figure out better way to divide data into trials respecting trial timing
    # Naively split data into trials, set aside test data
    trials_rest_im, n_feat_rest = preprocess_data(conn_rest['conn_data'][0], 'wronk_resting')
    trials_task_im, n_feat_task = preprocess_data(conn_task['conn_data'][0], 'eric_voc')
    assert n_feat_rest == n_feat_task, "n_features doesn't match in rest/task"

    ##################
    # Create test data
    ##################
    # Equalize test trial count
    min_trials = np.min([np.ceil(training_prop * trials_rest_im.shape[0]),
                         np.ceil(training_prop * trials_task_im.shape[0])])

    np.random.seed(seed)
    test_inds_rest = choice(range(trials_rest_im.shape[0]), min_trials,
                            replace=False)
    test_inds_task = choice(range(trials_task_im.shape[0]), min_trials,
                            replace=False)

    test_rest = trials_rest_im[test_inds_rest, :, :]
    test_task = trials_task_im[test_inds_task, :, :]

    # Delete rows used as test data
    trials_rest_im = np.delete(trials_rest_im, test_inds_rest, axis=0)
    trials_task_im = np.delete(trials_task_im, test_inds_task, axis=0)

    test_x = np.concatenate((test_rest, test_task))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = np.concatenate((np.zeros(test_rest.shape[0], dtype=int),
                             np.ones(test_task.shape[0], dtype=int)))

    return trials_rest_im, trials_task_im, test_x, test_y


###########################################################
# Loop through each subject; load connectivity; predict
###########################################################
sess = tf.Session()

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
tf.scalar_summary('prediction loss', pred_loss)
tf.scalar_summary('l1l2_loss', l1l2_loss)
tf.scalar_summary('accuracy', accuracy)
merged_summaries = tf.merge_all_summaries()

print '\nLoading pickled data'
for s_num in subj_nums:
    s_num_exp = '0%s' % s_num
    s_name = 'AKCLEE_1' + s_num
    print '\nProcessing: ' + s_name

    ##########################
    # Load and preprocess data
    ##########################
    train_x_rest, train_x_task, test_x, test_y = \
        get_training_test_data(s_num_exp)
    # Load dicts of data for each experiment type
    if np.any(np.isnan(train_x_rest)) or np.any(np.isnan(train_x_task)):
        raise ValueError('NaNs in input')

    print 'Rest data, %i total trials.' % (train_x_rest.shape[0])
    print 'Task data, %i total trials.' % (train_x_task.shape[0])

    ##################
    # Create TF model
    ##################

    #saver = tf.train.Saver()  # create saver for saving network weights
    init = tf.initialize_all_variables()
    train_writer = tf.train.SummaryWriter('./train_summaries_%s' % s_num_exp,
                                          sess.graph)
    sess.run(init)

    #########################
    # Train and predict
    #########################
    for ind in range(n_training_batches):

        batch_rest = get_training_batch(train_x_rest, batch_size / 2)
        batch_task = get_training_batch(train_x_task, batch_size / 2)
        batch_x = np.concatenate((batch_rest, batch_task))

        batch_y = np.concatenate((np.zeros(batch_size / 2, dtype=int),
                                  np.ones(batch_size / 2, dtype=int)))

        assert len(np.unique(batch_y)) == n_classes, \
            "Number of classes must match n_classes"

        # TODO: add loss weight parameters as placeholders
        feed_dict = {x_data: batch_x, y_: batch_y, l1l2_weights: [l1_w, l2_w]}

        # Save summaries for tensorboard every 10 steps
        if ind % 10 == 0:
            _, obj1, obj2, acc, summary = sess.run([train_op, pred_loss,
                                                    l1l2_loss, accuracy,
                                                    merged_summaries],
                                                   feed_dict)
            train_writer.add_summary(summary, ind)
            print("\titer: %03d, pred_loss: %.2f, l1l2_loss: %.2f, acc: %.2f" %
                  (ind, obj1, obj2, acc))

        else:
            _, obj1, obj2 = sess.run([train_op, pred_loss, l1l2_loss],
                                     feed_dict)

        if ind % 100 == 0 or ind == n_training_batches - 1:
            test_feed_dict = {x_data: test_x, y_: test_y}
            test_accuracies.append(accuracy.eval(session=sess,
                                                 feed_dict=test_feed_dict))
            print("Test accuracy: %.2f" % test_accuracies[-1])

    '''
    ### Save results
    # Check if save directory exists (and create it if necessary)
    save_dir = op.join(data_dir, '%s_%s' % (exp_heading, s_num_exp),
                       'connectivity_plots')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    '''

    test_accuracies.sort(reverse=True)
    avg_acc = np.mean(test_accuracies)

    print 'Analysis complete.'
    print 'Top 5 accuracies = %s' % str(test_accuracies[:5])
    print 'Average = %s' % str(avg_acc)

    with open('./train_summaries_%s/class_perf_lr.csv' % s_num_exp,
              'a') as class_perf_file:
        row = [batch_size] + [n_training_batches] + subj_nums + \
            test_accuracies[:3] + [avg_acc]
        writer = csv.writer(class_perf_file)
        writer.writerow(row)

sess.close()
