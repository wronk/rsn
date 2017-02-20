"""
classify_cnn.py

@author: wronk

Classify whether subject was resting or doing task using DL

TODO: 1) Feed raw BLP data. 2) Randomly sample overlapping windows to boost
      data. 3) 1D conv. in time. 4) Determine best params to optimize over
"""

import os
import os.path as op
import numpy as np
import cPickle
import csv

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

seed = 0
trial_len = 241

# Model params
cnn_layer_sizes = [10, 10]
fullC_layer_size = 32
pool_size = 2
filt_size = 2
dropout_keep_p = 0.5

# Training params
n_classes = 2
training_prop = 0.75
batch_size = 30
n_training_batches = 3000

subj_nums = config.subj_nums
subj_nums = ['17']

test_accuracies = []


def weight_variable(shape, name):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name))


def bias_variable(shape, name):
    #return(tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name))
    return(tf.Variable(tf.constant(0.1, shape=shape), name=name))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, pool_size, pool_size, 1], padding='SAME')


def create_model(cnn_layer_sizes, image_size):
    """Function to create CNN model

    cnn_layer_sizes: list of int
        Size of each layer in # of filters
    image_size:
        width and height of images

    Returns
    -------
    y_out:
        Output tf variable for model
    x_image:
        tf placeholder for images
    """

    x_train = tf.placeholder(tf.float32, shape=[None, image_size[0] * image_size[1]],
                             name='x_train')
    x_image = tf.reshape(x_train, [-1, image_size[0], image_size[1], 1])

    # First convolutional layer
    W_conv1 = weight_variable([filt_size, filt_size, 1, cnn_layer_sizes[0]], name='W1')
    b_conv1 = bias_variable([cnn_layer_sizes[0]], name='b1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([filt_size, filt_size, cnn_layer_sizes[0], cnn_layer_sizes[1]],
                              name='W2')
    b_conv2 = bias_variable([cnn_layer_sizes[1]], name='b2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    '''
    # Third convolutional layer
    W_conv3 = weight_variable([filt_size, filt_size, cnn_layer_sizes[1], cnn_layer_sizes[2]],
                              name='W3')
    b_conv3 = bias_variable([cnn_layer_sizes[2]], name='b3')
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3)

    # Fourth convolutional layer
    W_conv4 = weight_variable([filt_size, filt_size, cnn_layer_sizes[2], cnn_layer_sizes[3]],
                              name='W4')
    b_conv4 = bias_variable([cnn_layer_sizes[3]], name='b4')
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4)
    '''

    # First fully connected layer
    # Input: W_fc1 is image size x num_features
    n_pool_layers = float(len(cnn_layer_sizes))
    n_fc_vars = int(np.ceil(image_size[0] / pool_size ** n_pool_layers) *
                    np.ceil(image_size[1] / pool_size ** n_pool_layers) *
                    cnn_layer_sizes[-1])

    W_fc1 = weight_variable([n_fc_vars, fullC_layer_size], name='W_fc1')
    b_fc1 = bias_variable([fullC_layer_size], name='b_fc1')

    #h_pool2_flat = tf.reshape(h_pool4, [-1, n_fc_vars])
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_fc_vars])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Apply dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Second fully connected layer (includes output layer)
    W_fc2 = weight_variable([fullC_layer_size, n_classes], name='W_fc2')
    b_fc2 = bias_variable([n_classes], name='b_fc2')

    y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

    return x_train, y_out, keep_prob


#def get_training_batch(data, n_trials, train_prop, seed=None):
def get_training_batch(data, n_trials, seed=None):
    """Helper to get batch of training data from original data"""

    np.random.seed(seed)
    # Random sampling without replacement
    #rand_inds = np.random.choice(int(train_prop * data.shape[0]), size=(n_trials,),
    #                             replace=False)
    rand_inds = np.random.choice(data.shape[0], size=n_trials, replace=False)

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
    image_size = [n_pairs * n_freqs, trial_len]

    # Roll axis so frequencies will be grouped together after reshape
    trial_vecs = np.rollaxis(trial_images, 2, 1)
    # Reshape trials into 1D vectors and normalize pixels
    trial_vecs = trial_vecs.reshape(trial_vecs.shape[0], -1,
                                    trial_vecs.shape[3])
    #trial_vecs = trial_images.reshape(trial_images.shape[0], -1,
    #                                  trial_images.shape[3])
    trial_vecs = trial_vecs + np.abs(np.min(trial_vecs))
    trial_vecs = trial_vecs / np.max(trial_vecs)

    return trial_vecs, image_size


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

###########################################################
# Loop through each subject; load connectivity; predict
###########################################################

print 'Loading pickled data'
for s_num in subj_nums:
    num_exp = '0%s' % s_num
    s_name = 'AKCLEE_1' + s_num
    print '\nProcessing: ' + s_name

    ##########################
    # Load and preprocess data
    ##########################
    # Load dicts of data for each experiment type
    conn_task, conn_rest = load_rest_exp_data(num_exp)

    # TODO: Figure out better way to divide data into trials respecting trial timing
    # Naively split data into trials, set aside test data
    trials_rest_im, image_size = preprocess_data(conn_rest['conn_data'][0], 'wronk_resting')
    trials_task_im, image_size = preprocess_data(conn_task['conn_data'][0], 'eric_voc')

    ##################
    # Create test data
    ##################

    min_trials = np.min([np.ceil(training_prop * trials_rest_im.shape[0]),
                         np.ceil(training_prop * trials_task_im.shape[0])])

    np.random.seed(seed)
    test_inds_rest = np.random.choice(range(trials_rest_im.shape[0]),
                                      min_trials, replace=False)
    test_inds_task = np.random.choice(range(trials_task_im.shape[0]),
                                      min_trials, replace=False)

    test_rest = trials_rest_im[test_inds_rest, :, :]
    test_task = trials_task_im[test_inds_task, :, :]

    trials_rest_im = np.delete(trials_rest_im, test_inds_rest, axis=0)
    trials_task_im = np.delete(trials_task_im, test_inds_task, axis=0)

    """
    start_inds = [np.ceil(training_prop * temp.shape[0])
                  for temp in [trials_rest_im, trials_task_im]]
    test_rest = trials_rest_im[start_inds[0]:, :, :]
    test_task = trials_task_im[start_inds[1]:, :, :]

    # XXX Equalizing test trial count
    min_trials = np.min([test_rest.shape[0], test_task.shape[0]])
    test_rest = test_rest[:min_trials, :, :]
    test_task = test_task[:min_trials, :, :]
    """

    test_x = np.concatenate((test_rest, test_task))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = np.concatenate((np.zeros(test_rest.shape[0], dtype=int),
                             np.ones(test_task.shape[0], dtype=int)))

    if np.any(np.isnan(trials_task_im)) or np.any(np.isnan(trials_rest_im)):
        raise ValueError('NaNs in input')

    print 'Rest data, %i total trials, %i for testing.' % \
        (trials_rest_im.shape[0], test_rest.shape[0])
    print 'Task data, %i total trials, %i for testing.' % \
        (trials_task_im.shape[0], test_rest.shape[0])
    ##################
    # Create TF model
    ##################
    x_train, y_out, keep_prob = create_model(cnn_layer_sizes, image_size)
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    # Create loss function, training step, etc.
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_out, y_))
    train_step = tf.train.AdamOptimizer(0.5e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(y_out, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Attach summaries
    tf.scalar_summary('cross_entropy', cross_entropy)
    tf.scalar_summary('accuracy', accuracy)
    merged_summaries = tf.merge_all_summaries()

    #saver = tf.train.Saver()  # create saver for saving network weights
    init = tf.initialize_all_variables()
    sess = tf.Session()

    train_writer = tf.train.SummaryWriter('./train_summaries_%s' % num_exp,
                                          sess.graph)
    sess.run(init)

    #########################
    # Train and predict
    #########################
    for ind in range(n_training_batches):

        #batch_rest = get_training_batch(trials_rest_im, batch_size / 2, training_prop)
        #batch_task = get_training_batch(trials_task_im, batch_size / 2, training_prop)
        batch_rest = get_training_batch(trials_rest_im, batch_size / 2)
        batch_task = get_training_batch(trials_task_im, batch_size / 2)
        batch_x = np.concatenate((batch_rest, batch_task))

        batch_y = np.concatenate((np.zeros(batch_size / 2, dtype=int),
                                  np.ones(batch_size / 2, dtype=int)))

        assert len(np.unique(batch_y)) == n_classes, \
            "Number of classes must match n_classes"

        #XXX Does it matter that they're always the same position?
        one_hot_y = np.eye(n_classes)[batch_y]

        feed_dict = {x_train: batch_x, y_: batch_y, keep_prob: dropout_keep_p}

        # Save summaries for tensorboard every 10 steps
        if ind % 10 == 0:
            _, obj, acc, summary = sess.run([train_step, cross_entropy,
                                             accuracy, merged_summaries],
                                            feed_dict)
            train_writer.add_summary(summary, ind)
            print("\titer: %03d, cost: %.2f, acc: %.2f" % (ind, obj, acc))

        else:
            _, obj = sess.run([train_step, cross_entropy], feed_dict)

        if ind % 100 == 0 or ind == n_training_batches - 1:
            test_feed_dict = {x_train: test_x, y_: test_y, keep_prob: 1.}
            test_accuracies.append(accuracy.eval(session=sess,
                                                 feed_dict=test_feed_dict))
            print("Test accuracy: %.2f" % test_accuracies[-1])

    '''
    ### Save results
    # Check if save directory exists (and create it if necessary)
    save_dir = op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                       'connectivity_plots')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    '''

test_accuracies.sort(reverse=True)
avg_acc = np.mean(test_accuracies)

print 'Analysis complete.'
print 'Top 5 accuracies = %s' % str(test_accuracies[:5])
print 'Average = %s' % str(avg_acc)

with open('./train_summaries_%s/class_perf.csv' % num_exp, 'a') as class_perf_file:
    row = cnn_layer_sizes + [fullC_layer_size] + [batch_size] + \
        [n_training_batches] + subj_nums + test_accuracies[:3] + [avg_acc]

    writer = csv.writer(class_perf_file)
    writer.writerow(row)
