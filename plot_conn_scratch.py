"""
plot_conn_scratch.py

@author: wronk

Plot basic connectivity for exploratory analysis
"""

import os
import os.path as op
import numpy as np
import cPickle

import matplotlib as mpl
from matplotlib import pyplot as plt

import config

################
# Define globals
################
struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])

use_shuffled = True
save_data = True  # Whether or not to pickle data

# Choose to process resting data or task data
#exp_heading = 'wronk_resting'
exp_heading = 'eric_voc'


if exp_heading == 'wronk_resting':
    data_dir = op.join(data_head, 'rsn_data')
else:
    data_dir = op.join(data_head, 'voc_data')

subj_nums = config.subj_nums
#subj_nums = ['15', '17']

###########################################################
# Loop through each subject; load connectivity
###########################################################
#plt.ion()

conn_list = []

print 'Experiment: ' + exp_heading
print 'Use shuffled data: %s\n' % str(use_shuffled)
print 'Loading pickled data'

for s_num in subj_nums:
    num_exp = '0%s' % s_num
    s_name = 'AKCLEE_1%s' % s_num
    print '\nProcessing: ' + s_name

    shuffle_add = 'shuffled_' if use_shuffled else ''
    load_dir = op.join(data_dir, '%s_connectivity' % exp_heading)
    load_file = op.join(load_dir, 'conn_results_%s%s_%s.pkl' % (
        shuffle_add, exp_heading, num_exp))

    with open(load_file, 'rb') as pkl_obj:
        conn_results = cPickle.load(pkl_obj)

    # Store all connectivity data in a list
    conn_list.append(conn_results)

    n_pairs = conn_results['conn_data'][0].shape[1]
    n_freqs = conn_results['conn_data'][0].shape[2]

    ########################
    # Plot
    ########################

    fig0, axes0 = plt.subplots(nrows=n_pairs, ncols=n_freqs, figsize=(24, 8),
                               sharex=True, sharey=True)

    # Check if save directory exists (and create it if necessary)
    save_dir = op.join(data_dir, '%s_%s' % (exp_heading, num_exp),
                       'connectivity_plots')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    t1 = 0
    t2 = t1 + 1297
    for ai0, ai1 in np.ndindex(n_pairs, n_freqs):
        ax = axes0[ai0, ai1]
        times = np.arange(t1, t2) / conn_results['sfreq']
        ax.plot(times, conn_results['conn_data'][0][0, ai0, ai1, t1:t2])
        #ax.set_xlim([0, len(conn_results['conn_data'][0][ai0, ai1, :])])
        ax.hlines(0, times[0], times[-1], colors='0.5', linestyles='dashed')
        ax.locator_params(axis='x', nbins=8)
        ax.locator_params(axis='y', nbins=3)

        if ai0 is 0:
            ax.set_title('Wavelet: %i Hz' %
                         conn_results['conn_methods'][0]['cwt_frequencies'][ai1])
        if ai0 is n_pairs - 1:
            ax.set_xlabel('Time (s)')
        if ai1 is 0:
            ax.set_ylabel('%i <-> %i\nBLP corr.' %
                          (conn_results['conn_pairs'][0][ai0],
                           conn_results['conn_pairs'][1][ai0]))

    #fig0.tight_layout()
    fig0.subplots_adjust(wspace=0.05, hspace=0.15)

    shuffle_add = '_shuffled' if use_shuffled else ''
    filename_power = op.join(save_dir, 'power_correlations%s.pdf' % shuffle_add)
    fig0.savefig(filename_power)

    ##########################
    # Compute basic statistics
    ##########################
    power_avg = conn_results['conn_data'][0][:, :, :, 120:].mean(axis=(0, 3))
    power_var = conn_results['conn_data'][0][:, :, :, 120:].var(axis=(0, 3))
    np.savetxt(op.join(save_dir, 'power_avg%s.csv' % shuffle_add), power_avg,
               fmt='%.05f', delimiter=',')
    np.savetxt(op.join(save_dir, 'power_var%s.csv' % shuffle_add), power_var,
               fmt='%.05f', delimiter=',')

#plt.show()
print 'Analysis complete.'
