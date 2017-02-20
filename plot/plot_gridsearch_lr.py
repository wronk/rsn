"""
plot_gridsearch_lr.py

author: @wronk

Plots saved logisitic regression gridsearch data
"""

import os
import os.path as op
import numpy as np
from itertools import product

import matplotlib as mpl
from matplotlib import pyplot as plt

import cPickle
from time import strftime

################
# Define globals
################
save_plot = True  # Whether or not to pickle data
shuffled = False

struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])
saved_data_dir = op.join(data_head, 'rsn_results')

shuffled_add = '_shuffled' if shuffled else ''
file_name = 'lr_scores_2016_11_13__13_07%s.pkl' % shuffled_add
#file_name = 'lr_scores_2016_11_12__08_18%s.pkl' % shuffled_add

###################
# Load pickled data
###################
fpath_data = op.join(saved_data_dir, file_name)
with open(fpath_data) as pkl_file:
    result_dict = cPickle.load(pkl_file)

#print 'Loaded data (%s) from %s' % (fpath_data, result_dict['time_finished'])
print 'Hyper param format: %s' % result_dict['hyper_param_desc']
print 'Hyper param size: %s' % str(result_dict['class_scores'].shape)

# Average over repeats
cls_arr = np.mean(result_dict['class_scores'], -1) * 100.

cls_shape = cls_arr.shape

hyp_p = result_dict['hyper_params']

###################
# Plot
###################
plt.ion()

fig1, axes1 = plt.subplots(cls_shape[0], cls_shape[1], figsize=(7.5, 14),
                           sharex=True, sharey=True)
if len(axes1.shape) == 1:
    axes1 = axes1[:, np.newaxis]

#vmin, vmax = np.min(cls_arr), np.max(cls_arr)
vmin, vmax = 50., 60
cmap = plt.get_cmap('viridis')

for ind_2d in list(product(*[range(cls_shape[0]), range(cls_shape[1])])):
    im = axes1[ind_2d].imshow(cls_arr[ind_2d[0], ind_2d[1], :, :],
                              interpolation='none', vmin=vmin, vmax=vmax,
                              cmap=cmap)
    axes1[ind_2d].set_adjustable('box-forced')

    # Set ticks and labels to match L1/L2 loss
    axes1[ind_2d].set_xticks(range(cls_shape[3]))
    axes1[ind_2d].set_xticklabels(hyp_p[2], rotation=-45)

    axes1[ind_2d].set_yticks(range(cls_shape[2]))
    axes1[ind_2d].set_yticklabels(hyp_p[1])

    # Set batch
    if ind_2d[0] == 0:
        title = 'Batch size: %i' % hyp_p[0][ind_2d[1]]
        axes1[ind_2d].set_title(title)
    elif ind_2d[0] == len(result_dict['subj_nums']) - 1:
        axes1[ind_2d].set_xlabel('L2 loss weight')

    # Set y label to display subject
    if ind_2d[1] == 0:
        y_label = 'Subj: %s\n' % result_dict['subj_nums'][ind_2d[0]] + \
            'L1 loss weight'
        axes1[ind_2d].set_ylabel(y_label)

# Adjust spacing and add colorbar
fig1.subplots_adjust(hspace=0.1, wspace=0.1)
cax = fig1.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = fig1.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad = 15
cbar.set_label('% Accuracy', rotation=270)

plt.show()

if save_plot:
    save_fname = op.join(saved_data_dir, 'lr_gridsearch',
                         'lr_plot_%02i_subjs_%s%s' % (cls_shape[0],
                                                      result_dict['time_finished'],
                                                      shuffled_add))

    fig1.savefig(save_fname + '.png')
    fig1.savefig(save_fname + '.pdf')
