"""
plot_gridsearch_rf.py

author: @wronk

Plots saved random forest gridsearch data
"""

import os
import os.path as op
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

import cPickle

################
# Define globals
################
save_plot = True  # Whether or not to pickle data
shuffled = False

struct_dir = os.environ['SUBJECTS_DIR']
data_head = op.join(os.environ['CODE_ROOT'])
saved_data_dir = op.join(data_head, 'rsn_results')

shuffled_add = '_shuffled' if shuffled else ''
#file_name = 'rf_scores_2016_11_28__18_43%s.pkl' % shuffled_add
file_name = 'rf_scores_2016_12_01__00_30%s.pkl' % shuffled_add

###################
# Load pickled data
###################
fpath_data = op.join(saved_data_dir, file_name)
with open(fpath_data) as pkl_file:
    result_dict = cPickle.load(pkl_file)

#print 'Loaded data (%s) from %s' % (fpath_data, result_dict['time_finished'])
print 'Hyper param format: %s' % result_dict['hyper_param_desc']
print 'Hyper param size: %s' % str(result_dict['class_scores'].shape)

# Average over repeats and cross-validation folds
cls_arr = np.mean(result_dict['class_scores'], (-2, -1)) * 100.

# Roll so order is (subj, batch, l1, l2)
#cls_arr = np.rollaxis(cls_arr, 3, 1)  # Corrected in new `classify_lr` script
cls_shape = cls_arr.shape

hyp_p = result_dict['hyper_params']

###################
# Plot
###################
plt.ion()

fig1, axes1 = plt.subplots(1, cls_shape[0], figsize=(16, 5), sharey=True)
axes1 = np.atleast_1d(axes1)
#vmin, vmax = np.min(cls_arr), np.max(cls_arr)
vmin, vmax = 50., 65
cmap = plt.get_cmap('viridis')

for ax_i, ax in enumerate(axes1):
    im = ax.imshow(cls_arr[ax_i, :, :], interpolation='none', vmin=vmin,
                   vmax=vmax, cmap=cmap)
    ax.set_adjustable('box-forced')

    # Set ticks and labels to match C and gamma params
    ax.set_xticks(range(cls_shape[2]))
    ax.set_xticklabels(hyp_p[1], rotation=-45, fontsize=10)

    ax.set_yticks(range(cls_shape[1]))
    ax.set_yticklabels(hyp_p[0], fontsize=10)

    # Set labels/title
    ax.set_title('Subject: %i' % result_dict['subj_nums'][ax_i])

    ax.set_xlabel('Max features')
    ax.set_ylabel('Num. Estimators')

# Adjust spacing and add colorbar
fig1.subplots_adjust(hspace=0.1, wspace=0.25)
cax = fig1.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = fig1.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad = 15
cbar.set_label('% Accuracy', rotation=270)

#############################
# Average over all subjects
#############################
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
#vmin, vmax = np.min(cls_arr), np.max(cls_arr)

im = ax2.imshow(cls_arr.mean(0), interpolation='none', vmin=vmin, vmax=vmax,
                cmap=cmap)
# Set ticks and labels to match C and gamma params
ax2.set_xticks(range(cls_shape[2]))
ax2.set_xticklabels(hyp_p[1], rotation=-45, fontsize=10)

ax2.set_yticks(range(cls_shape[1]))
ax2.set_yticklabels(hyp_p[0], fontsize=10)

# Set labels/title
ax2.set_title('Average over %i subjects' % cls_arr.shape[0])
ax2.set_xlabel('Max features')
ax2.set_ylabel('Num. Estimators')


# Adjust spacing and add colorbar
fig2.tight_layout(pad=4)
cax2 = fig2.add_axes([0.85, 0.25, 0.02, 0.6])
cbar2 = fig2.colorbar(im, cax=cax2)
cbar2.ax.get_yaxis().labelpad = 15
cbar2.set_label('% Accuracy', rotation=270)

plt.show()

###################
# Save png and pdf
###################
if save_plot:
    save_fname_1 = op.join(saved_data_dir, 'rf_gridsearch',
                           'rf_plot_%02i_subjs_%s%s' % (cls_shape[0],
                                                        result_dict['time_finished'],
                                                        shuffled_add))
    save_fname_2 = op.join(saved_data_dir, 'rf_gridsearch',
                           'rf_plot_%02i_subjs_avg_%s%s' % (cls_shape[0],
                                                            result_dict['time_finished'],
                                                            shuffled_add))

    fig1.savefig(save_fname_1 + '.png')
    fig1.savefig(save_fname_1 + '.pdf')

    fig2.savefig(save_fname_2 + '.png')
    fig2.savefig(save_fname_2 + '.pdf')
