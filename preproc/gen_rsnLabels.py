'''
gen_rsnLabels.py

Morph and split Yeo 2011 labels from fsaverage to all other subjects

@Author: wronk
'''

import os
import os.path as op
from surfer import Brain
import mne
from copy import deepcopy
import config

# Whether or not to save divided labels for fsaverage
save_fs_divs = False
# whether or not to morph labels to all other subjects
doMorph = True
# Whether or not to plot brain
doPlot = True

subjectDir = os.environ['SUBJECTS_DIR']
modelSubj = 'fsaverage'
hemi = 'lh'
surface = 'inflated'
views = ['lat']

hcp_path = '/media/Toshiba/Code/hcp_data'
new_subjects_dir = op.join(hcp_path, 'anatomy')
# Get all subject ID numbers
#subject_list = [d for d in os.listdir(new_subjects_dir)
#                if len(d) is 6 and os.path.isdir(op.join(hcp_path, d))]
subject_list = [str(num) for num in config.subj_num_hcp]

vert_cutoff = 50  # Min number of vertices to designate a label
n_smooth = 1

#subject_list = ['AKCLEE_101']

# 7Networks_3 cooresponds to DAN
# 7Networks_7 cooresponds to DMN

#######################
# Load and morph labels
#######################

# Load LH and RH labels from Yeo, 2011
labels = ['lh.7Networks_%i.label' % net_num for net_num in range(1, 8)]
labels.extend(['rh.7Networks_%i.label' % net_num for net_num in range(1, 8)])

# Load original RSN labels
orig_label_list = [mne.read_label(op.join(subjectDir, modelSubj, 'label', l),
                                  subject=modelSubj) for l in labels]
# Split each RSN label into contiguous regions and store
split_labels_temp = [label.split(parts='contiguous')
                     for label in orig_label_list]

# Create one list from list of lists
labels_to_morph = []
for label_set in split_labels_temp:
    labels_to_morph.extend([l for l in label_set
                            if len(l.vertices) > vert_cutoff])
labels_to_morph.extend(deepcopy(orig_label_list))

# Save divisions of labels for fsaverage
if save_fs_divs:
    print 'Saving divided labels for `fsaverage`'
    for i, label in enumerate(labels_to_morph):
        if 'div' in label.name:
            labelSavePath = op.join(subjectDir, 'fsaverage', 'label',
                                    label.name[:-3] + '.label')
            label.save(labelSavePath)

# Actually morph labels from fsaverage to other subjects
if doMorph:
    print 'Morphing %i labels to %i subjects...\n' % (len(labels_to_morph),
                                                      len(subject_list))

    for i, label in enumerate(labels_to_morph):
        for subject in subject_list:
            print '\n' + subject
            label.values.fill(1.0)
            morphedLabel = label.morph(subject_from=modelSubj,
                                       subject_to=subject, smooth=n_smooth,
                                       n_jobs=6)
            labelSavePath = op.join(subjectDir, subject, 'label',
                                    morphedLabel.name[:-3] + '.label')
            morphedLabel.save(labelSavePath)
print 'Morphing Complete'

###########
# Plot
###########
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
color_hexes = ['4C0000', 'FF4D4D', 'FF0000', 'FF8080', 'FFCCCC']

if doPlot:
    brain = Brain(modelSubj, hemi='lh', surf='white', views=views,
                  show_toolbar=False)

    for li, label in enumerate([l for l in orig_label_list if l.hemi == 'lh']):
        brain.add_label(label=label, color=(colors * 2)[li], alpha=0.7,
                        hemi=label.hemi)
