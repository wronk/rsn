'''
gen_rsnLabels.py

Morph and split Yeo 2011 labels from fsaverage to all other subjects

@Author: wronk
'''

import os
import os.path as op
from surfer import Brain
import mne

# Whether or not to save divided labels for fsaverage
save_fs_divs = True
# whether or not to morph labels to all other subjects
doMorph = False
# Whether or not to plot brain
doPlot = False

subjectDir = os.environ['SUBJECTS_DIR']
modelSubj = 'fsaverage'
hemi = 'lh'
surface = 'inflated'
views = ['lat']

nums = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
        '113', '114', '115', '118', '119', '120', '121', '122', '123', '124',
        '125', '126', '127', '128', '129', '130', '131', '132', '133', '134',
        '135', '136', '137', '138', '139']
nums = ['117']
subject_list = ['AKCLEE_' + num for num in nums]
vert_cutoff = 50
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
labels_to_morph.extend(orig_label_list)

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

    for li, label in enumerate(orig_label_list):
        brain.add_label(label=label, color=(colors * 2)[li], alpha=0.7,
                        hemi=label.hemi)
