'''
gen_motorLabels.py

Generate motor-BCI relevant labels

@author: wronk
'''

import os
import os.path as op
import mne
import numpy as np
from scipy.spatial.distance import cdist

from rsn.config import subj_nums_hcp

# whether or not to morph hand motor labels to all other subjects
doMorph = True

subjectDir = os.environ['SUBJECTS_DIR']
modelSubj = 'fsaverage'

# Distance from fMRI-based central point to include in hand motor label
labelRadii = [10]
subject_set = [str(subj) for subj in subj_nums_hcp]

# Load list of labels and pull out pre-central gyrus
labelList = mne.read_labels_from_annot(subject=modelSubj, parc='aparc.a2009s',
                                       hemi='both')
primaryMotor = [l for l in labelList if 'G_precentral-' in l.name]

# Get MNI coords for vertices in L/R hemis
primaryMotorMNI_pos = [mne.vertex_to_mni(primaryMotor[0].vertices, 0, modelSubj),
                       mne.vertex_to_mni(primaryMotor[1].vertices, 1, modelSubj)]

# Find closest point in fsaverage brain space according to
# Witt 2008, Functional neuroimaging correlates of finger-tapping task
# variations coords in Talairach space
# list is [left hemi, right hemi]
hand_knob_pos = [np.atleast_2d(np.array([-38, -26, 50])),
                 np.atleast_2d(np.array([36, -22, 54]))]
SMA = [-4, -8, 52]

dists = []
for orig_label, label_pos, knob_pos in zip(primaryMotor, primaryMotorMNI_pos, hand_knob_pos):
    # Find dist between MNI point and all vertices
    dists.append(np.squeeze(cdist(knob_pos, label_pos, 'euclidean')))
    # Find min dist and index
    min_dist, ind = min((min_dist, ind) for (ind, min_dist) in enumerate(dists[-1]))
    print 'Min dist: %02.02f Ind: %i' % (min_dist, ind)

    # Set seed vertex (closest vert to MNE point)
    seed = [orig_label.vertices[ind]]
    seed = seed * len(labelRadii)

    # Generate circular labels
    hemi_i = ['lh', 'rh'].index(orig_label.hemi)
    handMotorLabels = mne.grow_labels(modelSubj, seed, labelRadii, hemi_i, n_jobs=6)

    # find intersection circular labels with pre-central sulcus label for each radii
    overlapInds = [np.in1d(np.array(synthetic_label.vertices),
                           np.array(orig_label.vertices))
                   for synthetic_label in handMotorLabels]

    # Restrict vertices in synthetic labels
    for i, label in enumerate(handMotorLabels):

        label.vertices = label.vertices[overlapInds[i]]
        label.pos = label.pos[overlapInds[i]]
        label.values = label.values[overlapInds[i]]

        label.subject = modelSubj
        label.name = 'G_precentral_handMotor_radius_' + str(labelRadii[i]) + 'mm'
        #labels_motor.append(label)

        #subjectFolder_fname = op.join(savePath, 'HandMotor', label.subject)
        #if not op.exists(subjectFolder_fname):
        #    os.makedirs(subjectFolder_fname)

        #labelSavePath = op.join(subjectFolder_fname, label.name + '-' + hemi +
        #                        '.label')
        #mne.write_label(labelSavePath, label)

        # Morph to subjects of interest and save
        if doMorph:
            for subject in subject_set:
                # Copy necessary to prevent modification in place of label obj
                label_copy = label.copy()
                morphedLabel = label_copy.morph(subject_from=modelSubj,
                                                subject_to=subject,
                                                smooth=5, n_jobs=6)
                labelSavePath = op.join(subjectDir, subject, 'label',
                                        morphedLabel.hemi + '.' +
                                        morphedLabel.name + '.label')
                morphedLabel.save(labelSavePath)
