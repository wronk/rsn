"""
comp_connectivity_v1.py

@author: wronk

Compute connectivity metrics between RSN areas
"""

import os
import os.path as op
import numpy as np
import cPickle
from copy import deepcopy

import mne
from mne import read_epochs, read_source_spaces
from mne.minimum_norm import apply_inverse_epochs as apply_inv, read_inverse_operator as read_inv
from mne.connectivity import spectral_connectivity as conn
from mne.label import read_label

################
# Define globals
################
save_data = False  # Whether or not to pickle data

data_dir = op.join(os.environ['CODE_ROOT'], 'rsn_data')
struct_dir = os.environ['SUBJECTS_DIR']

subj_nums = ['004', '007', '015', '017', '019', '020', '023', '031', '032',
             '034', '038']
subj_nums = ['017']


# TODO: check labels for each subject, not all have the same number of
# divisions

# Divisions on fsaverage
# div1: MPFC (Medial frontal region)
# div2: STS (Superior temporal sulcus)
# div3: PCC, pCun (Posterior cing. cort., preCuneus)
# div4: TPJ, (temporoparietal junction)
# div5: XXX Name? Small region on medial/ventral side of temporal lobe

# Define processing parameters
common_conn_kwargs = dict(fmin=2., fmax=50., fskip=4)
conn_methods = [dict(method='coh').update(common_conn_kwargs),
                dict(method='pli').update(common_conn_kwargs)]

conn_params = dict(conn_pairs=[(2, 0), (2, 4)],
                   conn_methods=conn_methods,
                   rsn_labels=['lh.7Networks_7_div%i.label' % i
                               for i in range(1, 6)],
                   subj_nums=subj_nums,
                   mean_mode='mean_flip',
                   n_jobs=6)


##################
# Helper functions
##################
def get_lab_list(conn_params, s_name):
    """Helper to get list of labels and a summed master label"""

    assert('rsn_labels' in conn_params.keys())
    print 'Loading %i labels for subject: %s' % \
        (len(conn_params['rsn_labels']), s_name)

    lab_list = []

    for label_name in conn_params['rsn_labels']:
        fname_label = op.join(struct_dir, s_name, 'label', label_name)
        lab_list.append(read_label(fname_label, subject=s_name))
        #summed_label += lab_list[-1]

    summed_label = deepcopy(lab_list[0])
    for temp_lab in lab_list[1:]:
        summed_label += temp_lab

    return lab_list, summed_label

###########################################################
# Compute connectivity between RSN regions for each subject
###########################################################

for s_num in conn_params['subj_nums']:
    s_name = 'AKCLEE_' + s_num
    struct_name = 'AKCLEE_1' + s_num[-2:]

    # Load Epochs
    fname_epo = op.join(data_dir, 'wronk_resting_%s' % s_num, 'epochs',
                        'All_55-sss_wronk_resting_%s-epo.fif' % s_num)
    epo = read_epochs(fname_epo)

    # Get list of labels and summed label to facilitate source estimation
    lab_list, summed_label = get_lab_list(conn_params, struct_name)

    # Generate source activity restricted to pre-defined RSN areas
    fname_inv = op.join(data_dir, 'wronk_resting_%s' % s_num, 'inverse',
                        'wronk_resting_%s-55-sss-meg-erm-fixed-inv.fif' % s_num)
    inv = read_inv(fname_inv)
    stc = apply_inv(epo, inv, lambda2=1. / 9., label=summed_label,
                    method='MNE')

    # XXX: Could morph to fsaverage if we want to
    src = read_source_spaces(op.join(struct_dir, struct_name, 'bem',
                                     struct_name + '-7-src.fif'),
                             verbose=False)

    # Compute connectivity
    conn_results = []
    for label_pair in conn_params['conn_pairs']:
        import ipdb; ipdb.set_trace()
        traces = [mne.extract_label_time_course(stc[0], lab_list[li], src,
                                                mode=conn_params['mean_mode'])
                  for li in label_pair]

        for conn_dict in conn_params['conn_methods']:
            temp_conn = conn(np.array(traces),
                             indicies=(np.array([0]), np.array([1])),
                             sfreq=epo.info['sfreq'], n_jobs=params['n_jobs'],
                             **conn_dict)

            conn_results.append(temp_conn)

    ##############
    # Save Results
    ##############
    if save_data:

        # Check if save directory exists (and create it if necessary)
        save_dir = op.join(data_dir, 'wronk_resting_%s' % s_num,
                           'connectivity')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Save results as pkl file
        save_file = op.join(save_dir, 'conn_results.pkl')
        with open(save_file, 'wb') as pkl_obj:
            results_to_save = deepcopy(conn_params)
            results_to_save['conn_results'] = conn_results

            cPickle.dump(results_to_save, pkl_obj)
