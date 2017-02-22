"""
gen_invs.py

@author: wronk

Script to loop over HCP subjects and compute inverses
Note: run after motor epochs are constructed
"""

from os import path as op

import mne
from mne.minimum_norm import write_inverse_operator as write_inv

from rsn import config as cf
from rsn.config import hcp_path
from rsn.comp_fun import check_and_create_dir

stored_subjects_dir = '/media/Toshiba/MRI_Data/structurals'
new_subjects_dir = op.join(hcp_path, 'anatomy')
head_trans_dir = op.join(hcp_path, 'hcp-meg')

n_jobs = 6
exp_type = 'motor'

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

data_type = 'task_motor' if exp_type == 'motor' else 'rest'
hcp_params = dict(hcp_path=hcp_path, data_type=data_type)

for subj_fold in dirs:

    ###########################################################################
    # Construct inverse solutions

    # Load fwd
    # XXX: force fixed?
    fwd_fname = op.join(hcp_path, subj_fold, 'forward',
                        '%s-meg-fwd.fif' % subj_fold)
    fwd = mne.read_forward_solution(fwd_fname)

    # Load noise covariance and epochs (for info object)
    noise_cov_fname = op.join(hcp_path, subj_fold, 'covariance',
                              '%s-erm-noise-cov.fif' % subj_fold)
    noise_cov = mne.read_cov(noise_cov_fname)

    epo_fname = op.join(hcp_path, subj_fold, 'epochs', '%s_%s_run0-epo.fif' %
                        (subj_fold, exp_type))
    temp_epo = mne.read_epochs(epo_fname, proj=False, preload=False)

    # Calculate inverse
    inv = mne.minimum_norm.make_inverse_operator(temp_epo.info, fwd,
                                                 noise_cov=noise_cov)

    ###########################################################################
    # Save inverse solutions

    inv_fold = op.join(hcp_path, subj_fold, 'inverse')
    check_and_create_dir(inv_fold)
    write_inv(op.join(inv_fold, '%s_%s-inv.fif' % subj_fold, exp_type), inv)
