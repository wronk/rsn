"""
preproc_hcp.py

@author: wronk

Script to loop over HCP subjects, compute anatomy, fwd soln, and noise cov
"""

from os import path as op

import mne
import hcp
#import config
from rsn import config as cf
from rsn.config import hcp_path
from rsn.comp_fun import check_and_create_dir

stored_subjects_dir = '/media/Toshiba/MRI_Data/structurals'
new_subjects_dir = op.join(hcp_path, 'anatomy')
head_trans_dir = op.join(hcp_path, 'hcp-meg')

n_jobs = 6

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = list(set(dirs) - set(cf.hcp_subj_remove))
dirs = [str(temp_d) for temp_d in dirs]


for subj_fold in dirs:
    if True:
        # Construct MNE-compatible anatomy
        subj_dir = op.join(hcp_path, subj_fold)
        hcp.make_mne_anatomy(subj_fold, subjects_dir=stored_subjects_dir,
                             hcp_path=hcp_path,
                             recordings_path=hcp_path + '/hcp-meg')

    if True:
        # Construct fwd solution
        # Leaving defaults, so add_dist=True and uses rest data run_index=0
        src_outputs = hcp.anatomy.compute_forward_stack(
            subject=subj_fold,
            subjects_dir=stored_subjects_dir,
            hcp_path=hcp_path,
            recordings_path=head_trans_dir,
            n_jobs=6)

        # Save forward modeling information outputjs
        #src_outputs['bem_sol'].save(op.join(stored_subjects_dir, subj_fold,
        #                                    'bem', '%s-bem-sol.fif'))
        src_fold = op.join(stored_subjects_dir, subj_fold, 'src')
        fwd_fold = op.join(hcp_path, subj_fold, 'forward')
        check_and_create_dir(src_fold)
        check_and_create_dir(fwd_fold)

        mne.write_source_spaces(op.join(src_fold, 'subject-src.fif'),
                                src_outputs['src_subject'])
        mne.write_source_spaces(op.join(src_fold, 'fsaverage-src.fif'),
                                src_outputs['src_fsaverage'])
        mne.write_forward_solution(op.join(hcp_path, subj_fold, 'forward',
                                           '%s-meg-fwd.fif' % subj_fold),
                                   src_outputs['fwd'], overwrite=True)

    if True:
        raw_noise = hcp.read_raw(subject=subj_fold, hcp_path=hcp_path,
                                 data_type='noise_empty_room')
        raw_noise.load_data()

        # apply ref channel correction, drop ref channels, and filter
        hcp.preprocessing.apply_ref_correction(raw_noise)
        raw_noise.filter(0.50, None, method='iir',
                         iir_params=dict(order=4, ftype='butter'),
                         n_jobs=n_jobs)
        raw_noise.filter(None, 60, method='iir',
                         iir_params=dict(order=4, ftype='butter'),
                         n_jobs=n_jobs)

        # Compute and save covariance data
        noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')

        cov_fold = op.join(hcp_path, subj_fold, 'covariance')
        check_and_create_dir(cov_fold)
        mne.write_cov(op.join(cov_fold, '%s-erm-noise-cov.fif' % subj_fold),
                      noise_cov)
