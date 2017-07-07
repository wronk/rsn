"""
gen_anat_fwd_cov.py

@author: wronk

Script to loop over HCP subjects, compute anatomy, fwd soln, and noise cov
"""

from os import path as op

import mne
import hcp
from rsn import config as cf
from rsn.config import hcp_path, motor_params, rest_params

from rsn.comp_fun import check_and_create_dir, preproc_gen_ssp

stored_subjects_dir = '/media/Toshiba/MRI_Data/structurals'
new_subjects_dir = op.join(hcp_path, 'anatomy')
head_trans_dir = op.join(hcp_path, 'hcp-meg')
apply_ref_correction = False

n_jobs = 6

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]

for subj_fold in dirs:
    if False:
        # Construct MNE-compatible anatomy
        subj_dir = op.join(hcp_path, subj_fold)
        hcp.make_mne_anatomy(subj_fold, subjects_dir=stored_subjects_dir,
                             hcp_path=hcp_path,
                             recordings_path=hcp_path + '/hcp-meg')

    if False:
        # Construct fwd solutions
        # Leaving most defaults for src_params. Defaults are:
        #     src_params = dict(subject='fsaverage', fname=None,
        #                       spacing='oct6', n_jobs=2, surface='white',
        #                       subjects_dir=subjects_dir, add_dist=True)
        src_params = dict(subject='fsaverage', fname=None, spacing='ico5',
                          n_jobs=6, surface='white', overwrite=True,
                          subjects_dir=stored_subjects_dir, add_dist=True)
        src_outputs = hcp.anatomy.compute_forward_stack(
            subject=subj_fold,
            subjects_dir=stored_subjects_dir,
            hcp_path=hcp_path,
            recordings_path=head_trans_dir,
            src_params=src_params,
            n_jobs=6)

        # Save forward modeling information outputs
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
        # Construct SSP projectors using all raw objects
        raw_list =[]
        hcp_params = dict(hcp_path=hcp_path, subject=subj_fold)

        for run_i in rest_params['runs']:
            hcp_params.update(dict(data_type='rest', run_index=run_i))
            raw = hcp.read_raw(**hcp_params)
            raw.load_data()

            annots = hcp.read_annot(**hcp_params)
            bad_seg = (annots['segments']['all']) / raw.info['sfreq']
            annotations = mne.Annotations(
                bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
                description='bad')

            raw.annotations = annotations

            raw_list.append(raw)

        for run_i in motor_params['runs']:
            hcp_params.update(dict(data_type='task_motor', run_index=run_i))
            raw = hcp.read_raw(**hcp_params)
            raw.load_data()

            annots = hcp.read_annot(**hcp_params)
            bad_seg = (annots['segments']['all']) / raw.info['sfreq']
            annotations = mne.Annotations(
                bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
                description='bad')

            raw.annotations = annotations

            raw_list.append(raw)

        full_raw = mne.concatenate_raws(raw_list)
        full_raw.pick_types(meg=True, eog=True, ecg=True, ref_meg=False)

        del raw_list
        preproc_gen_ssp(subj_fold, full_raw)
        del full_raw


    if False:
        # Generate covariance matrices from ERM data
        raw_noise = hcp.read_raw(subject=subj_fold, hcp_path=hcp_path,
                                 data_type='noise_empty_room')
        raw_noise.load_data()

        # apply ref channel correction, drop ref channels, and filter
        if apply_ref_correction:
            hcp.preprocessing.apply_ref_correction(raw_noise)
        raw_noise.filter(cf.filt_params['l_freq'], None, method='iir',
                         iir_params=dict(order=4, ftype='butter'),
                         n_jobs=n_jobs)
        raw_noise.filter(None, cf.filt_params['h_freq'], method='iir',
                         iir_params=dict(order=4, ftype='butter'),
                         n_jobs=n_jobs)

        # Compute and save covariance data
        noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')

        cov_fold = op.join(hcp_path, subj_fold, 'covariance')
        check_and_create_dir(cov_fold)
        mne.write_cov(op.join(cov_fold, '%s-erm-noise-cov.fif' % subj_fold),
                      noise_cov)
