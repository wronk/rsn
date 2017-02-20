"""
gen_rest_epochs_hcp.py

@author: wronk

Script to loop over HCP subjects, compute rest epochs
"""

from os import path as op
import numpy as np

import mne
from mne.minimum_norm import write_inverse_operator as write_inv
import hcp
import hcp.preprocessing as preproc
from hcp.preprocessing import interpolate_missing as interp_missing

from rsn import config as cf
from rsn.config import hcp_path, rest_params
from rsn.comp_fun import check_and_create_dir

stored_subjects_dir = '/media/Toshiba/MRI_Data/structurals'
new_subjects_dir = op.join(hcp_path, 'anatomy')
head_trans_dir = op.join(hcp_path, 'hcp-meg')

n_jobs = 6

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = list(set(dirs) - set(cf.hcp_subj_remove))
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

hcp_params = dict(hcp_path=hcp_path, data_type='rest')

for subj_fold in dirs:
    hcp_params['subject'] = subj_fold

    # Construct folder for saving epochs
    epo_fold = op.join(hcp_path, subj_fold, 'epochs')
    check_and_create_dir(epo_fold)

    # Construct trial info for each run
    # Note: need unprocessed data and preprocessed data (to get trial info
    #       information in this step)
    '''
    trial_infos = []
    for run_index in rest_params['runs']:
        hcp_params['run_index'] = run_index
        trial_info = hcp.read_trial_info(**hcp_params)
        trial_infos.append(trial_info)

    print('\n' + 'Data info:' + '\n')
    print(trial_info['stim']['comments'][:10])  # which column? # 3
    print(set(trial_info['stim']['codes'][:, 3]))  # check values

    all_events = []
    for trial_info in trial_infos:
        events = np.c_[
            trial_info['stim']['codes'][:, rest_params['time_samp_col']] - 1,
            np.zeros(len(trial_info['stim']['codes'])),
            trial_info['stim']['codes'][:, rest_params['stim_code_col']]
        ].astype(int)

        # Unfortunately, HCP data the time events may not always be unique
        unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]
        events = events[unique_subset]  # use diff to find first unique events

        all_events.append(events)
    '''

    ###########################################################################
    # Construct epochs

    epo_list = []
    for run_index in rest_params['runs']:

        hcp_params['run_index'] = run_index

        raw = hcp.read_raw(**hcp_params)
        raw.load_data()

        # Make 1 second events
        events = mne.make_fixed_length_events(raw, id=1, duration=1.)
        #mne.write_events(eve_name, eve)

        # apply ref channel correction and drop ref channels
        preproc.apply_ref_correction(raw)

        # construct MNE annotations
        annots = hcp.read_annot(**hcp_params)
        bad_seg = (annots['segments']['all']) / raw.info['sfreq']
        annotations = mne.Annotations(
            bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
            description='bad')

        raw.annotations = annotations
        raw.info['bads'].extend(annots['channels']['all'])
        raw.pick_types(meg=True, ref_meg=False)

        # Note: MNE complains on Python 2.7
        raw.filter(0.50, None, method='iir',
                   iir_params=dict(order=4, ftype='butter', output='sos'),
                   n_jobs=n_jobs)
        raw.filter(None, 60, method='iir',
                   iir_params=dict(order=4, ftype='butter', output='sos'),
                   n_jobs=n_jobs)

        # Read ICA and remove EOG ECG
        # Note that the HCP ICA assumes that bad channels have been removed
        ica_mat = hcp.read_ica(**hcp_params)

        # We will select the brain ICs only
        exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
                   if ii not in annots['ica']['brain_ic_vs']]
        preproc.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)

        # Create and save epochs
        #     No baselining applied, XXX update rejects

        events = np.sort(events, 0)
        epochs = mne.Epochs(raw, events=events,
                            event_id=rest_params['event_id'],
                            tmin=rest_params['tmin'],
                            tmax=rest_params['tmax'],
                            reject=None, baseline=None,
                            decim=rest_params['decim'], preload=True)

        # Add back out channels for comparison across runs
        epochs = interp_missing(epochs, subject=subj_fold,
                                data_type='rest', run_index=run_index,
                                hcp_path=hcp_path, mode='accurate')

        epo_fname = op.join(epo_fold, '%s_rest_run%i-epo.fif' %
                            (subj_fold, run_index))
        epochs.save(epo_fname)

        epo_list.append(epochs)
        del raw#, epochs

    '''
    ###########################################################################
    # Construct inverse solutions

    # Load fwd
    # XXX: force fixed?
    fwd_fname = op.join(hcp_path, subj_fold, 'forward',
                        '%s-meg-fwd.fif' % subj_fold)
    fwd = mne.read_forward_solution(fwd_fname)

    # Load noise covariance
    noise_cov_fname = op.join(hcp_path, subj_fold, 'covariance',
                              '%s-erm-noise-cov.fif' % subj_fold)
    noise_cov = mne.read_cov(noise_cov_fname)

    inv = mne.minimum_norm.make_inverse_operator(epo_list[-1].info, fwd,
                                                 noise_cov=noise_cov)

    ###########################################################################
    # Save inverse solutions

    inv_fold = op.join(hcp_path, subj_fold, 'inverse')
    check_and_create_dir(inv_fold)
    write_inv(op.join(inv_fold, '%s_rest-inv.fif' % subj_fold), inv)
    '''
