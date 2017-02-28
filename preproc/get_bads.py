"""
gen_bads.py

@author: wronk

Get all bad channels across all raw files and save.
This is needed so that all runs across all tasks can use one set of raw files
"""

from os import path as op

import hcp

from rsn import config as cf
from rsn.config import hcp_path

exp_types = ['rest', 'task_motor']

hcp_params = dict(hcp_path=hcp_path)

# Get all subject ID numbers, exclude missing rest/motor/story&math/working mem
dirs = cf.subj_nums_hcp
dirs = [str(temp_d) for temp_d in dirs]
dirs = dirs[0:1]

bad_ch_list = []
for subj_fold in dirs:
    print('%s\nSubj: %s' % (40 * '=', subj_fold))
    hcp_params = dict(hcp_path=hcp_path, subject=subj_fold)

    for exp_type in exp_types:

        if exp_type is 'task_motor':
            runs = cf.motor_params['runs']
        elif exp_type is 'rest':
            runs = cf.rest_params['runs']
        else:
            raise RuntimeError('Incorrect trial designation: %s' % exp_type)
        hcp_params['data_type'] = exp_type

        for run_i in runs:
            hcp_params['run_index'] = run_i

            # Load annotations and grab all bad channels
            annots = hcp.read_annot(**hcp_params)
            bad_ch_list.extend(annots['channels']['all'])

            print('\tExp: %s, Run: %i, bads: %s' %
                  (exp_type, run_i, annots['channels']['all']))

    ######################################################
    # Convert to set and save all bad channels as txt file
    bad_ch_set = set(bad_ch_list)
    print('\n\tSaving set: %s' % bad_ch_set)

    fname_bad = op.join(hcp_path, subj_fold, 'unprocessed', 'MEG',
                        'prebad_chans.txt')
    with open(fname_bad, 'wb') as out_file:
        for bad_ch in bad_ch_set:
            out_file.write('%s\n' % bad_ch)
