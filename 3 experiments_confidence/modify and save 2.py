#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:09:18 2018

@author: nmei
"""

import pandas as pd
import os
working_dir = ''
batch_dir = 'batch'
if not os.path.exists(batch_dir):
    os.mkdir(batch_dir)
if not os.path.exists('batch/test_run'):
    os.mkdir('batch/test_run')

subs = ['ack', 'cc', 'ck', 'cpj', 'em', 'es', 'fd', 'jmac', 'lidia', 'ls','mimi', 'pr', 'pss', 'sva', 'tj']
for participant in subs:
    with open(os.path.join(batch_dir,'e2 (experiment and chance scores) ({}).py'.format(participant)),'wb') as new_file:
        with open('analysis e2.py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant     = 'ack'","participant = '{}'".format(participant)))

content = """
#!/bin/bash

# This is a script to send e2 (experiment and chance scores) ({}).py as a batch job.
# it works on dataset {}

#$ -cwd
#$ -o test_run/out_{}.txt
#$ -e test_run/err_{}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "e2_{}"
#$ -S /bin/bash

module load rocks-python-2.7

python "e2 (experiment and chance scores) ({}).py"
"""
for participant in subs:
    with open(os.path.join(batch_dir,'model_comparison_e2_{}'.format(participant)),'w') as f:
        f.write(content.format(participant,participant,participant,participant,participant,participant,participant))

content = '''
#!/bin/bash

# This is a script to qsub jobs

#$ -cwd
#$ -o test_run/out_q.txt
#$ -e test_run/err_q.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "qjobE2"
#$ -S /bin/bash


'''
with open('batch/qsub_jobs_e2','w') as f:
    f.write(content)

with open('batch/qsub_jobs_e2','a') as f:
    for participant in subs:
        f.write('qsub model_comparison_e2_{}\n'.format(participant))

with open('batch/utils.py','wb') as new_utils:
    with open('utils.py','rb') as the_utils:
        for line in the_utils:
            new_utils.write(line)









































