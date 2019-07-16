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

subs = ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']
for participant in subs:
    with open(os.path.join(batch_dir,'e1 (experiment and chance scores) ({}).py'.format(participant)),'wb') as new_file:
        with open('analysis e1.py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant     = 'ah'","participant = '{}'".format(participant)))

content = """
#!/bin/bash

# This is a script to send e1 (experiment and chance scores) ({}).py as a batch job.
# it works on dataset {}

#$ -cwd
#$ -o test_run/out_{}.txt
#$ -e test_run/err_{}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "e1_{}"
#$ -S /bin/bash

module load rocks-python-2.7

python "e1 (experiment and chance scores) ({}).py"
"""
for participant in subs:
    with open(os.path.join(batch_dir,'model_comparison_e1_{}'.format(participant)),'w') as f:
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
#$ -N "qjobE1"
#$ -S /bin/bash


'''
with open('batch/qsub_jobs_e1','w') as f:
    f.write(content)

with open('batch/qsub_jobs_e1','a') as f:
    for participant in subs:
        f.write('qsub model_comparison_e1_{}\n'.format(participant))

with open('batch/utils.py','wb') as new_utils:
    with open('utils.py','rb') as the_utils:
        for line in the_utils:
            new_utils.write(line)


batch_dir = 'batch/correlation'
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)
if not os.path.exists(os.path.join(batch_dir,'test_run')):
    os.mkdir(os.path.join(batch_dir,'test_run'))

#subs = ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']
for participant in subs:
    with open(os.path.join(batch_dir,'correlate predicted awareness with correctness e1 ({}).py'.format(participant)),'wb') as new_file:
        with open('correlate predicted awareness with correctness e1.py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant     = 'ah'","participant = '{}'".format(participant)))

content = """
#!/bin/bash

# This is a script to send correlate predicted awareness with correctness e1 ({}).py as a batch job.
# it works on dataset {}

#$ -cwd
#$ -o test_run/out_{}.txt
#$ -e test_run/err_{}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "e1_{}"
#$ -S /bin/bash

module load rocks-python-2.7

python "correlate predicted awareness with correctness e1 ({}).py"
"""
for participant in subs:
    with open(os.path.join(batch_dir,'model_comparison_e1_{}'.format(participant)),'w') as f:
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
#$ -N "qjobE1"
#$ -S /bin/bash


'''
with open(os.path.join(batch_dir,'qsub_jobs_e1'),'w') as f:
    f.write(content)

with open(os.path.join(batch_dir,'qsub_jobs_e1'),'a') as f:
    for participant in subs:
        f.write('qsub model_comparison_e1_{}\n'.format(participant))

with open(os.path.join(batch_dir,'utils.py'),'wb') as new_utils:
    with open('utils.py','rb') as the_utils:
        for line in the_utils:
            new_utils.write(line)




































