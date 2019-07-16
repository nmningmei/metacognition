#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:38:04 2018

@author: nmei
"""
import pandas as pd
import os
working_dir = ''
batch_dir = 'batch'
if not os.path.exists(batch_dir):
    os.mkdir(batch_dir)

content = '''
#!/bin/bash

# This is a script to qsub jobs

#$ -cwd
#$ -o test_run/out_q.txt
#$ -e test_run/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qjob"
#$ -S /bin/bash


'''
with open(os.path.join(batch_dir,'qsub_jobs'),'w') as f:
    f.write(content)





df         = pd.read_csv(os.path.join(working_dir,'../data/PoSdata.csv'))
df         = df[df.columns[1:]]
df.columns = ['participant',
              'blocks',
              'trials',
              'firstgabor',
              'success',
              'tilted',
              'correct',
              'RT_correct',
              'awareness',
              'RT_awareness',
              'confidence',
              'RT_confidence']
participants = pd.unique(df['participant'])
# estimate the experimental score
for participant in participants:
    with open(os.path.join(batch_dir,'classifcation_pos_n_trials_back (experiment score) ({}).py'.format(participant)),'wb') as new_file:
        with open('classifcation_pos_n_trials_back (experiment score).py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant = 'AC'","participant = '{}'".format(participant)))
# estimator chance level score
for participant in participants:
    with open(os.path.join(batch_dir,'classifcation_pos_n_trials_back (empirical chance level) ({}).py'.format(participant)),'wb') as new_file:
        with open('classifcation_pos_n_trials_back (empirical chance level).py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant = 'AC'","participant = '{}'".format(participant)))

content = """
#!/bin/bash

# This is a script to send classifcation_pos_n_trials_back (empirical chance level) ({}).py as a batch job.
# it works on dataset {}

#$ -cwd
#$ -o test_run/out_{}.txt
#$ -e test_run/err_{}.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "pos_{}"
#$ -S /bin/bash

module load rocks-python-2.7

python "classifcation_pos_n_trials_back (experiment score) ({}).py"
python "classifcation_pos_n_trials_back (empirical chance level) ({}).py"
"""

for participant in participants:
    with open(os.path.join(batch_dir,'model_comparison_pos_{}'.format(participant)),'w') as f:
        f.write(content.format(participant,participant,participant,participant,participant,participant,participant))

with open(os.path.join(batch_dir,'qsub_jobs'),'a') as f:
    for participant in participants:
        f.write('qsub model_comparison_pos_{}\n'.format(participant))


df         = pd.read_csv(os.path.join(working_dir,'../data/ATTfoc.csv'))
df         = df[df.columns[1:]]
df.columns = ['participant',
              'blocks',
              'trials',
              'firstgabor',
              'attention',
              'tilted',
              'correct',
              'RT_correct',
              'awareness',
              'RT_awareness',
              'confidence',
              'RT_confidence']
participants = pd.unique(df['participant'])
batch_dir = 'batch'
if not os.path.exists(batch_dir):
    os.mkdir(batch_dir)
# estimate the experimental score
for participant in participants:
    with open(os.path.join(batch_dir,'classifcation_att_n_trials_back (experiment score) ({}).py'.format(participant)),'wb') as new_file:
        with open('classifcation_att_n_trials_back (experiment score).py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant = 'AS'","participant = '{}'".format(participant)))
# estimator chance level score
for participant in participants:
    with open(os.path.join(batch_dir,'classifcation_att_n_trials_back (empirical chance level) ({}).py'.format(participant)),'wb') as new_file:
        with open('classifcation_att_n_trials_back (empirical chance level).py','rb') as old_file:
            for line in old_file:
                new_file.write(line.replace("participant = 'AS'","participant = '{}'".format(participant)))

content = """
#!/bin/bash

# This is a script to send classifcation_att_n_trials_back (empirical chance level) ({}).py as a batch job.
# it works on dataset {}

#$ -cwd
#$ -o test_run/out_{}.txt
#$ -e test_run/err_{}.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "att_{}"
#$ -S /bin/bash

module load rocks-python-2.7

python "classifcation_att_n_trials_back (experiment score) ({}).py"
python "classifcation_att_n_trials_back (empirical chance level) ({}).py"
"""
for participant in participants:
    with open(os.path.join(batch_dir,'model_comparison_att_{}'.format(participant)),'w') as f:
        f.write(content.format(participant,participant,participant,participant,participant,participant,participant))




with open(os.path.join(batch_dir,'qsub_jobs'),'a') as f:
    for participant in participants:
        f.write('qsub model_comparison_att_{}\n'.format(participant))































