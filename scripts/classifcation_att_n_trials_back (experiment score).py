# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:02:16 2018

@author: ning
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from utils import (classification)
saving_dir = '../../results/experiment_score'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

# Exp 2
#['AS', 'BG', 'EU', 'IK', 'JD', 'JZ', 'KK', 'KS', 'OE', 'OS', 'PC','RL', 'SO', 'SP', 'WT', 'YS']
participant = 'AS'
experiment = 'att'
df         = pd.read_csv(os.path.join(working_dir,'../../data/ATTfoc.csv'))
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
df_sub = df[df['participant'] == participant]
# make sure all the attributes are either 0 or 1
df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
##################################################################
np.random.seed(12345)
# use all 6 possible features
feature_names = [
                 'correct',
                 'awareness',
                 'confidence',
                 'RT_correct',
                 'RT_awareness',
                 'RT_confidence']
target_name = 'attention'
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correct          = [],
               awareness        = [],
               confidence       = [],
               RT_correct       = [],
               RT_awareness     = [],
               RT_confidence    = [],
               )

for n_back in range(5): # loop through the number of trials looking back
# this is the part that is redundent and the code is long
    results          = classification(
                                      df_sub,
                                      feature_names,
                                      target_name,
                                      results,
                                      participant,
                                      experiment,
                                      window=n_back,
                                      chance = False,
                                      ) 
temp = pd.DataFrame(results)
temp.to_csv(os.path.join(saving_dir,'att_6_features (experiment score)_{}.csv'.format(participant)),index=False) # save as a csv
################################################################################
# use success, awareness, and confidence as features
np.random.seed(12345)
# use judgement features
feature_names = [
                 'correct',
                 'awareness',
                 'confidence',]
target_name = 'attention'
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correct          = [],
               awareness        = [],
               confidence       = [],
               )

for n_back in range(5): # loop through the number of trials looking back
# this is the part that is redundent and the code is long
    results          = classification(
                                      df_sub,
                                      feature_names,
                                      target_name,
                                      results,
                                      participant,
                                      experiment,
                                      window=n_back,
                                      chance = False,
                                      ) 
temp = pd.DataFrame(results)
temp.to_csv(os.path.join(saving_dir,'att_3_1_features (experiment score)_{}.csv'.format(participant)),index=False) # save as a csv
###############################################################################
# use reactimes as features
np.random.seed(12345)
# use all 6 possible features
feature_names = [
                 'RT_correct',
                 'RT_awareness',
                 'RT_confidence']
target_name = 'attention'
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               RT_correct       = [],
               RT_awareness     = [],
               RT_confidence    = [],
               )

for n_back in range(5): # loop through the number of trials looking back
# this is the part that is redundent and the code is long
    results          = classification(
                                      df_sub,
                                      feature_names,
                                      target_name,
                                      results,
                                      participant,
                                      experiment,
                                      window=n_back,
                                      chance = False,
                                      ) 
temp = pd.DataFrame(results)
temp.to_csv(os.path.join(saving_dir,'att_RT_features (experiment score)_{}.csv'.format(participant)),index=False) # save as a csv


































