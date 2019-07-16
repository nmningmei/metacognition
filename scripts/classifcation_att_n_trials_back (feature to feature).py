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
from utils import (classification_simple_logistic)
from itertools import permutations
saving_dir = '../results/simple_regression_11'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)


# Exp 2
for participant in ['AS', 'BG', 'EU', 'IK', 'JD', 'JZ', 'KK', 'KS', 'OE', 'OS', 'PC','RL', 'SO', 'SP', 'WT', 'YS']:
    experiment = 'att'
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
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
    ##################################################################
    ##################################################################
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    feature_sets = ['awareness','confidence','correct']
    for a,b in permutations(feature_sets,2):
        print(a,b)
        feature_names = [a]
        target_name = b
        results = dict(sub              = [],
                       model            = [],
                       score            = [],
                       window           = [],
                       feature_name     = [],
                       target_name      = [],
                       chance           = [],
                       )
        
        for n_back in range(5): # loop through the number of trials looking back
        # this is the part that is redundent and the code is long
            results          = classification_simple_logistic(
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
        temp.to_csv(os.path.join(saving_dir,f'att_3_1_features_{participant}_{a}_{b}.csv'),index=False) # save as a csv
###############################################################################
# Exp 2
for participant in ['AS', 'BG', 'EU', 'IK', 'JD', 'JZ', 'KK', 'KS', 'OE', 'OS', 'PC','RL', 'SO', 'SP', 'WT', 'YS']:
    experiment = 'att'
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
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
    ##################################################################
    ##################################################################
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    feature_sets = ['awareness','confidence','correct']
    for a,b in permutations(feature_sets,2):
        print(a,b)
        feature_names = [a]
        target_name = b
        results = dict(sub              = [],
                       model            = [],
                       score            = [],
                       window           = [],
                       feature_name     = [],
                       target_name      = [],
                       chance           = [],
                       )
        
        for n_back in range(5): # loop through the number of trials looking back
        # this is the part that is redundent and the code is long
            results          = classification_simple_logistic(
                                              df_sub,
                                              feature_names,
                                              target_name,
                                              results,
                                              participant,
                                              experiment,
                                              window=n_back,
                                              chance = True,
                                              ) 
        temp = pd.DataFrame(results)
        temp.to_csv(os.path.join(saving_dir,f'att_3_1_features_{participant}_{a}_{b} (chance).csv'),index=False) # save as a csv


































