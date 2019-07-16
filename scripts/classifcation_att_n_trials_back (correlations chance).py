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
from scipy import stats
from itertools import combinations
from sklearn.utils import shuffle
saving_dir = '../results/correlation_chance'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

# Exp 2
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
for participant in ['AS', 'BG', 'EU', 'IK', 'JD', 'JZ', 'KK', 'KS', 'OE', 'OS', 'PC','RL', 'SO', 'SP', 'WT', 'YS']:
    
    
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
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
                   correlation      = [],
                   pvals            = [],
                   feature_name_1   = [],
                   feature_name_2   = [],
                   )
    for _ in range(500):
        for name1,name2 in combinations(feature_names,2):
            corr,pval = stats.pointbiserialr(df_sub[name1].values,
                                             shuffle(df_sub[name2].values))
            results['sub'].append(participant)
            results['correlation'].append(corr)
            results['pvals'].append(pval)
            results['feature_name_1'].append(name1)
            results['feature_name_2'].append(name2)
    
    temp = pd.DataFrame(results)
    temp.to_csv(os.path.join(saving_dir,'att_3_1_features (correlation)_{}.csv'.format(participant)),index=False) # save as a csv
    

































