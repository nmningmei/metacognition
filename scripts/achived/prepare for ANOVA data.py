#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:21:32 2018

@author: nmei
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
saving_dir = '../data/ANOVA'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

experiment = 'pos'
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
results = dict(subject = [],
               window = [],
               experiment = [],
               JoAwareness_0 = [],
               JoAwareness_1 = [],
               JoConfidence_0 = [],
               JoConfidence_1 = [],
               JoCorrectness_0 = [],
               JoCorrectness_1 = [],)
for n_back in range(4): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        feature_names   = [
                         'correct',
                         'awareness',
                         'confidence',
                         'RT_correct',
                         'RT_awareness',
                         'RT_confidence']
        target_name     = 'success'
        # the shifting is done within each block of a subject
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        temp              = pd.DataFrame(features,columns = feature_names)
        temp['targets']   = targets
        # separate the 2 values of POS
        temp_0            = temp[temp['targets'] == 0]
        temp_1            = temp[temp['targets'] == 1]
        results['subject'].append(participant)
        results['window'].append(n_back)
        results['JoAwareness_0'].append(temp_0['RT_awareness'].median())
        results['JoAwareness_1'].append(temp_1['RT_awareness'].median())
        results['JoConfidence_0'].append(temp_0['RT_confidence'].median())
        results['JoConfidence_1'].append(temp_1['RT_confidence'].median())
        results['JoCorrectness_0'].append(temp_0['RT_correct'].median())
        results['JoCorrectness_1'].append(temp_1['RT_correct'].median())
        results['experiment'].append(experiment)
results = pd.DataFrame(results)
writer = pd.ExcelWriter(os.path.join(saving_dir,'{}.xlsx'.format(experiment)),engine = 'xlsxwriter')
for n_back in range(4):
    results[results['window'] == n_back].to_excel(writer,sheet_name='{} trials back'.format(n_back),
           index=False)
writer.save()

#############################################################################################################
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
results = dict(subject = [],
               window = [],
               experiment = [],
               JoAwareness_0 = [],
               JoAwareness_1 = [],
               JoConfidence_0 = [],
               JoConfidence_1 = [],
               JoCorrectness_0 = [],
               JoCorrectness_1 = [],)
for n_back in range(4): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'attention'   ] = df_sub.loc[:,'attention'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        feature_names   = [
                         'correct',
                         'awareness',
                         'confidence',
                         'RT_correct',
                         'RT_awareness',
                         'RT_confidence']
        target_name     = 'attention'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift upward so that the last n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting downward 
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        temp              = pd.DataFrame(features,columns = feature_names)
        temp['targets']   = targets
        
        temp_0            = temp[temp['targets'] == 0]
        temp_1            = temp[temp['targets'] == 1]
        results['subject'].append(participant)
        results['window'].append(n_back)
        results['JoAwareness_0'].append(temp_0['RT_awareness'].median())
        results['JoAwareness_1'].append(temp_1['RT_awareness'].median())
        results['JoConfidence_0'].append(temp_0['RT_confidence'].median())
        results['JoConfidence_1'].append(temp_1['RT_confidence'].median())
        results['JoCorrectness_0'].append(temp_0['RT_correct'].median())
        results['JoCorrectness_1'].append(temp_1['RT_correct'].median())
        results['experiment'].append(experiment)
results = pd.DataFrame(results)
writer = pd.ExcelWriter(os.path.join(saving_dir,'{}.xlsx'.format(experiment)),engine = 'xlsxwriter')
for n_back in range(4):
    results[results['window'] == n_back].to_excel(writer,sheet_name='{} trials back'.format(n_back),index=False)
writer.save()














































