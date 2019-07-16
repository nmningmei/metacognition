#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:38:22 2018

@author: nmei
"""

import pandas as pd
import numpy as np
import os
import pickle
from utils import logistic_regression,compute_mse,compute_ppc,compute_r2
from scipy import stats
results_dir = '../results/'
working_dir = '../data/'

experiment = 'att'
df         = pd.read_csv(os.path.join(working_dir,'ATTfoc.csv'))
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
np.random.seed(12345)
results={'feature':[],
         'r_squred':[],
         'beta_mean':[],
         'beta_higher_bound':[],
         'beta_lower_bound':[],
         'window':[],
         'sub':[]}


for n_back in range(3): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        model_save={'trace':[],
                    'model':[],
                    'feature':[],
                    'window':[],
                    'sub':[]}
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        features                   = []
        targets                    = []
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        for block, df_block in df_sub.groupby('blocks'):
            feature = (df_block[[
#                                 'trials',
                                 'correct',
                                 'awareness',
                                 'confidence']].shift(-n_back) # shift upward so that the last n_back rows are gone
                                               .dropna() # since some of rows are gone, so they are nans
                                               .values # I only need the matrix not the data frame
                       )
            target = (df_block[[
#                                'trials',
                                'attention']].shift(n_back) # same thing for the target, but shifting downward 
                                           .dropna()
                                           .values
                     )
            features.append(feature)
            targets.append(target)
        features = np.concatenate(features)
        targets  = np.concatenate(targets)
#        features,targets = shuffle(features,targets)

        df_working = pd.DataFrame(np.hstack([features,targets]),)
        df_working.columns=['correct','awareness','confidence','attention']
        traces,models = logistic_regression(df_working,sample_size=5000)
        ppcs = {name:compute_ppc(trace,m,samples=int(1e4)) for (name,trace),m in zip(traces.items(),models.values())}
        mses = {name:compute_mse(df_working,ppc,'attention') for name,ppc in ppcs.items()}
        r2s = {name:compute_r2(df_working,ppc,'attention') for name,ppc in ppcs.items()}
        for name in df_working.columns[:-1]:
            results['feature'].append(name)
            results['r_squred'].append(np.mean(r2s[name]))
            results['beta_mean'].append(np.mean(traces[name].get_values(name)))
            results['beta_higher_bound'].append(stats.scoreatpercentile(traces[name].get_values(name),97.5))
            results['beta_lower_bound'].append(stats.scoreatpercentile(traces[name].get_values(name),2.5))
            results['window'].append(n_back)
            results['sub'].append(participant)
            
            model_save['trace'].append(traces[name])
            model_save['model'].append(models[name])
            model_save['feature'].append(name)
            model_save['window'].append(n_back)
            model_save['sub'].append(participant)
        pickle.dump(model_save,open(os.path.join(results_dir,'models','attention_logistic_{}.pkl'.format(participant)),'wb'))
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(results_dir,'attention_logistic.csv'))
    
experiment = 'pos'
df         = pd.read_csv(os.path.join(working_dir,'PoSdata.csv'))
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
np.random.seed(12345)
results={'feature':[],
         'r_squred':[],
         'beta_mean':[],
         'beta_higher_bound':[],
         'beta_lower_bound':[],
         'window':[],
         'sub':[]}


for n_back in range(3): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        model_save={'trace':[],
                    'model':[],
                    'feature':[],
                    'window':[],
                    'sub':[]}
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        features                   = []
        targets                    = []
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        for block, df_block in df_sub.groupby('blocks'):
            feature = (df_block[[
#                                 'trials',
                                 'correct',
                                 'awareness',
                                 'confidence']].shift(-n_back) # shift upward so that the last n_back rows are gone
                                               .dropna() # since some of rows are gone, so they are nans
                                               .values # I only need the matrix not the data frame
                       )
            target = (df_block[[
#                                'trials',
                                'success']].shift(n_back) # same thing for the target, but shifting downward 
                                           .dropna()
                                           .values
                     )
            features.append(feature)
            targets.append(target)
        features = np.concatenate(features)
        targets  = np.concatenate(targets)
#        features,targets = shuffle(features,targets)

        df_working = pd.DataFrame(np.hstack([features,targets]),)
        df_working.columns=['correct','awareness','confidence','POS']
        traces,models = logistic_regression(df_working,sample_size=5000)
        ppcs = {name:compute_ppc(trace,m,samples=int(1e4)) for (name,trace),m in zip(traces.items(),models.values())}
        mses = {name:compute_mse(df_working,ppc,'POS') for name,ppc in ppcs.items()}
        r2s = {name:compute_r2(df_working,ppc,'POS') for name,ppc in ppcs.items()}
        for name in df_working.columns[:-1]:
            results['feature'].append(name)
            results['r_squred'].append(np.mean(r2s[name]))
            results['beta_mean'].append(np.mean(traces[name].get_values(name)))
            results['beta_higher_bound'].append(stats.scoreatpercentile(traces[name].get_values(name),97.5))
            results['beta_lower_bound'].append(stats.scoreatpercentile(traces[name].get_values(name),2.5))
            results['window'].append(n_back)
            results['sub'].append(participant)
            
            model_save['trace'].append(traces[name])
            model_save['model'].append(models[name])
            model_save['feature'].append(name)
            model_save['window'].append(n_back)
            model_save['sub'].append(participant)
    pickle.dump(model_save,open(os.path.join(results_dir,'models','pos_logistic_{}.pkl'.format(participant)),'wb'))
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(results_dir,'pos_logistic.csv'))
    





















































