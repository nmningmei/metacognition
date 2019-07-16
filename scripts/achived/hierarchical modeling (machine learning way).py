#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:51:47 2018

@author: nmei

require: pymc3
"""

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score,make_scorer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit
import os
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
results=dict(r2_mean=[],
         r2_std=[],
         sub=[],
         feature=[],
         window=[],
         weight=[],
         variation=[],
         )


for n_back in range(5): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
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
        features,targets = shuffle(features,targets)

        df_working = pd.DataFrame(np.hstack([features,targets]),)
        df_working.columns=['correct','awareness','confidence','attention']
        for name in df_working.columns[:-1]:
            X = df_working[name].values
            y = df_working['attention'].values
            scorer = make_scorer(explained_variance_score,greater_is_better=False)
            cv = StratifiedShuffleSplit(100,test_size=0.2,random_state=12345)
            clf = LogisticRegressionCV(Cs=np.logspace(-4,4,9),random_state=12345)
            scores = cross_val_score(clf,
                                     X.ravel().reshape(-1,1),
                                     y.ravel(),
                                     cv=cv,
                                     scoring=scorer,
                                     )
            clf.fit(X.ravel().reshape(-1,1),
                    y.ravel())
            print('{},{},{},{},{:.2f}'.format(experiment,participant,n_back,name,np.mean(scores)))
            results['r2_mean'].append(np.mean(scores))
            results['r2_std'].append(np.std(scores))
            results['weight'].append(clf.coef_[0][0])
            results['feature'].append(name)
            results['sub'].append(participant)
            results['window'].append(n_back)
            results['variation'].append('univariate')
        X = df_working.values[:,:-1]
        y = df_working['attention'].values
        scorer = make_scorer(explained_variance_score,greater_is_better=False)
        cv = StratifiedShuffleSplit(100,test_size=0.2,random_state=12345)
        clf = LogisticRegressionCV(Cs=np.logspace(-4,4,9),random_state=12345)
        scores = cross_val_score(clf,
                                 X,
                                 y,
                                 cv=cv,
                                 scoring=scorer,
                                 )
        clf.fit(X,y)
        print('{},{},{},{:.2f}'.format(participant,n_back,'full',np.mean(scores)))
        coefs = clf.coef_#normalize(clf.coef_)
        for name,c in zip(['correct','awareness','confidence'],coefs[0]):
            results['r2_mean'].append(np.mean(scores))
            results['r2_std'].append(np.std(scores))
            results['weight'].append(c)
            results['feature'].append('{}'.format(name))
            results['sub'].append(participant)
            results['window'].append(n_back)
            results['variation'].append('multivariate')
            
            
results_to_save = pd.DataFrame(results)
att=results_to_save.copy()
att.to_csv(os.path.join(results_dir,'att_uni_multi.csv'),index=False)


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
results=dict(r2_mean=[],
         r2_std=[],
         sub=[],
         feature=[],
         window=[],
         weight=[],
         variation=[]
         )


for n_back in range(5): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success' ] = df_sub.loc[:,'success'   ].values - 1
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
        features,targets = shuffle(features,targets)

        df_working = pd.DataFrame(np.hstack([features,targets]),)
        df_working.columns=['correct','awareness','confidence','success']
        for name in df_working.columns[:-1]:
            X = df_working[name].values
            y = df_working['success'].values
            scorer = make_scorer(explained_variance_score,greater_is_better=False)
            cv = StratifiedShuffleSplit(100,test_size=0.2,random_state=12345)
            clf = LogisticRegressionCV(Cs=np.logspace(-4,4,9),random_state=12345)
            scores = cross_val_score(clf,
                                     X.ravel().reshape(-1,1),
                                     y.ravel(),
                                     cv=cv,
                                     scoring=scorer,
                                     )
            clf.fit(X.ravel().reshape(-1,1),
                    y.ravel())
            print('{},{},{},{},{:.2f}'.format(experiment,participant,n_back,name,np.mean(scores)))
            results['r2_mean'].append(np.mean(scores))
            results['r2_std'].append(np.std(scores))
            results['weight'].append(clf.coef_[0][0])
            results['feature'].append(name)
            results['sub'].append(participant)
            results['window'].append(n_back)
            results['variation'].append('univariate')
        X = df_working.values[:,:-1]
        y = df_working['success'].values
        scorer = make_scorer(explained_variance_score,greater_is_better=False)
        cv = StratifiedShuffleSplit(100,test_size=0.2,random_state=12345)
        clf = LogisticRegressionCV(Cs=np.logspace(-4,4,9),random_state=12345)
        scores = cross_val_score(clf,
                                 X,
                                 y,
                                 cv=cv,
                                 scoring=scorer,
                                 )
        clf.fit(X,y)
        print('{},{},{},{:.2f}'.format(participant,n_back,'full',np.mean(scores)))
        coefs = clf.coef_#normalize(clf.coef_)
        for name,c in zip(['correct','awareness','confidence'],coefs[0]):
            results['r2_mean'].append(np.mean(scores))
            results['r2_std'].append(np.std(scores))
            results['weight'].append(c)
            results['feature'].append('{}'.format(name))
            results['sub'].append(participant)
            results['window'].append(n_back)
            results['variation'].append('multivariate')
            
results_to_save = pd.DataFrame(results)
pos = results_to_save.copy()
pos.to_csv(os.path.join(results_dir,'pos_uni_multi.csv'),index=False)






































