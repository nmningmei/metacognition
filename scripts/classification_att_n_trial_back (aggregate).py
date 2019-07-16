#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:36:53 2019

@author: nmei
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
saving_dir = '../results/aggregate_experiment_score'
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
# use judgement features
# use judgement features
feature_names = [
                 'correct',
                 'awareness',
                 'confidence',]
target_name = 'attention'
results = dict(
        sub_name = [],
        model_name = [],
        scores_mean = [],
        scores_std = [],)
new_names = [f"{name}_{n_back}" for n_back in np.arange(1,5) for name in feature_names ]
for name in new_names: 
    results[name] = []

for participant in tqdm(['AS', 'BG', 'EU', 'IK', 'JD', 
                         'JZ', 'KK', 'KS', 'OE', 'OS', 
                         'PC','RL', 'SO', 'SP', 'WT', 'YS']):
    
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
    
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    features = []
    targets  = []
    for block,df_block in df_sub.groupby('blocks'):
        df_block['row'] = np.arange(df_block.shape[0])
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        a,b = [],[]
        for n_back in np.arange(1,5):
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the last n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            n_remove = 4 - n_back
            a.append(feature[n_remove:,:])
        
        feature = np.hstack(a)
        features.append(feature)
        targets.append(target)
    features    = np.concatenate(features).astype(int)
    targets     = np.concatenate(targets).astype(int)
    subs        = np.array([participant] * len(targets))
    
    Logisticregression = LogisticRegression(C = 1e10,
                         max_iter               = int(1e2),# default was 100
                         tol                    = 0.0001, # default value
                         solver                 = 'liblinear', # default solver
                         random_state           = 12345)
    Randomforestclassifier  = XGBClassifier(n_estimators         = 500, 
                                            random_state         = 12345,
                                            max_depth            = 3,
                                            objective            = "binary:logistic",
                                            subsample            = 0.9,
                                            n_jobs               = 6,
                                                     )
    cv = StratifiedShuffleSplit(n_splits = 100,
                                test_size = 0.2,
                                random_state = 12345)
    idxs_train,idxs_test = [],[]
    for idx_train,idx_test in cv.split(features,targets):
        idxs_train.append(idx_train)
        idxs_test.append(idx_test)
        
    for clf,clf_name in zip([Logisticregression,Randomforestclassifier],
                            ['LogisticRegression','RandomForest']):
        print(clf_name)
        res = cross_validate(clf,
                             features,
                             targets,
                             cv = cv,
                             scoring='roc_auc',
                             return_estimator = True)
        
        preds = np.array([clf.predict_proba(features[idx_test])[:,-1] for clf,idx_test in zip(res['estimator'],idxs_test)])
        from sklearn import metrics, preprocessing
        scores = np.array([metrics.roc_auc_score(targets[idx_test],
                                                pred) for pred,idx_test in zip(
                                                        preds,idxs_test)])
        if clf_name == "RandomForest":
            feature_importances = np.array([clf.feature_importances_ for idx_train,clf in zip(idxs_train,res['estimator'])])
            feature_importances = preprocessing.scale(feature_importances,axis = 1)
            temp = {name:fi for name,fi in zip(new_names,feature_importances.mean(0))}
        elif clf_name == "LogisticRegression":
            coefs = np.array([clf.coef_[0] for clf in res['estimator']])
            temp = {name:cf for name,cf in zip(new_names,coefs.mean(0))}
        
        results['sub_name'].append(participant)
        results['model_name'].append(clf_name)
        results['scores_mean'].append(scores.mean())
        results['scores_std'].append(scores.std())
        for name in new_names:
            results[name].append(temp[name])
        
results = pd.DataFrame(results)
results.to_csv(os.path.join(saving_dir,
                            f'aggregate {experiment}.csv'),
    index = False)


















