#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:59:50 2018

@author: nmei

cross experiment generalization, individual subject level

1. get all the data of experiment A of all subjects
2. for each subject in experiment B, we get the features and labels
3. fit a model in the full data of experiment A
4. test the model in the given subject in experiment B

"""

import os
working_dir = '../data/'
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
saving_dir = '../results/cross_experiment_generalization'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
from utils import (get_features_targets_groups,
                   make_clfs)
# Exp 1
#['AC', 'CL', 'FW', 'HB', 'KK', 'LM', 'MC', 'MP1', 'MP2', 'NN', 'RP','SD', 'TJ', 'TS', 'WT']
# Exp 2
#['AS', 'BG', 'EU', 'IK', 'JD', 'JZ', 'KK', 'KS', 'OE', 'OS', 'PC','RL', 'SO', 'SP', 'WT', 'YS']
n_cv = 100 # number of cross validation
pr   = 0.2 # selected proportion of the test data 
#################################  load the raw data ########################################################
# Exp 1
experiment  = 'pos'
pos         = pd.read_csv(os.path.join(working_dir,'PoSdata.csv'))
pos         = pos[pos.columns[1:]]
# rename columns
pos.columns = ['participant',
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
# Exp 2
experiment  = 'att'
att         = pd.read_csv(os.path.join(working_dir,'ATTfoc.csv'))
att         = att[att.columns[1:]]
# rename columns
att.columns = ['participant',
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

###############################################################################################################
results = dict(
                score               = [],
                participant         = [],
                fold                = [],
                model               = [],
                experiment_train    = [],
                experiment_test     = [],
                window              = [],
                )
for n_back in range(5): # loop through the number of N-back trials
    # get the features, targets, and subject groups for both experiments and the given n_back trial
    X_att,y_att,groups_att          = get_features_targets_groups(
            att,# the loaded dataframe
            n_back                  = n_back, # n_back trials
            names                   = ['attention',# need to normalize to 0 and 1
                                       'awareness',# need to normalize to 0 and 1
                                       'confidence'],# need to normalize to 0 and 1
            independent_variables   = ['correct',
                                       'awareness',
                                       'confidence'],
            dependent_variable      = 'attention'
            )
    X_pos,y_pos,groups_pos          = get_features_targets_groups(
            pos,
            n_back                  = n_back,
            names                   = ['success',# need to normalize to 0 and 1
                                       'awareness',# need to normalize to 0 and 1
                                       'confidence'],# need to normalize to 0 and 1
            independent_variables   = ['correct',
                                       'awareness',
                                       'confidence'],
            dependent_variable      = 'success'
            )
    ##################################################################################
    ################## train on one experiment and test on the individual subjects in
    ################## the other experiment ##########################################
    # train on POS first
    experiment_train                = 'POS' # define the source data
    experiment_test                 = 'ATT' # define the target data
    for participant,df_sub in att.groupby('participant'): # loop through the subjects in ATT as test data
        for model_name in make_clfs().keys(): # loop through the 2 models
            cv                      = StratifiedShuffleSplit(n_splits       = n_cv,
                                                             test_size      = pr,
                                                             random_state   = 12345)
            # in each fold of the cross-validation
            for fold,(train,_ )in enumerate(cv.split(X_pos,y_pos)):
                X_train             = X_pos[train] # pick a proportion of the training/source data
                y_train             = y_pos[train] # pick a proportion of the training/source lables
                
                clf                 = make_clfs()[model_name] # initialize the model
                clf.fit(X_train,y_train) # fit the model
                # prepare the test for a give subject
                X_test,y_test,_     = get_features_targets_groups(
                        df_sub,
                        n_back      = n_back,
                        names       = ['attention',# need to normalize to 0 and 1
                                       'awareness',# need to normalize to 0 and 1
                                       'confidence'],# need to normalize to 0 and 1
                        independent_variables = ['correct',
                                                 'awareness',
                                                 'confidence'],
                        dependent_variable    =  'attention'
                        )
                
                pred                = clf.predict_proba(X_test)[:,-1]
                score               = roc_auc_score(y_test,pred)
                
                results['window'            ].append(n_back)
                results['score'             ].append(score)
                results['participant'       ].append(participant)
                results['fold'              ].append(fold+1)
                results['model'             ].append(model_name)
                results['experiment_train'  ].append(experiment_train)
                results['experiment_test'   ].append(experiment_test)
                print('{},{},{},{},{:.4f}'.format(experiment_test,
                                fold+1,
                                model_name,
                                participant,
                                score))
    results_to_save                 = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(saving_dir,'cross experiment generalization (individual level).csv'),index=False)
    #################################################################################
    #############
    # train on ATT first
    experiment_train                = 'ATT'
    experiment_test                 = 'POS'
    for participant,df_sub in pos.groupby('participant'):
        for model_name in make_clfs().keys():
            cv                      = StratifiedShuffleSplit(n_splits       = n_cv,
                                                             test_size      = pr,
                                                             random_state   = 12345)
            
            for fold, (train,_) in enumerate(cv.split(X_att,y_att)):
                X_train             = X_att[train]
                y_train             = y_att[train]
                
                clf                 = make_clfs()[model_name]
                clf.fit(X_train,y_train)
                
                X_test,y_test,_     = get_features_targets_groups(
                        df_sub,
                        n_back      = n_back,
                        names       = ['success',# need to normalize to 0 and 1
                                       'awareness',# need to normalize to 0 and 1
                                       'confidence'],# need to normalize to 0 and 1
                        independent_variables = ['correct',
                                                 'awareness',
                                                 'confidence'],
                        dependent_variable    =  'success'
                        )
                
                pred                = clf.predict_proba(X_test)[:,-1]
                score               = roc_auc_score(y_test,pred)
                results['window'            ].append(n_back)
                results['score'             ].append(score)
                results['participant'       ].append(participant)
                results['fold'              ].append(fold+1)
                results['model'             ].append(model_name)
                results['experiment_train'  ].append(experiment_train)
                results['experiment_test'   ].append(experiment_test)
                print('{},{},{},{},{:.4f}'.format(experiment_test,
                                fold+1,
                                model_name,
                                participant,
                                score))
    results_to_save                 = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(saving_dir,'cross experiment generalization (individual level).csv'),index=False)
                


























































