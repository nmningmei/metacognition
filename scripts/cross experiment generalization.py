#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:51:46 2018

@author: nmei
"""

import os
working_dir = '../data/'
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
saving_dir = '../results/cross_experiment_generalization'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
from utils import (get_features_targets_groups,
                   make_clfs)
# Exp 1
#['AC', 'CL', 'FW', 'HB', 'KK', 'LM', 'MC', 'MP1', 'MP2', 'NN', 'RP','SD', 'TJ', 'TS', 'WT']
# Exp 2
#['AS', 'BG', 'EU', 'IK', 'JD', 'JZ', 'KK', 'KS', 'OE', 'OS', 'PC','RL', 'SO', 'SP', 'WT', 'YS']

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
np.random.seed(12345)
results = dict(
        model       = [],
        train       = [],
        test        = [],
        score_mean  = [],
        score_std   = [],
        pval        = [],
        window      = []
        )
results_full = dict(
        model       = [],
        train       = [],
        test        = [],
        score       = [],
        fold        = [],
        window      = []
        )
for n_back in range(5): # loop through the number of N-back trials
    # get the features, targets, and subject groups for both experiments and the given n_back trial
    X_att,y_att,groups_att              = get_features_targets_groups(
            att,# the loaded dataframe
            n_back                      = n_back, # n_back trials
            names                       = ['attention',# need to normalize to 0 and 1
                                           'awareness',# need to normalize to 0 and 1
                                           'confidence'],# need to normalize to 0 and 1
            independent_variables       = ['correct',
                                           'awareness',
                                           'confidence'],
            dependent_variable          =  'attention'
            )
    X_pos,y_pos,groups_pos              = get_features_targets_groups(
            pos,
            n_back                      = n_back,
            names                       = ['success',# need to normalize to 0 and 1
                                           'awareness',# need to normalize to 0 and 1
                                           'confidence'],# need to normalize to 0 and 1
            independent_variables       = ['correct',
                                           'awareness',
                                           'confidence'],
            dependent_variable          =  'success'
            )
    ##################################################################################
    ###################### after we prepare the train-test data ######################
    ###################### we are ready to cross experiment validation ###############
    # train at pos and test at att - n_cv = 100
    n_cv = 100 # number of cross validation
    pr   = 0.7 # selected proportion of the data 
    # select subset of the traiing data and the test data to estimate the variance
    # of the cross validation
    
    # select a proportion of the training data
    idxs_train = [np.random.choice(len(X_pos),
                                   size     = int(pr*len(X_pos)),
                                   replace  = False
                                   ) for ii in range(n_cv)]
    # select a proportion of the test data
    idxs_test  = [np.random.choice(len(X_att),
                                   size     = int(pr*len(X_att)),
                                   replace  = False
                                   ) for ii in range(n_cv)]
    # for 2 models, we will perform the cross experiment validation
    for model_name in make_clfs().keys():
        scores              = []
        permutation_scores  = []
        n_permutations      = 2000
        for fold,(idx_train,idx_test) in tqdm(enumerate(zip(idxs_train,idxs_test)),
                  desc='cv-{}'.format(model_name)):
            # initialize the classifier - LG or RF
            clf     = make_clfs()[model_name]
            X_train = X_pos[idx_train]# get the training features
            y_train = y_pos[idx_train]# get the training targets
            
            X_test  = X_att[idx_test ]# get the testing features
            y_test  = y_att[idx_test ]# get the testing targets
            
            clf.fit(X_train,y_train)
            preds   = clf.predict_proba(X_test)
            score   = roc_auc_score(y_test,preds[:,-1])
            permutation_scores_ = []
            # estimate the chance 
            # to save time, instead of re-order the training features and targets, training the decoder,
            # we re-order the testing features and targets, making the testing data uninformative
            # if the pattern we learn from the perfectly ordered data still performs beyond chance in a 
            # randomly ordered data, that means the pattern we learned is randomization patterns, 
            # otherwise, the pattern we learned is different from randomization patterns.
            for n_permutation in range(n_permutations):
                y_test_randome = shuffle(y_test)
                permutation_scores_.append(roc_auc_score(
                        y_test_randome,preds[:,-1]
                        ))
            scores.append(score)
            permutation_scores.append(permutation_scores_)
            
            results_full['model'    ].append(model_name)
            results_full['score'    ].append(score)
            results_full['window'   ].append(n_back)
            results_full['fold'     ].append(fold+1)
            results_full['train'    ].append('POS')
            results_full['test'     ].append('ATT')
            
        permutation_scores  = np.array(permutation_scores)
        scores              = np.array(scores)
        # save the results
        results['model'     ].append(model_name)
        results['score_mean'].append(scores.mean())
        results['score_std' ].append(scores.std())
        results['train'     ].append('POS')
        results['test'      ].append('ATT')
        results['window'    ].append(n_back)
        pval = (np.sum(permutation_scores.mean(0) >= scores.mean()) + 1.0) / (n_permutations + 1)
        results['pval'      ].append(pval.mean())
        print('att,window {},model {},scores = {:.3f}+/-{:.3f},p = {:.4f}'.format(
                n_back,model_name,
                scores.mean(),scores.std(),pval.mean()))
        
    # train at att and test at pos - n_cv = 100
    idxs_train = [np.random.choice(len(X_att),
                                   size     = int(pr*len(X_att)),
                                   replace  = False
                                   ) for ii in range(n_cv)]
    idxs_test  = [np.random.choice(len(X_pos),
                                   size     = int(pr*len(X_pos)),
                                   replace  = False
                                   ) for ii in range(n_cv)]
    # 
    for model_name in make_clfs().keys():
#        print('cv - {}'.format(model_name))
        scores              = []
        permutation_scores  = []
        n_permutations      = 2000
        for fold,(idx_train,idx_test) in tqdm(enumerate(zip(idxs_train,idxs_test)),desc='cv-{}'.format(model_name)):
            clf     = make_clfs()[model_name]
            X_train = X_att[idx_train]
            y_train = y_att[idx_train]
            
            X_test  = X_pos[idx_test ]
            y_test  = y_pos[idx_test ]
            
            clf.fit(X_train,y_train)
            preds   = clf.predict_proba(X_test)
            score   = roc_auc_score(y_test,preds[:,-1])
            permutation_scores_ = []
            for n_permutation in range(n_permutations):
                y_test_randome = shuffle(y_test)
                permutation_scores_.append(roc_auc_score(
                        y_test_randome,preds[:,-1]
                        ))
            scores.append(score)
            permutation_scores.append(permutation_scores_)
            
            results_full['model'    ].append(model_name)
            results_full['score'    ].append(score)
            results_full['window'   ].append(n_back)
            results_full['fold'     ].append(fold+1)
            results_full['train'    ].append('ATT')
            results_full['test'     ].append('POS')
            
        permutation_scores = np.array(permutation_scores)
        scores      = np.array(scores)
        # save the results
        results['model'     ].append(model_name)
        results['score_mean'].append(scores.mean())
        results['score_std' ].append(scores.std())
        results['train'     ].append('ATT')
        results['test'      ].append('POS')
        results['window'    ].append(n_back)
        pval = (np.sum(permutation_scores.mean(0) >= scores.mean()) + 1.0) / (n_permutations + 1)
        results['pval'      ].append(pval.mean())
        print('pos,window {},model {},scores = {:.3f}+/-{:.3f},p = {:.4f}'.format(
                n_back,model_name,
                scores.mean(),scores.std(),pval.mean()))
result_to_save = pd.DataFrame(results_full)
result_to_save.to_csv(os.path.join(saving_dir,
                                   'cross experiment generalization (folds).csv'),
                      index=False)
df = pd.DataFrame(results)
df.to_csv(os.path.join(saving_dir,
                       'cross experiment generalization.csv'),
          index=False)

import os
from utils import (MCPConverter)
import pandas as pd
saving_dir          = '../results/cross_experiment_generalization'
df                  = pd.read_csv(os.path.join(saving_dir,
                                               'cross experiment generalization.csv'))
df_corrected        = []
for (model,exp_train),df_sub in df.groupby(['model','train']):
    idx_sort        = np.argsort(df_sub.pval.values)
    df_sub          = df_sub.iloc[idx_sort,:]
    pvals           = df_sub.pval.values
    converter       = MCPConverter(pvals = pvals)
    d               = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    df_corrected.append(df_sub)
df_corrected        = pd.concat(df_corrected)
df_corrected.to_csv(os.path.join(saving_dir,
                                 'cross experimnet validation post test.csv'),
                    index=False)



































