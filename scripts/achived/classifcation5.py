#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:31:08 2018

@author: nmei

Cross experiment validation


"""

import os
working_dir = '../data/'
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from utils import (MCPConverter,
                   get_features_targets_groups,
                   make_clfs,
                   resample_ttest)

graph_dir = os.path.join(working_dir,'graph')
dot_dir = os.path.join(working_dir,'dot')
save_dir = '../results/'
if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)
if not os.path.exists(dot_dir):
    os.mkdir(dot_dir)

###########################  load the data of Exp 1 and Exp 2 #######################
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
###########################################  data is ready ######################
###########################################  initialization #####################
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
for n_back in range(11): # loop through the number of trials looking back
    # get the features, targets, and subject groups for Exp 2 and the given n_back trial
    X_att,y_att,groups_att = get_features_targets_groups(
            att,# the loaded dataframe
            n_back = n_back, # n_back trials
            names = ['attention',# need to normalize to 0 and 1
                     'awareness',# need to normalize to 0 and 1
                     'confidence'],# need to normalize to 0 and 1
            independent_variables = ['attention',
                                     'awareness',
                                     'confidence'],
            dependent_variable = 'correct'
            )
    X_pos,y_pos,groups_pos = get_features_targets_groups(
            pos,
            n_back = n_back,
            names = ['success',# need to normalize to 0 and 1
                     'awareness',# need to normalize to 0 and 1
                     'confidence'],# need to normalize to 0 and 1
            independent_variables = ['success',
                                     'awareness',
                                     'confidence'],
            dependent_variable = 'correct'
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
    # select 80% of the test data
    idxs_test  = [np.random.choice(len(X_att),
                                   size     = int(pr*len(X_att)),
                                   replace  = False
                                   ) for ii in range(n_cv)]
    # for 2 models, we will perform the cross experiment validation
    for model_name in make_clfs().keys():
        scores = []
        permutation_scores = []
        n_permutations = 2000
        for idx_train,idx_test in tqdm(zip(idxs_train,idxs_test),desc='cv-{}'.format(model_name)):
            clf     = make_clfs()[model_name]
            X_train = X_pos[idx_train]
            y_train = y_pos[idx_train]
            
            X_test  = X_att[idx_test ]
            y_test  = y_att[idx_test ]
            
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
        permutation_scores = np.array(permutation_scores)
        scores      = np.array(scores)
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
        scores = []
        permutation_scores = []
        n_permutations = 2000
        for idx_train,idx_test in tqdm(zip(idxs_train,idxs_test),desc='cv-{}'.format(model_name)):
            clf = make_clfs()[model_name]
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

df = pd.DataFrame(results)

df_corrected = []
for (model,exp_train),df_sub in df.groupby(['model','train']):
#    idx_sort = np.argsort(df_sub.pval.values)
#    df_sub = df_sub.iloc[idx_sort,:]
    pvals = df_sub.pval.values
    converter = MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    df_corrected.append(df_sub)
df_corrected = pd.concat(df_corrected)
df_corrected.to_csv(os.path.join(save_dir,'cross experimnet validation.csv'),
                    index=False)

import seaborn as sns
sns.set_context('poster')
sns.set_style('whitegrid')
df_corrected = pd.read_csv(os.path.join('../results/cross experimnet validation.csv'))

g = sns.factorplot(x='window',
                   y='score_mean',
                   hue='model',
                   data=df_corrected,
                   row = 'train',
                   aspect=2,
                   dodge = .1,
                   ci = 99,
                   kind = 'point',
                   )
for ax in g.fig.axes:
    ax.axhline(0.5,linestyle='--',color='black',alpha=0.5)

(g.set_axis_labels('Trials look back','ROC AUC scores'))
g.fig.suptitle('Cross Experiment Validation\nTrain on one and test on the other',y=1.09)
g.savefig('../figures/Cross Experiment Validation Scores.png',
          dpi=400,bbox_inches='tight')

g = sns.factorplot(x='window',
                   y='p_corrected',
                   hue='model',
                   data = df_corrected,
                   row = 'train',
                   aspect = 2,
                   kind = 'bar')






























