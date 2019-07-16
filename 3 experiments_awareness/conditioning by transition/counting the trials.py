#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:25:01 2018

@author: nmei
"""

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import utils
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('poster')
    
    figure_dir = 'figures'
    results_dir = 'results'
    for dirs in [figure_dir,results_dir]:
        if not os.path.exists(dirs):
            os.mkdir(dirs)
    results_to_save = []
    n_bootstrap = 1000
    alpha = 0.5
    fig,axes = plt.subplots(figsize=(20,20),nrows=3,ncols=2)
    
    try:# the subject level processing
        df1         = pd.read_csv('e1.csv').iloc[:,1:]
    except: # when I test the script
        df1         = pd.read_csv('../e1.csv').iloc[:,1:]
    df              = df1.copy()
    # select the columns that I need
    df              = df[['blocks.thisN',
                          'trials.thisN',
                          'key_resp_2.keys',
                          'resp.corr',
                          'resp_mrating.keys',
                          'participant',]]
    # rename the columns
    df.columns      = ['blocks',
                       'trials',
                       'awareness',
                       'correctness',
                       'confidence',
                       'participant',]
    # preallocate the data frame structure
    results         = dict(sub              = [],
                           model            = [],
                           score            = [],
                           window           = [],
                           correctness      = [],
                           awareness        = [],
                           confidence       = [],
                           chance           = [],
                           )
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'awareness'
    experiment      = 'e1'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    results = {'1-1':[],
               '1-2':[],
               '2-2':[],
               '2-1':[]}
    # ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']
    # get one of the participants' data
    for participant in ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']:
        df_sub          = df[df['participant'] == participant].dropna()
        results_sub = {'1-1':{'awareness':[],'correctness':[]},
                       '1-2':{'awareness':[],'correctness':[]},
                       '2-2':{'awareness':[],'correctness':[]},
                       '2-1':{'awareness':[],'correctness':[]},}
        for block, df_ in df_sub.groupby(['blocks']):
            current_trial = df_[['awareness','correctness']].shift(1).dropna().values
            previous_trial = df_['awareness'].shift(-1).dropna().values
            awareness_trials = np.vstack([current_trial[:,0],previous_trial]).T
            correctness = current_trial[:,1]
            for trial,correct in zip(awareness_trials,correctness):
                trial,correct
                trial_type = '{}-{}'.format(int(trial[0]),int(trial[1]))
                results_sub[trial_type]['awareness'].append(trial)
                results_sub[trial_type]['correctness'].append(correct)
        for trial_type in results.keys():
            results[trial_type].append(float(np.sum(results_sub[trial_type]['correctness'])) / len(results_sub[trial_type]['correctness']))
    results = pd.DataFrame(results)
    ax = axes[0][0]
    ps_1 = utils.resample_ttest_2sample(results['2-1'].values,
                                        results['1-1'].values,
                                        n_ps = 100,
                                        n_permutation = 10000,
                                        one_tail = False,
                                        match_sample_size = True)
    resample_21 = utils.bootstrap_resample(results['2-1'].values,n = None)
    resample_11 = utils.bootstrap_resample(results['1-1'].values,n = None)
    ax.hist(results['2-1'].values,label='2-1',color='red',density=True,alpha=alpha)
    ax.hist(results['1-1'],label='1-1',color='blue',density=True,alpha=alpha)
    ax.legend()
    title = 'experiment {}, p = {:.5}'.format(experiment,ps_1.mean())
    ax.set(ylabel='count',xlabel='Accuracy',title=title)
    ax = axes[0][1]
    ps_2 = utils.resample_ttest_2sample(results['2-2'].values,
                                        results['1-2'].values,
                                        n_ps = 100,
                                        n_permutation = 10000,
                                        one_tail = False,
                                        match_sample_size = True)
    resample_12 = utils.bootstrap_resample(results['1-2'].values,n = None)
    resample_22 = utils.bootstrap_resample(results['2-2'].values,n = None)
    ax.hist(results['1-2'].values,label='1-2',color='red',density=True,alpha=alpha)
    ax.hist(results['2-2'].values,label='2-2',color='blue',density=True,alpha=alpha)
    ax.legend()
    title = 'experiment {}, p = {:.5}'.format(experiment, ps_2.mean())
    ax.set(ylabel='count',xlabel='Accuracy',title=title)
    #############################################################################################
    #############################################################################################
    try:# the subject level processing
        df1         = pd.read_csv('e2.csv').iloc[:,1:]
    except: # when I test the script
        df1         = pd.read_csv('../e2.csv').iloc[:,1:]
    df              = df1.copy()
    # select the columns that I need
    df              = df[['blocks.thisN',
                          'trials.thisN',
                          'key_resp_2.keys',
                          'resp.corr',
                          'resp_mrating.keys',
                          'participant',]]
    # rename the columns
    df.columns      = ['blocks',
                       'trials',
                       'awareness',
                       'correctness',
                       'confidence',
                       'participant',]
    # preallocate the data frame structure
    results         = dict(sub              = [],
                           model            = [],
                           score            = [],
                           window           = [],
                           correctness      = [],
                           awareness        = [],
                           confidence       = [],
                           chance           = [],
                           )
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'awareness'
    experiment      = 'e2'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    # 'ack', 'cc', 'ck', 'cpj', 'em', 'es', 'fd', 'jmac', 'lidia', 'ls','mimi', 'pr', 'pss', 'sva', 'tj'
    results = {'1-1':[],
               '1-2':[],
               '2-2':[],
               '2-1':[]}
    # ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']
    # get one of the participants' data
    for participant in ['ack', 'cc', 'ck', 'cpj', 'em', 'es', 'fd', 'jmac', 'lidia', 'ls','mimi', 'pr', 'pss', 'sva', 'tj']:
        df_sub          = df[df['participant'] == participant].dropna()
        df_sub          = df_sub[df_sub['awareness'] != 3]
        results_sub = {'1-1':{'awareness':[],'correctness':[]},
                       '1-2':{'awareness':[],'correctness':[]},
                       '2-2':{'awareness':[],'correctness':[]},
                       '2-1':{'awareness':[],'correctness':[]},}
        for block, df_ in df_sub.groupby(['blocks']):
            current_trial = df_[['awareness','correctness']].shift(1).dropna().values
            previous_trial = df_['awareness'].shift(-1).dropna().values
            awareness_trials = np.vstack([current_trial[:,0],previous_trial]).T
            correctness = current_trial[:,1]
            for trial,correct in zip(awareness_trials,correctness):
                trial,correct
                trial_type = '{}-{}'.format(int(trial[0]),int(trial[1]))
                results_sub[trial_type]['awareness'].append(trial)
                results_sub[trial_type]['correctness'].append(correct)
        for trial_type in results.keys():
            results[trial_type].append(float(np.sum(results_sub[trial_type]['correctness'])) / len(results_sub[trial_type]['correctness']))
    results = pd.DataFrame(results)
    ax = axes[1][0]
    ps_1 = utils.resample_ttest_2sample(results['2-1'].values,
                                        results['1-1'].values,
                                        n_ps = 100,
                                        n_permutation = 10000,
                                        one_tail = False,
                                        match_sample_size = True)
    resample_21 = utils.bootstrap_resample(results['2-1'].values,n = None)
    resample_11 = utils.bootstrap_resample(results['1-1'].values,n = None)
    ax.hist(results['2-1'].values,label='2-1',color='red',density=True,alpha=alpha)
    ax.hist(results['1-1'],label='1-1',color='blue',density=True,alpha=alpha)
    ax.legend()
    title = 'experiment {}, p = {:.5}'.format(experiment,ps_1.mean())
    ax.set(ylabel='count',xlabel='Accuracy',title=title)
    ax = axes[1][1]
    ps_2 = utils.resample_ttest_2sample(results['2-2'].values,
                                        results['1-2'].values,
                                        n_ps = 100,
                                        n_permutation = 10000,
                                        one_tail = False,
                                        match_sample_size = True)
    resample_12 = utils.bootstrap_resample(results['1-2'].values,n = None)
    resample_22 = utils.bootstrap_resample(results['2-2'].values,n = None)
    ax.hist(results['1-2'].values,label='1-2',color='red',density=True,alpha=alpha)
    ax.hist(results['2-2'].values,label='2-2',color='blue',density=True,alpha=alpha)
    ax.legend()
    title = 'experiment {}, p = {:.5}'.format(experiment,ps_2.mean())
    ax.set(ylabel='count',xlabel='Accuracy',title=title)
    #########################################################################################
    #########################################################################################
    #########################################################################################
    try:# the subject level processing
        df1         = pd.read_csv('e3.csv').iloc[:,1:]
    except: # when I test the script
        df1         = pd.read_csv('../e3.csv').iloc[:,1:]
    df              = df1.copy()
    # select the columns that I need
    df              = df[['blocks.thisN',
                          'trials.thisN',
                          'key_resp_2.keys',
                          'resp.corr',
                          'resp_mrating.keys',
                          'participant',]]
    # rename the columns
    df.columns      = ['blocks',
                       'trials',
                       'awareness',
                       'correctness',
                       'confidence',
                       'participant',]
    # preallocate the data frame structure
    results         = dict(sub              = [],
                           model            = [],
                           score            = [],
                           window           = [],
                           correctness      = [],
                           awareness        = [],
                           confidence       = [],
                           chance           = [],
                           )
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'awareness'
    experiment      = 'e3'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    # ['ab', 'eb', 'er', 'hgh', 'kb', 'kj', 'mp', 'rb', 'vs', 'wp']
    # get one of the participants' data
    results = {'1-1':[],
               '1-2':[],
               '2-2':[],
               '2-1':[]}
    # get one of the participants' data
    for participant in ['ab', 'eb', 'er', 'hgh', 'kb', 'kj', 'mp', 'rb', 'vs', 'wp']:
        df_sub          = df[df['participant'] == participant].dropna()
        results_sub = {'1-1':{'awareness':[],'correctness':[]},
                       '1-2':{'awareness':[],'correctness':[]},
                       '2-2':{'awareness':[],'correctness':[]},
                       '2-1':{'awareness':[],'correctness':[]},}
        for block, df_ in df_sub.groupby(['blocks']):
            current_trial = df_[['awareness','correctness']].shift(1).dropna().values
            previous_trial = df_['awareness'].shift(-1).dropna().values
            awareness_trials = np.vstack([current_trial[:,0],previous_trial]).T
            correctness = current_trial[:,1]
            for trial,correct in zip(awareness_trials,correctness):
                trial,correct
                trial_type = '{}-{}'.format(int(trial[0]),int(trial[1]))
                results_sub[trial_type]['awareness'].append(trial)
                results_sub[trial_type]['correctness'].append(correct)
        for trial_type in results.keys():
            results[trial_type].append(float(np.sum(results_sub[trial_type]['correctness'])) / len(results_sub[trial_type]['correctness']))
    results = pd.DataFrame(results)
    ax = axes[2][0]
    ps_1 = utils.resample_ttest_2sample(results['2-1'].values,
                                        results['1-1'].values,
                                        n_ps = 100,
                                        n_permutation = 10000,
                                        one_tail = False,
                                        match_sample_size = True)
    resample_21 = utils.bootstrap_resample(results['2-1'].values,n = None)
    resample_11 = utils.bootstrap_resample(results['1-1'].values,n = None)
    ax.hist(results['2-1'].values,label='2-1',color='red',density=True,alpha=alpha)
    ax.hist(results['1-1'],label='1-1',color='blue',density=True,alpha=alpha)
    ax.legend()
    title = 'experiment {}, p = {:.5}'.format(experiment,ps_1.mean())
    ax.set(ylabel='count',xlabel='Accuracy',title=title)
    ax = axes[2][1]
    ps_2 = utils.resample_ttest_2sample(results['2-2'].values,
                                        results['1-2'].values,
                                        n_ps = 100,
                                        n_permutation = 10000,
                                        one_tail = False,
                                        match_sample_size = True)
    resample_12 = utils.bootstrap_resample(results['1-2'].values,n = None)
    resample_22 = utils.bootstrap_resample(results['2-2'].values,n = None)
    ax.hist(results['1-2'].values,label='1-2',color='red',density=True,alpha=alpha)
    ax.hist(results['2-2'].values,label='2-2',color='blue',density=True,alpha=alpha)
    ax.legend()
    title = 'experiment {}, p = {:.5}'.format(experiment,ps_2.mean())
    ax.set(ylabel='count',xlabel='Accuracy',title=title)
    
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir,'accuracy as a function of consective trial awareness ratings.png'),
                dpi = 450,
                bbox_inches = 'tight')



















