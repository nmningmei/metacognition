#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:00:06 2018

@author: nmei
"""

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import utils
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('poster')
    from matplotlib import pyplot as plt
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from itertools import combinations
    import pymc3 as pm
    import theano.tensor as tt
    # define result saving directory
    dir_saving      = 'consective_transitions'
    if not os.path.exists(dir_saving):
        os.makedirs(dir_saving)
    ############################################################################################
    df1         = pd.read_csv('e1.csv').iloc[:,1:]
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
    # ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']
    # get one of the participants' data
    transition_matrix,sub = [], []
    transition_count = []
    for participant,df_sub in df.groupby('participant'):
        awareness = df_sub['awareness'].values - 1
        with pm.Model() as model:
            a = pm.Beta('a', 0.5, 0.5)
            yl = pm.Bernoulli('yl', a, observed=awareness)
            trace = pm.sample(1000,
                              step=pm.SMC(),
                              random_seed=42)
        # for 1-back
        temp = pd.crosstab(pd.Series(df_sub['awareness'].values[1:],name='N'),
                           pd.Series(df_sub['awareness'].values[:-1],name='N-1'),
                           normalize=1)
        transition_matrix.append(temp.get_values().flatten())
        temp = pd.crosstab(pd.Series(df_sub['awareness'].values[1:],name='N'),
                           pd.Series(df_sub['awareness'].values[:-1],name='N-1'),)
        transition_count.append(temp.get_values().flatten())
        sub.append(participant)
    df1_transition = pd.DataFrame(transition_matrix,columns=['unaware-unaware',
                                                             'aware-unaware',
                                                             'unaware-aware',
                                                             'aware-aware'])
    df1_count = pd.DataFrame(transition_count,columns=['unaware-unaware',
                                                       'aware-unaware',
                                                       'unaware-aware',
                                                       'aware-aware'])
    df1_transition['experiment'] = 1
    df1_transition['sub'] = sub
    df1_count['experiment'] = 1
    df1_count['sub'] = sub
    ##############################################################################
    df2         = pd.read_csv('e2.csv').iloc[:,1:]
    df              = df2.copy()
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
    transition_matrix,sub = [],[]
    transition_count = []
    for participant,df_sub in df.groupby('participant'):
        df_sub          = df_sub[df_sub['awareness'] != 3]
        # for 1-back
        temp = pd.crosstab(pd.Series(df_sub['awareness'].values[1:],name='N'),
                           pd.Series(df_sub['awareness'].values[:-1],name='N-1'),
                           normalize=1)
        transition_matrix.append(temp.get_values().flatten())
        temp = pd.crosstab(pd.Series(df_sub['awareness'].values[1:],name='N'),
                           pd.Series(df_sub['awareness'].values[:-1],name='N-1'),)
        transition_count.append(temp.get_values().flatten())
        sub.append(participant)
    df2_transition = pd.DataFrame(transition_matrix,columns=['unaware-unaware',
                                                             'aware-unaware',
                                                             'unaware-aware',
                                                             'aware-aware'])
    df2_count = pd.DataFrame(transition_count,columns=['unaware-unaware',
                                                       'aware-unaware',
                                                       'unaware-aware',
                                                       'aware-aware'])
    df2_transition['experiment'] = 2
    df2_transition['sub'] = sub
    df2_count['experiment'] = 2
    df2_count['sub'] = sub
    ##############################################################################
    df3         = pd.read_csv('e3.csv').iloc[:,1:]
    df              = df3.copy()
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
    transition_matrix,sub = [],[]
    transition_count = []
    for participant,df_sub in df.groupby('participant'):
        df_sub          = df_sub[df_sub['awareness'] != 3]
        # for 1-back
        temp = pd.crosstab(pd.Series(df_sub['awareness'].values[1:],name='N'),
                           pd.Series(df_sub['awareness'].values[:-1],name='N-1'),
                           normalize=1)
        transition_matrix.append(temp.get_values().flatten())
        temp = pd.crosstab(pd.Series(df_sub['awareness'].values[1:],name='N'),
                           pd.Series(df_sub['awareness'].values[:-1],name='N-1'),)
        transition_count.append(temp.get_values().flatten())
        sub.append(participant)
    df3_transition = pd.DataFrame(transition_matrix,columns=['unaware-unaware',
                                                             'aware-unaware',
                                                             'unaware-aware',
                                                             'aware-aware'])
    df3_count = pd.DataFrame(transition_count,columns=['unaware-unaware',
                                                       'aware-unaware',
                                                       'unaware-aware',
                                                       'aware-aware'])
    df3_transition['experiment'] = 3
    df3_transition['sub'] = sub
    df3_count['experiment'] = 3
    df3_count['sub'] = sub
    ##################################################################################
    df_transition = pd.concat([df1_transition,df2_transition,df3_transition])
    df_count = pd.concat([df1_count,df2_count,df3_count])
    df_plot = pd.melt(df_transition,id_vars=['experiment','sub'],
                      value_vars=['unaware-unaware',
                                  'aware-unaware',
                                  'unaware-aware',
                                  'aware-aware'])
    df_plot.columns = ['experiment','sub','Transitions','Transition_Probability']
    g = sns.catplot(x = 'experiment',
                    y = 'Transition_Probability',
                    hue = 'Transitions',
                    hue_order = ['unaware-unaware',
                                 'aware-unaware',
                                 'unaware-aware',
                                 'aware-aware'],
                    data = df_plot,
                    kind = 'bar',
                    aspect = 1.5)
    g.set_axis_labels('Experiment','Transition Probability')
    g.savefig(os.path.join(dir_saving,'transition probabilities (prob by exp trans).png'),
              dpi = 300,
              bbox_inches='tight')
    df_plot.to_csv(os.path.join(dir_saving,'for_anovaRM.csv'),index=False)
    formula = 'Transition_Probability ~ C(experiment) + C(Transitions) + C(experiment):C(Transitions)'
    model = ols(formula, df_plot).fit()
    aov_table = anova_lm(model, typ=2)
    utils.eta_squared(aov_table)
    utils.omega_squared(aov_table)
    aov_table.round(5).to_csv(os.path.join(dir_saving,'ANOVA (prob by exp trans).csv'),
                    index=False)
    # 3-2-1 post hoc
    classes = pd.unique(df_plot['Transitions'])
    pairs = combinations(classes,2)
    results = dict(
            pair1 = [],
            pair2 = [],
            ps_mean = [],
            ps_std = [],
            difference = [],
            )
    for (a,b) in list(pairs):
        c1 = df_plot[df_plot['Transitions'] == a]
        c2 = df_plot[df_plot['Transitions'] == b]
        ps = utils.resample_ttest_2sample(c1['Transition_Probability'].values,
                                          c2['Transition_Probability'].values,
                                          n_ps = 200,
                                          n_permutation = 5000,
                                          one_tail = False,
                                          match_sample_size = True)
        results['pair1'].append(a)
        results['pair2'].append(b)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['difference'].append(np.mean(np.abs(c1['Transition_Probability'].values - c2['Transition_Probability'].values)))
    results = pd.DataFrame(results)
    
    temp = []
    for ii, row in df_plot.iterrows():
        a,b = row['Transitions'].split('-')
        if a == b:
            temp.append('same')
        elif a != b:
            temp.append('different')
    
    df_plot['same_different'] = temp
    formula = 'Transition_Probability ~ C(experiment) + C(same_different) + C(experiment):C(same_different)'
    model = ols(formula, df_plot).fit()
    aov_table = anova_lm(model, typ=2)
    utils.eta_squared(aov_table)
    utils.omega_squared(aov_table)
    aov_table.round(5).to_csv(os.path.join(dir_saving,'ANOVA (prob by exp sd).csv'),
                    index=False)
    g = sns.catplot(x = 'experiment',
                    y = 'Transition_Probability',
                    hue = 'same_different',
                    data = df_plot,
                    kind = 'bar',
                    aspect = 1.5)
    g.set_axis_labels('Experiment','Transition Probability')
    g.savefig(os.path.join(dir_saving,'transition probabilities (prob by exp sd).png'),
              dpi = 300,
              bbox_inches='tight')
    
    a = df_plot[df_plot['same_different'] == 'same']
    b = df_plot[df_plot['same_different'] == 'different']
    ps = utils.resample_ttest_2sample(a['Transition_Probability'].values,
                                      b['Transition_Probability'].values,
                                      n_ps = 200,
                                      n_permutation = 50000,
                                      one_tail = False,
                                      match_sample_size = True)
    
    # modeling the transition probability
    df_count['same'] = df_count['aware-aware'] + df_count['unaware-unaware']
    df_count['different'] = df_count['aware-unaware'] + df_count['unaware-aware']
    df_count['total'] = df_count['same'] + df_count['different']
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    