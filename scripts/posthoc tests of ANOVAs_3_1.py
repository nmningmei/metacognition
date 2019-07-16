#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:34:12 2018

@author: nmei
"""

import pandas as pd
import numpy as np
from utils import (posthoc_multiple_comparison,
                   post_processing,
                   posthoc_multiple_comparison_interaction,
                   resample_ttest,
                   MCPConverter,
                   stars)
n_ps = 200
n_permutation = int(1e5)
np.random.seed(12345)
if __name__ == '__main__':
    ############################## using 3 features ##########################
    pos = pd.read_csv('../results/Pos_3_1_features.csv')
    att = pd.read_csv('../results/ATT_3_1_features.csv')
    # pos
    df              = pos.copy()
    id_vars         = ['model',
                       'score',
                       'sub',
                       'window',]
    value_vars      =[
                      'correct',
                      'awareness',
                      'confidence',
                      ]
    df_post         = post_processing(df[(df['window'] >0) & (df['window'] < 5)],
                                                    id_vars,value_vars)
    c               = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
    # interaction
    level_window = pd.unique(c['Window'])
    level_attribute = pd.unique(c['Attributes'])
    unique_levels = []
    for w in level_window:
        for a in level_attribute:
            unique_levels.append([w,a])
#    results = []
#    for model_name,df_sub in c.groupby(['Models']):
#        # main effect of window
#        factor = 'Window'
#        result = posthoc_multiple_comparison(df_sub,
#                                             depvar         = 'Values',
#                                             factor         = factor,
#                                             n_ps           = n_ps,
#                                             n_permutation  = n_permutation)
#        result['Model'] = model_name
#        results.append(result)
#        # main effect of attributes
#        factor = 'Attributes'
#        result = posthoc_multiple_comparison(df_sub,
#                                             depvar         = 'Values',
#                                             factor         = factor,
#                                             n_ps           = n_ps,
#                                             n_permutation  = n_permutation)
#        result['Model'] = model_name
#        results.append(result)
#        # interaction
#        result = posthoc_multiple_comparison_interaction(
#                                             df_sub,
#                                             depvar         = 'Values',
#                                             unique_levels  = ["Window","Attributes"],
#                                             n_ps           = n_ps,
#                                             n_permutation  = n_permutation)
#        result['Model'] = model_name
#        results.append(result)
#    
#    results = pd.concat(results)
#    results.to_csv('../results/post hoc multiple comparison POS_3_1_features.csv',
#                   index=False)
    
    # 1 sample t test against baseline
    c_test = c.copy()
    # normalize random forest feature importance because of reasons ...
#    from sklearn.preprocessing import scale
#    c_test.loc[c_test['Models'] == "RandomForestClassifier",'Values'] = \
#        scale(c_test.loc[c_test['Models'] == "RandomForestClassifier",'Values'].values)
    ttest_results   = dict(
        model_name  = [],
        window      = [],
        attribute   = [],
        ps_mean     = [],
        ps_std      = [],
        value_mean  = [],
        value_std   = [],
        baseline    = [],
        )
    for (model_name,attribute,window),df_sub in c_test.groupby(['Models','Attributes','Window']):
        if model_name == "RandomForestClassifier":
            baseline = 1/3.
            ps = resample_ttest(df_sub['Values'].values,
                                baseline        = baseline,
                                n_ps            = n_ps,
                                n_permutation   = n_permutation,
                                one_tail        = True,)
        elif model_name == "LogisticRegression":
            baseline = 1
            ps = resample_ttest(df_sub['Values'].values,
                                baseline        = baseline,
                                n_ps            = n_ps,
                                n_permutation   = n_permutation,
                                one_tail        = True,)
        ttest_results['model_name'].append(model_name)
        ttest_results['window'].append(window)
        ttest_results['attribute'].append(attribute)
        ttest_results['ps_mean'].append(ps.mean())
        ttest_results['ps_std'].append(ps.std())
        ttest_results['value_mean'].append(df_sub['Values'].values.mean())
        ttest_results['value_std'].append(df_sub['Values'].values.std())
        ttest_results['baseline'].append(baseline)
    ttest_results   = pd.DataFrame(ttest_results)
    temp            = []
    for model_name, df_sub in ttest_results.groupby(['model_name']):
        df_sub                  = df_sub.sort_values(['ps_mean'])
        converter               = MCPConverter(pvals = df_sub['ps_mean'].values)
        d                       = converter.adjust_many()
        df_sub['ps_corrected']  = d['bonferroni'].values
        temp.append(df_sub)
    ttest_results               = pd.concat(temp)
    ttest_results               = ttest_results.sort_values(['model_name','window','attribute'])
    ttest_results['stars']      = ttest_results['ps_corrected'].apply(stars)
    ttest_results.to_csv('../results/one sample t test POS_3_1_features.csv',
                         index = False)
    # att
    df              = att.copy()
    id_vars         = ['model',
                       'score',
                       'sub',
                       'window',]
    value_vars      =[
                      'correct',
                      'awareness',
                      'confidence',
                      ]
    df_post         = post_processing(df[(df['window'] >0) & (df['window'] < 5)],
                                                    id_vars,value_vars)
    c               = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
    # interaction
    level_window = pd.unique(c['Window'])
    level_attribute = pd.unique(c['Attributes'])
    unique_levels = []
    for w in level_window:
        for a in level_attribute:
            unique_levels.append([w,a])
#    results = []
#    for model_name,df_sub in c.groupby(['Models']):
#        # main effect of window
#        factor = 'Window'
#        result = posthoc_multiple_comparison(df_sub,
#                                             depvar         = 'Values',
#                                             factor         = factor,
#                                             n_ps           = n_ps,
#                                             n_permutation  = n_permutation)
#        result['Model'] = model_name
#        results.append(result)
#        # main effect of attributes
#        factor = 'Attributes'
#        result = posthoc_multiple_comparison(df_sub,
#                                             depvar         = 'Values',
#                                             factor         = factor,
#                                             n_ps           = n_ps,
#                                             n_permutation  = n_permutation)
#        result['Model'] = model_name
#        results.append(result)
#        # interaction
#        result = posthoc_multiple_comparison_interaction(
#                                             df_sub,
#                                             depvar         = 'Values',
#                                             unique_levels  = ["Window","Attributes"],
#                                             n_ps           = n_ps,
#                                             n_permutation  = n_permutation)
#        result['Model'] = model_name
#        results.append(result)
#    
#    results = pd.concat(results)
#    results.to_csv('../results/post hoc multiple comparison ATT_3_1_features.csv',
#                   index=False)
    
    # 1 sample t test against baseline
    c_test = c.copy()
    # normalize random forest feature importance because reasons...
#    from sklearn.preprocessing import scale
#    c_test.loc[c_test['Models'] == "RandomForestClassifier",'Values'] = \
#        scale(c_test.loc[c_test['Models'] == "RandomForestClassifier",'Values'].values)
    ttest_results   = dict(
        model_name  = [],
        window      = [],
        attribute   = [],
        ps_mean     = [],
        ps_std      = [],
        value_mean  = [],
        value_std   = [],
        baseline    = [],
        )
    for (model_name,attribute,window),df_sub in c_test.groupby(['Models','Attributes','Window']):
        if model_name == "RandomForestClassifier":
            baseline = 1/3.
            ps = resample_ttest(df_sub['Values'].values,
                                baseline        = baseline,
                                n_ps            = n_ps,
                                n_permutation   = n_permutation,
                                one_tail        = True,)
        elif model_name == "LogisticRegression":
            baseline = 1
            ps = resample_ttest(df_sub['Values'].values,
                                baseline        = baseline,
                                n_ps            = n_ps,
                                n_permutation   = n_permutation,
                                one_tail        = True,)
        ttest_results['model_name'].append(model_name)
        ttest_results['window'].append(window)
        ttest_results['attribute'].append(attribute)
        ttest_results['ps_mean'].append(ps.mean())
        ttest_results['ps_std'].append(ps.std())
        ttest_results['value_mean'].append(df_sub['Values'].values.mean())
        ttest_results['value_std'].append(df_sub['Values'].values.std())
        ttest_results['baseline'].append(baseline)
    ttest_results   = pd.DataFrame(ttest_results)
    temp            = []
    for model_name, df_sub in ttest_results.groupby(['model_name']):
        df_sub                  = df_sub.sort_values(['ps_mean'])
        converter               = MCPConverter(pvals = df_sub['ps_mean'].values)
        d                       = converter.adjust_many()
        df_sub['ps_corrected']  = d['bonferroni'].values
        temp.append(df_sub)
    ttest_results               = pd.concat(temp)
    ttest_results               = ttest_results.sort_values(['model_name','window','attribute'])
    ttest_results['stars']      = ttest_results['ps_corrected'].apply(stars)
    ttest_results.to_csv('../results/one sample t test ATT_3_1_features.csv',
                         index = False)
































