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
                   posthoc_multiple_comparison_interaction)
n_ps = 500
n_permutation = int(1e4)
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
    results = []
    for model_name,df_sub in c.groupby(['Models']):
        # main effect of window
        factor = 'Window'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # main effect of attributes
        factor = 'Attributes'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # interaction
        result = posthoc_multiple_comparison_interaction(
                                             df_sub,
                                             depvar         = 'Values',
                                             unique_levels  = ["Window","Attributes"],
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv('../results/post hoc multiple comparison POS_3_1_features.csv',
                   index=False)
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
    results = []
    for model_name,df_sub in c.groupby(['Models']):
        # main effect of window
        factor = 'Window'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # main effect of attributes
        factor = 'Attributes'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # interaction
        result = posthoc_multiple_comparison_interaction(
                                             df_sub,
                                             depvar         = 'Values',
                                             unique_levels  = ["Window","Attributes"],
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv('../results/post hoc multiple comparison ATT_3_1_features.csv',
                   index=False)
    ############################## using RT features ##########################
    pos = pd.read_csv('../results/Pos_RT_features.csv')
    att = pd.read_csv('../results/ATT_RT_features.csv')
    
    df = pos.copy()
    id_vars         = ['model',
                       'score',
                       'sub',
                       'window',]
    value_vars      =[
                      'RT_correct',
                      'RT_awareness',
                      'RT_confidence',
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
    results = []
    for model_name,df_sub in c.groupby(['Models']):
        # main effect of window
        factor = 'Window'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # main effect of attributes
        factor = 'Attributes'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # interaction
        result = posthoc_multiple_comparison_interaction(
                                             df_sub,
                                             depvar         = 'Values',
                                             unique_levels  = ["Window","Attributes"],
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv('../results/post hoc multiple comparison POS_RT_features.csv',
                   index=False)
    
    # att
    df = pos.copy()
    id_vars         = ['model',
                       'score',
                       'sub',
                       'window',]
    value_vars      =[
                      'RT_correct',
                      'RT_awareness',
                      'RT_confidence',
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
    results = []
    for model_name,df_sub in c.groupby(['Models']):
        # main effect of window
        factor = 'Window'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # main effect of attributes
        factor = 'Attributes'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # interaction
        result = posthoc_multiple_comparison_interaction(
                                             df_sub,
                                             depvar         = 'Values',
                                             unique_levels  = ["Window","Attributes"],
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv('../results/post hoc multiple comparison ATT_RT_features.csv',
                   index=False)
    ############################## using 3 features ##########################
    pos = pd.read_csv('../results/Pos_6_features.csv')
    att = pd.read_csv('../results/ATT_6_features.csv')
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
                      'RT_correct',
                      'RT_awareness',
                      'RT_confidence',
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
    results = []
    for model_name,df_sub in c.groupby(['Models']):
        # main effect of window
        factor = 'Window'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # main effect of attributes
        factor = 'Attributes'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # interaction
        result = posthoc_multiple_comparison_interaction(
                                             df_sub,
                                             depvar         = 'Values',
                                             unique_levels  = ["Window","Attributes"],
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv('../results/post hoc multiple comparison POS_6_features.csv',
                   index=False)
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
                      'RT_correct',
                      'RT_awareness',
                      'RT_confidence',
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
    results = []
    for model_name,df_sub in c.groupby(['Models']):
        # main effect of window
        factor = 'Window'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # main effect of attributes
        factor = 'Attributes'
        result = posthoc_multiple_comparison(df_sub,
                                             depvar         = 'Values',
                                             factor         = factor,
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
        # interaction
        result = posthoc_multiple_comparison_interaction(
                                             df_sub,
                                             depvar         = 'Values',
                                             unique_levels  = ["Window","Attributes"],
                                             n_ps           = n_ps,
                                             n_permutation  = n_permutation)
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv('../results/post hoc multiple comparison ATT_6_features.csv',
                   index=False)

































