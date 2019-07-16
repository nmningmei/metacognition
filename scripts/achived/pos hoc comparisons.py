#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:51:50 2018

@author: nmei
"""

import pandas as pd
import numpy as np
figure_dir = '../figures'
save_dir = '../results'
from utils import resample_ttest_2sample,MCPConverter


# exps
pos = pd.read_csv('../results/Pos.csv')
att = pd.read_csv('../results/ATT.csv')

# epx 1
results = dict(greater = [],
               lesser  = [],
               ps_mean = [],
               ps_std  = [],
               model   = [],)
df = pos[(pos['window'] > 0) & (pos['window'] < 4)]
for model,df_sub in df.groupby('model'):
    pairs = [['awareness','confidence'],
             ['awareness','correct'],
             ['confidence','correct']]
    for pair in pairs:
        a = df_sub[pair[0]].values
        b = df_sub[pair[1]].values
        if a.mean() < b.mean():
            pair = [pair[1],pair[0]]
            a = df_sub[pair[0]].values
            b = df_sub[pair[1]].values
        ps = resample_ttest_2sample(a,b,500,10000)
        results['greater'].append(pair[0])
        results['lesser'].append(pair[1])
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['model'].append(model)
results = pd.DataFrame(results)

temp = []
for modle, df_sub in results.groupby('model'):
    idx_sort = np.argsort(df_sub['ps_mean'].values)
    df_sub = df_sub.iloc[idx_sort,:]
    converter = MCPConverter(df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
results = pd.concat(temp)
results_pos = results.copy()
results_pos['exp'] = 'pos'

# epx 2
results = dict(greater = [],
               lesser  = [],
               ps_mean = [],
               ps_std  = [],
               model   = [],)
df = att[(att['window'] > 0) & (att['window'] < 4)]
for model,df_sub in df.groupby('model'):
    pairs = [['awareness','confidence'],
             ['awareness','correct'],
             ['confidence','correct']]
    for pair in pairs:
        a = df_sub[pair[0]].values
        b = df_sub[pair[1]].values
        if a.mean() < b.mean():
            pair = [pair[1],pair[0]]
            a = df_sub[pair[0]].values
            b = df_sub[pair[1]].values
        ps = resample_ttest_2sample(a,b,500,10000)
        results['greater'].append(pair[0])
        results['lesser'].append(pair[1])
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['model'].append(model)
results = pd.DataFrame(results)

temp = []
for modle, df_sub in results.groupby('model'):
    idx_sort = np.argsort(df_sub['ps_mean'].values)
    df_sub = df_sub.iloc[idx_sort,:]
    converter = MCPConverter(df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
results = pd.concat(temp)
results_att = results.copy()
results_att['exp'] = 'att'

results = pd.concat([results_pos,results_att])
results.to_csv('../results/pos hoc comparisons.csv')

window = dict(greater = [],
              lesser  = [],
              ps_mean = [],
              ps_std  = [],
              model   = [],)
for model,df_sub in df.groupby('model'):
    pairs = [[1,2],
             [2,3],
             [1,3]]
    for pair in pairs:
        a = df_sub[df_sub['window'] == pair[0]][['awareness','confidence','correct']].mean(1).values
        b = df_sub[df_sub['window'] == pair[1]][['awareness','confidence','correct']].mean(1).values
        if a.mean() < b.mean():
            pair = [pair[1],pair[0]]
            a = df_sub[df_sub['window'] == pair[0]][['awareness','confidence','correct']].mean(1).values
            b = df_sub[df_sub['window'] == pair[1]][['awareness','confidence','correct']].mean(1).values
        ps = resample_ttest_2sample(a,b,500,10000)
        window['greater'].append(pair[0])
        window['lesser'].append(pair[1])
        window['ps_mean'].append(ps.mean())
        window['ps_std'].append(ps.std())
        window['model'].append(model)
window = pd.DataFrame(window)

temp = []
for model, df_sub in window.groupby('model'):
    idx_sort = np.argsort(df_sub['ps_mean'].values)
    df_sub = df_sub.iloc[idx_sort,:]
    converter = MCPConverter(df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
window = pd.concat(temp)