# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:52:14 2019

@author: ning
"""

import os
from glob import glob

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from utils import (MCPConverter,
                   stars)
from itertools import combinations
sns.set_context('poster')
sns.set_style('whitegrid')

working_dir = '../results/linear_mixed'
figure_dir = '../figures/linear_mixed'
working_data = glob(os.path.join(working_dir,'*fit.csv'))
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
n_sub = {'POS':15,'ATT':16}
df_plot = []
for f in working_data:
    f = f.replace('\\','/')
    df = pd.read_csv(f)
    attr = df['Unnamed: 0'].apply(lambda x:x.split('_')[0])
    time = df['Unnamed: 0'].apply(lambda x:int(x.split('_')[1]))
    df = df.iloc[:,1:]
    df['Attributes'] = attr
    df['time'] = time
    df['experiment'] = f.split('/')[-1].split('_')[0]
    df = df.sort_values(['sign'])
    converter = MCPConverter(df['sign'].values)
    d = converter.adjust_many()
    df['ps_corrected'] = d['bonferroni'].values
    df_plot.append(df)
df_plot = pd.concat(df_plot)
df_plot['star'] = df_plot['ps_corrected'].apply(stars)
df_plot['Attributes'] = df_plot['Attributes'].map({'awareness':'awareness','confidence':'confidence','correct':'correctness'})

res = dict(experiment = [],
           time = [],
           t = [],
           p = [],
           level1 = [],
           level2 = [],
           diff_mean = [],
           diff_std = [],
           dof = [],)
for (exp,time),df_sub in df_plot.groupby(['experiment','time']):
    unique_factors = pd.unique(df_sub['Attributes'])
    pairs = combinations(unique_factors,2)
    df_sub['sd'] = df_sub['se'] * np.sqrt(df_sub['dof'])
    for level1,level2 in pairs:
        a = df_sub[df_sub['Attributes'] == level1]
        b = df_sub[df_sub['Attributes'] == level2]
        t,p = stats.ttest_ind_from_stats(a['Estimate'].values[0],
                                         a['sd'].values[0],
                                         a['dof'].values[0],
                                         b['Estimate'].values[0],
                                         b['sd'].values[0],
                                         b['dof'].values[0],
                                         equal_var = False)
        res['experiment'].append(exp)
        res['time'].append(time)
        res['t'].append(t)
        res['p'].append(p)
        res['level1'].append(level1)
        res['level2'].append(level2)
        res['dof'].append(n_sub[exp])
        res['diff_mean'].append(a.Estimate.values[0] - b.Estimate.values[0])
        res['diff_std'].append(np.sqrt(a.sd.values[0]**2 + b.sd.values[0]**2))
res = pd.DataFrame(res)
temp = []
for exp,df_sub in res.groupby(['experiment']):
    df_sub = df_sub.sort_values(['p'])
    converter = MCPConverter(df_sub['p'].values)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
res = pd.concat(temp)
res['star'] = res['p_corrected'].apply(stars)
df_plot.to_csv('../results/mixed_linear_model.csv',index=False)
res.to_csv('../results/mixed_linear_model_pairwise.csv',index=False)

y_pos = {'POS':0.55,
         'ATT': 0.17}
for ii,exp in enumerate(['POS','ATT']):
    fig,ax = plt.subplots(figsize = (12,6))

    df_sub = df_plot[df_plot['experiment'] == exp]
    res_sub = res[res['experiment'] == exp]
    ax = sns.barplot(x = 'time',
                     y = 'Estimate',
                     hue = 'Attributes',
                     hue_order = ['correctness','awareness','confidence',],
                     data = df_sub,
                     ax = ax,
                     alpha = 0.9,)
    df_sub = pd.concat([df_sub[df_sub['Attributes'] == attr].sort_values(['time']) for attr in ['correctness','awareness','confidence']])
    df_sub['subtract'] = np.concatenate([np.arange(-0.27,3,),
                                         np.arange(0,4),
                                         np.arange(0.27,4)])
    df_sub = df_sub.sort_values(['time'])
    
    
    ax.errorbar(df_sub['subtract'].values,
                df_sub['Estimate'].values,
                yerr = df_sub['se'].values,
#                yerr = (df_sub['upr'].values - df_sub['Estimate'].values,
#                        df_sub['Estimate'].values - df_sub['lwr'].values),
                linestyle = '',
                color = 'black',
                linewidth = 6,
                alpha = 1.,
                capsize = 2,)
    exp_title = "Exp. 1" if exp == 'POS' else "Exp. 2"
    ax.set(title = exp_title,
           xlabel = 'Trial back',
           ylabel = 'Coefficients',
           ylim = (-0.06,y_pos[exp] + 0.1 * y_pos[exp]))
    ax.legend(loc = 'upper right',bbox_to_anchor=(1.,0.8))
    for kk,row in df_sub.iterrows():
        if row['star'] != 'n.s.':
            ax.annotate(row['star'],
                        xy = (row['subtract'],y_pos[exp]),
                        ha='center')
    fig.savefig(os.path.join(figure_dir,
                             f'{exp}.jpeg'),
    dpi = 300,
    bbox_inches = 'tight')




