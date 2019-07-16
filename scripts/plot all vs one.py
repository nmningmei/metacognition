# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:07:04 2019

@author: ning
"""

import pandas as pd
import os
from glob import glob
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt
from utils import resample_ttest_2sample,MCPConverter

working_dir = '../results/all_vs_one'

working_data = glob(os.path.join(working_dir,'pos*.csv'))
df_pos = pd.concat([pd.read_csv(f) for f in working_data])
df_pos_all = df_pos[df_pos['feature'] == 'all']
df_pos_one = df_pos[df_pos['feature'] != 'all']

working_data = glob(os.path.join(working_dir,'att*.csv'))
df_att = pd.concat([pd.read_csv(f) for f in working_data])
df_att_all = df_att[df_att['feature'] == 'all']
df_att_one = df_att[df_att['feature'] != 'all']

def comparison(df_,results,experiment):
    for window,df_sub in df_.groupby(['window']):
        df_sub_all = df_sub[df_sub['feature'] == 'all']
        df_sub_one = df_sub[df_sub['feature'] != 'all']
        for name in pd.unique(df_sub_one['feature']):
            a = df_sub_all['score'].values
            temp = df_sub_one[df_sub_one['feature'] == name]
            b = temp['score'].values
            
            ps = resample_ttest_2sample(a,b,
                                        n_ps = 500,
                                        n_permutation = int(1e4),
                                        one_tail = True)
            results['feature'].append(name)
            results['ps_mean'].append(ps.mean())
            results['ps_std'].append(ps.std())
            results['window'].append(window)
            results['experiment'].append(experiment)
    return results
results = dict(
        feature = [],
        ps_mean = [],
        ps_std = [],
        window = [],
        experiment = [],
        )
for df_,exp in zip([df_pos,df_att],['Exp 1.', 'Exp 2.']):
    results = comparison(df_,results,exp)

results = pd.DataFrame(results)

corrected = []
for exp,df_sub in results.groupby(['experiment']):
    df_sub = df_sub.sort_values(['ps_mean'])
    pvals = df_sub['ps_mean'].values
    converter = MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    corrected.append(df_sub)
corrected = pd.concat(corrected)
corrected = corrected.sort_values(['experiment','window'])

df_pos['experiment'] = 'Exp 1.'
df_att['experiment'] = 'Exp 2.'
df_full = pd.concat([df_pos,df_att])

fig,axes = plt.subplots(figsize = (16,10),
                        nrows = 2,
                        ncols = 2,
                        sharex = True,
                        sharey = True,)

ax = axes[0][0]
sns.barplot(x = 'window',
            y = 'score',
            data = df_pos_all,
            ax = ax,
            color = 'lightskyblue',)
ax.set(xlabel='',ylabel='ROC AUC',title='Exp 1. predicted by all')
ax = axes[0][1]
sns.barplot(x = 'window',
            y = 'score',
            hue = 'feature',
            data = df_pos_one,
            ax = ax,)
ax.set(xlabel='',ylabel='',title='Exp 1. predicted by one')
ax = axes[1][0]
sns.barplot(x = 'window',
            y = 'score',
            data = df_att_all,
            color = 'lightskyblue',
            ax = ax,
            )
ax.set(xlabel='Trials back',ylabel='ROC AUC',title='Exp 2. predicted by all')
ax = axes[1][1]
sns.barplot(x = 'window',
            y = 'score',
            hue = 'feature',
            data = df_att_one,
            ax = ax,)
ax.set(xlabel='Trials back',ylabel='',title='Exp 2. predicted by one')
[ax.axhline(0.5,linestyle='--',color='black',alpha=0.5,) for ax in axes.flatten()]
[ax.set(xlim=(-1,7.5)) for ax in axes.flatten()]

g = sns.catplot(x = 'window',
                y = 'score',
                hue = 'feature',
                hue_order = ['all',
                             'awareness',
                             'confidence',
                             'correct',],
                row = 'experiment',
                data = df_full,
                kind = 'bar',
                aspect = 2)
(g.set_axis_labels('Trial back','ROC AUC')
  .set_titles('{row_name}'))
[ax.axhline(0.5,linestyle='--',color='black',alpha=0.5,) for ax in g.axes.flatten()]
g.savefig('../figures/all vs one.png',
          dpi = 400,
          bbox_inches = 'tight')
g.savefig('../figures/all vs one (light).png',
          bbox_inches = 'tight')


























