#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:08:59 2018

@author: nmei
"""
import os
import pandas as pd
import numpy as np
from glob import glob
import re
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt
import utils

figure_dir = 'figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

to_decode = 'awareness'

working_dir = {ii+1:'batch/results_e{}'.format(ii+1) for ii in range(3)}
working_data = [glob(os.path.join(d,'*.csv')) for _,d in working_dir.items()]
working_data = np.concatenate(working_data)
df = []
for f in working_data:
    exp, = re.findall('\d',f)
    df_temp = pd.read_csv(f).iloc[:,1:]
    df_temp['experiment'] = exp
    df.append(df_temp)
df = pd.concat(df)


fig,axes = plt.subplots(figsize=(18,18),nrows = 3, ncols = 2)
for ((exp,model_name),df_sub),ax in zip(df.groupby(['experiment','model']),axes.flatten()):
    sns.violinplot(x = 'window',
                   y = 'score',
                   hue = 'chance',
                   data = df_sub,
                   ax = ax,
                   split = True,
                   cut = 0,
                   inner = 'quartile',
                   )
    ax.set(title = 'Experiment {} | {}'.format(exp,model_name))
    if int(exp) != 1:
        ax.get_legend().remove()
    else:
        ax.get_legend().set_title('')
        ax.get_legend().get_texts()[0].set_text('Decoding')
        ax.get_legend().get_texts()[1].set_text('Chance level')
fig.tight_layout()
fig.suptitle('Decode {} via [awareness,confidence,correctness]'.format(to_decode),
             y = 1.05)
fig.savefig(os.path.join(figure_dir,'decoding results.png'),dpi = 400, bbox_inches = 'tight')

df = df.sort_values(['sub','window','model','experiment','chance'])
decode = df[df['chance'] == False]
chance = df[df['chance'] == True]
diff = decode['score'].values - chance['score'].values
difference = decode.copy()
difference['diff'] = diff
g = sns.catplot(x = 'window',
                y = 'diff',
                hue = 'experiment',
                data = difference,#[df['chance'] == False],
                kind = 'bar',
                col = 'model',
                aspect = 2,)
(g.set_titles('{col_name}')
  .set_axis_labels('N trial back','Decode - Chance'))
g.fig.suptitle('difference between decoding and empirical chance level',y=1.)
g.fig.savefig(os.path.join(figure_dir,'difference between decoding and empirical chance level.png'),
              dpi = 400, bbox_inches='tight')

results = dict(
        experiment = [],
        model = [],
        window = [],
        ps_mean = [],
        ps_std = [],
        )
for attri, df_sub in df.groupby(['experiment','model','window']):
    df_sub_decode = df_sub[df_sub['chance'] == True]
    df_sub_chance = df_sub[df_sub['chance'] == False]
    ps = utils.resample_ttest_2sample(df_sub_decode['score'].values,
                                      df_sub_chance['score'].values,
                                      n_ps = 100,
                                      n_permutation = 5000,
                                      one_tail = True,)
    results['experiment'].append(attri[0])
    results['model'].append(attri[1])
    results['window'].append(attri[2])
    results['ps_mean'].append(np.mean(ps))
    results['ps_std'].append(np.std(ps))
results = pd.DataFrame(results)

corrected_results = []
for attri,df_sub in results.groupby(['experiment','model']):
    pvals = df_sub['ps_mean']
    idx_sort = np.argsort(pvals)
    df_sub = df_sub.iloc[idx_sort,:]
    pvals = df_sub['ps_mean'].values
    converter = utils.MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    corrected_results.append(df_sub)
corrected_results = pd.concat(corrected_results)
corrected_results = corrected_results.sort_values(['experiment','window','model'])


df_copy = df.copy()
for name in ['awareness','confidence','correctness']:
    df_copy[name][df_copy['model'] == 'LogisticRegression'] = df_copy[name][df_copy['model'] == 'LogisticRegression'].apply(np.exp)
df_coef = pd.melt(df_copy[df_copy['chance'] == False],id_vars = ['sub','window','experiment','model'],
                  value_vars = ['awareness','confidence','correctness'])
df_coef.columns = ['sub',
                   'window',
                   'experiment',
                   'model',
                   'Attributes',
                   'value']
g = sns.catplot(x = 'window',
                y = 'value',
                hue = 'Attributes',
                kind = 'bar',
                row = 'experiment',
                col = 'model',
                data = df_coef,
                aspect = 2,
                sharey = False)
reference = [1,1./3,1,1./3,1,1./3,]
ylabels = ['Odd Ratio','Feature Importance','Odd Ratio','Feature Importance','Odd Ratio','Feature Importance',]
[ax.axhline(r,linestyle='--',color='black',alpha=0.5) for r,ax in zip(reference,g.fig.axes)]
[ax.set(ylabel=ylabel) for ylabel,ax in zip(ylabels,g.fig.axes)]
g.fig.suptitle('Decode {} via [awareness,confidence,correctness]'.format(to_decode),
             y = 1.05)
g.savefig(os.path.join(figure_dir,'weights plot.png'),dpi=400,bbox_inches='tight')
















































