#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:00:17 2018

@author: nmei
"""

working_dir = 'hierarchical_regression'
saving_dir = 'hierarchical_regression'
figure_dir = 'hierarchical_regression'
import os
import utils
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt

df = pd.read_csv(os.path.join(working_dir,'hierarchical regression.csv'))
df1 = df[[u'experiment', u'model', u'model1', u'model2', u'model3', u'model4',u'sub', u'window']]
df2 = df[[u'experiment',u'sig21', u'sig32', u'sig42', u'sub', u'window']]
df1 = pd.melt(df1,id_vars=[u'experiment', u'model',u'sub', u'window'],
              value_vars=[u'model1', u'model2', u'model3', u'model4',])
df2 = pd.melt(df2,id_vars=[u'experiment', u'model',u'sub', u'window'],
              value_vars=[u'sig21', u'sig32', u'sig42',])
results = dict(
        model = [],
        experiment = [],
        window = [],
        ps21 = [],
        ps32 = [],
        ps42 = [],)
for attri,df_sub in df1.groupby(['model','experiment','window']):
    m1 = df_sub[df_sub['variable'] == 'model1']
    m2 = df_sub[df_sub['variable'] == 'model2']
    m3 = df_sub[df_sub['variable'] == 'model3']
    m4 = df_sub[df_sub['variable'] == 'model4']
    ps21 = utils.resample_ttest_2sample(m2['value'].values,
                                        m1['value'].values,
                                        n_ps = 200,
                                        n_permutation = 5000,
                                        one_tail = True)
    ps32 = utils.resample_ttest_2sample(m3['value'].values,
                                        m2['value'].values,
                                        n_ps = 200,
                                        n_permutation = 5000,
                                        one_tail = True)
    ps42 = utils.resample_ttest_2sample(m4['value'].values,
                                        m2['value'].values,
                                        n_ps = 200,
                                        n_permutation = 5000)
    for var_name,name in zip(['model','experiment','window'],attri):
        results[var_name].append(name)
    results['ps21'].append(ps21.mean())
    results['ps32'].append(ps32.mean())
    results['ps42'].append(ps42.mean())
results = pd.DataFrame(results)

corrected = []
for attri, df_sub in results.groupby(['model','experiment']):
    df_sub = df_sub.iloc[np.argsort(df_sub['ps21'].values),:]
    converter = utils.MCPConverter(pvals = df_sub['ps21'].values)
    d = converter.adjust('bonferroni')
    df_sub['ps21_corrected'] = d
    
    df_sub = df_sub.iloc[np.argsort(df_sub['ps32'].values),:]
    converter = utils.MCPConverter(pvals = df_sub['ps32'].values)
    d = converter.adjust('bonferroni')
    df_sub['ps32_corrected'] = d
    
    df_sub = df_sub.iloc[np.argsort(df_sub['ps42'].values),:]
    converter = utils.MCPConverter(pvals = df_sub['ps42'].values)
    d = converter.adjust('bonferroni')
    df_sub['ps42_corrected'] = d
    
    corrected.append(df_sub)
corrected = pd.concat(corrected)

df3 = pd.melt(corrected[[u'experiment', u'model',u'window',
       u'ps21_corrected', u'ps32_corrected', u'ps42_corrected']],
        id_vars = [u'experiment', u'model',u'window',],
        value_vars = [u'ps21_corrected', u'ps32_corrected', u'ps42_corrected'],)
df3 = df3.sort_values(['experiment','model','window','variable'])


fig,axes = plt.subplots(figsize=(20,10),
                        nrows = 2,ncols=3,
                        sharey = False)

for ii,(ax,(experiment,df_sub)) in enumerate(zip(axes[0],df1.groupby(['experiment']))):
    ax = sns.barplot(x = 'window',
                     y = 'value',
                     hue = 'variable',
                     data = df_sub,
                     ax = ax)
    ax.set(xlabel='')
    if ii > 0:
        ax.set(ylabel='')
    else:
        ax.set(ylabel='ROC_AUC')

for ii,(ax,(experiment,df_sub)) in enumerate(zip(axes[1],df2.groupby(['experiment']))):
    ax = sns.barplot(x = 'window',
                     y = 'value',
                     hue = 'variable',
                     data = df_sub,
                     ax = ax)
    ax.axhline(0.05,xlalel='N trial back')
    if ii > 0:
        ax.set(ylabel='')
    else:
        ax.set(ylabel='P values')

fig.suptitle('''model1: awareness_N ~ confidence_N-1
             model1: awareness_N ~ confidence_N-1 + awareness_N-1
             model3: awareness_N ~ confidence_N-1 + awareness_N-1 + correctness_N
             model3: awareness_N ~ confidence_N-1 + awareness_N-1 + correctness_N-1''',
             y = 1.1)

fig.savefig(os.path.join(figure_dir,'fig.png'),
            dpi = 400,bbox_inches='tight')


df1.columns = ['Experiment',
               'Model',
               'sub',
               'window',
               'Regression Models',
               'value']
g = sns.catplot(x = 'window',
                y = 'value',
                hue = 'Regression Models',
                row = 'Model',
                col = 'Experiment',
                data = df1,
                kind = 'bar',
                aspect = 2)
g.fig.suptitle('''model1: awareness_N ~ confidence_N-1
             model1: awareness_N ~ confidence_N-k + awareness_N-k
             model3: awareness_N ~ confidence_N-j + awareness_N-k + correctness_N
             model3: awareness_N ~ confidence_N-k + awareness_N-k + correctness_N-k''',
             y = 1.2)
[a.axhline(0.5,linestyle='--',color='black',alpha=0.5) for a in g.fig.axes]
(g.set_axis_labels('N trial back','ROC AUC'))



