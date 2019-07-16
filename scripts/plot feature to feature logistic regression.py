# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:14:35 2019

@author: ning
"""

import os
from glob import glob
import pandas as pd
import utils
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')


working_dir = '../results/simple_regression_11'
figure_dir = '../figures'

def add_exp(f):
    temp = pd.read_csv(f)
    if f.split(working_dir[3:])[-1][1:].split('_')[0] == 'Pos':
        temp['experiment'] = 'Exp.1'
    else:
        temp['experiment'] = 'Exp.2'
    
    return temp


working_data = glob(os.path.join(working_dir,'*.csv'))
df = pd.concat([add_exp(f) for f in working_data])

df['formula'] = df['feature_name'] + '_' + df['target_name']
df['Model'] = df['feature_name'] + ' predicts ' + df['target_name']
df = df[df['window'] > 0]
df = df[df['feature_name'] != 'correct']
df = df[df['target_name'] != 'correct']

results = dict(
        experiment = [],
        window = [],
        feature_name = [],
        target_name = [],
        ps_mean = [],
        ps_std = [],
        )

for (window,formula,exp),df_sub in df.groupby(['window','formula','experiment']):
    exp = exp.lower()
    scores = df_sub[df_sub['chance'] == False]
    chance = df_sub[df_sub['chance'] == True]
    
    pvals = utils.resample_ttest_2sample(scores['score'].values,
                                         chance['score'].values,
                                         n_ps = 100,
                                         n_permutation = int(1e4),
                                         one_tail = False)
    
    results['experiment'].append(exp)
    results['window'].append(window)
    results['feature_name'].append(formula.split('_')[0])
    results['target_name'].append(formula.split('_')[1])
    results['ps_mean'].append(pvals.mean())
    results['ps_std'].append(pvals.std())
results = pd.DataFrame(results)

corrected = []
for (exp),df_sub in results.groupby(['experiment']):
    df_sub = df_sub.sort_values(['ps_mean'])
    converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    
    corrected.append(df_sub)
corrected = pd.concat(corrected)
corrected['Model'] = corrected['feature_name'] + ' predicts ' + \
 corrected['target_name']

corrected = corrected.sort_values(['experiment','window'])

g = sns.catplot(x = 'window',
                y = 'score',
                hue = 'Model',
                hue_order = ['confidence predicts awareness',
                             'awareness predicts confidence',
                             ],
                row = 'experiment',
                row_order = ['Exp.1','Exp.2'],
#                col = 'chance',
                data = df[df['chance'] == False],
                kind = 'bar',
                aspect = 2)

kwargs = dict(fontsize = 24,
              color = 'red')
[ax.axhline(0.5,linestyle='--',color='black',alpha=0.6) for ax in g.axes.flatten()]
g.axes[0][0].text(-0.25,0.6,'*',**kwargs)
g.set(ylim=(0,0.7))
(g.set_axis_labels('Trial back','ROC AUC')
  .set_titles('{row_name}'))
g.savefig(os.path.join(
        figure_dir,
        'feature to feature prediction.png'),
        dpi = 400,
        bbox_inches = 'tight',)
g.savefig(os.path.join(
        figure_dir,
        'feature to feature prediction (light).png'),
        bbox_inches = 'tight')


"""
g.axes[0][0].text(-0.23,0.7,'*',**kwargs)
g.axes[0][0].text(-0.10,0.7,'*',**kwargs)
g.axes[0][0].text(0.03,0.7,'*',**kwargs)
g.axes[0][0].text(0.16,0.7,'*',**kwargs)
g.axes[0][0].text(0.29,0.7,'*',**kwargs)
g.axes[0][0].text(1.16,0.7,'*',**kwargs)
g.axes[0][0].text(2-0.23,0.7,'*',**kwargs)
g.axes[0][0].text(3.16,0.7,'*',**kwargs)
g.axes[1][0].text(-0.38,0.7,'*',**kwargs)
g.axes[1][0].text(-0.23,0.7,'*',**kwargs)
g.axes[1][0].text(-0.10,0.7,'*',**kwargs)
g.axes[1][0].text(0.03,0.7,'*',**kwargs)
g.axes[1][0].text(0.16,0.7,'*',**kwargs)
g.axes[1][0].text(0.29,0.7,'*',**kwargs)
g.axes[1][0].text(1.16,0.7,'*',**kwargs)
"""



