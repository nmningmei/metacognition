# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:01:18 2019

@author: ning
"""

import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
from utils import MCPConverter
from scipy import stats
sns.set_style('whitegrid')
sns.set_context('poster')

working_dir = '../results/correlation'
chance_dir = '../results/correlation_chance'
figure_dir = '../figures'
working_data = glob(os.path.join(working_dir,'*.csv'))
chance_data = glob(os.path.join(chance_dir,'*.csv'))
working_data = [item for item in working_data if ('3_1' in item)]
chance_data = [item for item in chance_data if ('3_1' in item)]

# pos
pos = pd.concat([pd.read_csv(f) for f in working_data if ('Pos' in f)])
pos = pos.groupby(['sub','feature_name_1','feature_name_2']).mean().reset_index()
#pos = pos.groupby(['window','feature_name_1','feature_name_2']).mean().reset_index()
pos['Comparison'] = pos['feature_name_1'] + '_' +pos['feature_name_2']
pos['Experiment'] = 'Exp.1'
pos['chance'] = False
# att
att = pd.concat([pd.read_csv(f) for f in working_data if ('att' in f)])
att = att.groupby(['sub','feature_name_1','feature_name_2']).mean().reset_index()
#att = att.groupby(['window','feature_name_1','feature_name_2']).mean().reset_index()
att['Comparison'] = att['feature_name_1'] + '_' + att['feature_name_2']
att['Experiment'] = 'Exp.2'
att['chance'] = False
df = pd.concat([pos,att])

## chance
# pos
pos = pd.concat([pd.read_csv(f) for f in chance_data if ('Pos' in f)])
pos = pos.groupby(['sub','feature_name_1','feature_name_2']).mean().reset_index()
#pos = pos.groupby(['window','feature_name_1','feature_name_2']).mean().reset_index()
pos['Comparison'] = pos['feature_name_1'] + '_' +pos['feature_name_2']
pos['Experiment'] = 'Exp.1'
pos['chance'] = True
# att
att = pd.concat([pd.read_csv(f) for f in chance_data if ('att' in f)])
att = att.groupby(['sub','feature_name_1','feature_name_2']).mean().reset_index()
#att = att.groupby(['window','feature_name_1','feature_name_2']).mean().reset_index()
att['Comparison'] = att['feature_name_1'] + '_' + att['feature_name_2']
att['Experiment'] = 'Exp.2'
att['chance'] = True
df_chance = pd.concat([pos,att])

t_test_results = dict(
        experiment = [],
        comparison = [],
        p = [],
        t = [],
        corr_mean = [],
        corr_std = [],)
for (experiment,comparison),df_sub in df.groupby(['Experiment','Comparison']):
    corr = df_sub['correlation'].values
    corr = np.arctanh(corr)
    t,p = stats.ttest_1samp(corr,0,)
    print(corr.shape)
    t_test_results['experiment'].append(experiment)
    t_test_results['comparison'].append(comparison)
    t_test_results['p'].append(p)
    t_test_results['t'].append(t)
    t_test_results['corr_mean'].append(df_sub['correlation'].values.mean())
    t_test_results['corr_std'].append(df_sub['correlation'].values.std())
t_test_results = pd.DataFrame(t_test_results)
temp = []
for experiment,df_sub in t_test_results.groupby(['experiment']):
    df_sub = df_sub.sort_values(['p'])
    ps = df_sub['p'].values
    converter = MCPConverter(pvals = ps)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
t_test_results = pd.concat(temp)
t_test_results = t_test_results[['experiment', 'comparison', 'p_corrected', 't', 'corr_mean', 'corr_std',
       'p']]

df_full = pd.concat([df,df_chance])
df_full['x'] = 0

g = sns.catplot(x = 'x',
                y = 'correlation',
                hue = 'Comparison',
                row = 'Experiment',
#                col = 'chance',
                data = df_full[df_full['chance']==False],
                aspect = 2,
                kind = 'bar',
                )
(g.set_axis_labels('','Correlation Coefficients'))
[ax.set(xticks=[]) for ax in g.axes.flatten()]
g.fig.savefig(os.path.join(figure_dir,
                           'correlation of both experiments among variables_3_1.png'),
                dpi = 400,
                bbox_inches = 'tight')
g.fig.savefig(os.path.join(figure_dir,
                           'correlation of both experiments among variables_3_1 (light).png'),
#                dpi = 400,
                bbox_inches = 'tight')
