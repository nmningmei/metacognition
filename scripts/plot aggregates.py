#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:10:24 2019

@author: nmei
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
sns.set_context('poster')
sns.set_style('whitegrid')
from utils import resample_ttest,MCPConverter,stars
from statsmodels.stats.anova import AnovaRM
#from matplotlib import pyplot as plt
figure_dir = '../figures'
saving_dir = '../results/aggregate_experiment_score'

feature_names = [
                 'correct',
                 'awareness',
                 'confidence',]
new_names = [f"{name}_{n_back}" for n_back in np.arange(1,5) for name in feature_names ]

working_dir = '../results/aggregate_experiment_score'
working_data = glob(os.path.join(working_dir,'aggregate*.csv'))

# exp 1 #########################
experiment = 'pos'
pos_file = [item for item in working_data if (f'{experiment}' in item)][0]
df_pos = pd.read_csv(pos_file)
df_pos_save = df_pos.copy()
df_pos_save.loc[df_pos_save['model_name'] == 'LogisticRegression',new_names] = \
    df_pos_save.loc[df_pos_save['model_name'] == 'LogisticRegression',new_names].apply(np.exp)
df_pos_save.to_csv(os.path.join(saving_dir,
                                f'features_normalized_{experiment}.csv'),
        index = False)

df_pos_plot = pd.melt(df_pos_save,
                  id_vars = ['sub_name','model_name'],
                  value_vars = new_names,
                  var_name = 'Attributes',)
def get_attr(x):
    return x.split('_')[0]
def get_window(x):
    return int(x.split('_')[-1])
df_pos_plot['attr'] = df_pos_plot['Attributes'].apply(get_attr)
df_pos_plot['window'] = df_pos_plot['Attributes'].apply(get_window)

df_pos_plot = df_pos_plot.sort_values(['model_name','window','attr'])

g = sns.catplot(x = 'window',
                y = 'value',
                row = 'model_name',
                hue = 'attr',
                data = df_pos_plot,
                kind = 'bar',
                aspect = 3,
                sharey = False,
                )
g._legend.set_title("Attibutes")
scores = df_pos[df_pos['model_name'] == "LogisticRegression"]['scores_mean'].values
g.axes.flatten()[0].set(ylabel = 'Odd Ratio',
              title = f"LogisticRegression scores = {scores.mean():.3f} +/- {scores.std():.3f}")
scores = df_pos[df_pos['model_name'] == "RandomForest"]['scores_mean'].values
g.axes.flatten()[1].set(ylabel = 'Feature Importance (normalized)',
               title = f"RandomForest scores = {scores.mean():.3f} +/- {scores.std():.3f}")
g.fig.suptitle("Exp. 1",y = 1.02)
g.savefig(os.path.join(figure_dir,
                       f'{experiment}_aggregate_features.png'),
        dpi = 400,
        bbox_inches = 'tight')

for model_name,df_sub in df_pos_plot.groupby(['model_name']):
    aovrm = AnovaRM(df_sub, 
                  'value', 
                  'sub_name', 
                  within = ['attr','window'])
    res = aovrm.fit()
    anova_table = res.anova_table
    anova_table.round(5).to_csv(os.path.join(
            saving_dir,
            f'{experiment}_{model_name}.csv'),index=True)

ttest_results = dict(
        model_name = [],
        window = [],
        attribute = [],
        ps_mean = [],
        ps_std = [],
        value_mean = [],
        value_std = [],
        baseline = [],
        )
for (model_name,window,attribute),df_sub in df_pos_plot.groupby([
                        'model_name',
                        'window',
                        'attr']):
    print(model_name,window,attribute,df_sub['value'].values.mean())
    if model_name == 'LogisticRegression':
        baseline = 1
        ps = resample_ttest(df_sub['value'].values,baseline = baseline,
                            n_ps = 100,n_permutation = int(1e6),one_tail = False)
    elif model_name == 'RandomForest':
        baseline = 0
        ps = resample_ttest(df_sub['value'].values,baseline = baseline,
                            n_ps = 100,n_permutation = int(1e6),one_tail = False)
    ttest_results['model_name'].append(model_name)
    ttest_results['window'].append(window)
    ttest_results['attribute'].append(attribute)
    ttest_results['ps_mean'].append(ps.mean())
    ttest_results['ps_std'].append(ps.std())
    ttest_results['value_mean'].append(df_sub['value'].values.mean())
    ttest_results['value_std'].append(df_sub['value'].values.std())
    ttest_results['baseline'].append(baseline)
ttest_results = pd.DataFrame(ttest_results)
temp = []
for model_name, df_sub in ttest_results.groupby(['model_name']):
    df_sub = df_sub.sort_values(['ps_mean'])
    converter = MCPConverter(pvals = df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
ttest_results = pd.concat(temp)
ttest_results = ttest_results.sort_values(['model_name','window','attribute'])
ttest_results['stars'] = ttest_results['ps_corrected'].apply(stars)
ttest_results.to_csv(os.path.join(
        saving_dir,
        f't_test_{experiment}.csv'),index=False)
# exp 2 ###################
experiment = 'att'
att_file = [item for item in working_data if (f'{experiment}' in item)][0]
df_att = pd.read_csv(att_file)
df_att_save = df_att.copy()
df_att_save.loc[df_att_save['model_name'] == 'LogisticRegression',new_names] = \
    df_att_save.loc[df_att_save['model_name'] == 'LogisticRegression',new_names].apply(np.exp)
df_att_save.to_csv(os.path.join(saving_dir,
                                f'features_normalized_{experiment}.csv'))

df_att_plot = pd.melt(df_att_save,
                  id_vars = ['sub_name','model_name'],
                  value_vars = new_names,
                  var_name = 'Attributes',)

df_att_plot['attr'] = df_att_plot['Attributes'].apply(get_attr)
df_att_plot['window'] = df_att_plot['Attributes'].apply(get_window)

df_att_plot = df_att_plot.sort_values(['model_name','window','attr'])

g = sns.catplot(x = 'window',
                y = 'value',
                row = 'model_name',
                hue = 'attr',
                data = df_att_plot,
                kind = 'bar',
                aspect = 3,
                sharey = False,
                )
g._legend.set_title("Attibutes")
scores = df_att[df_att['model_name'] == "LogisticRegression"]['scores_mean'].values
g.axes.flatten()[0].set(ylabel = 'Odd Ratio',
              title = f"LogisticRegression scores = {scores.mean():.3f} +/- {scores.std():.3f}")
scores = df_att[df_att['model_name'] == "RandomForest"]['scores_mean'].values
g.axes.flatten()[1].set(ylabel = 'Feature Importance (normalized)',
               title = f"RandomForest scores = {scores.mean():.3f} +/- {scores.std():.3f}")
g.fig.suptitle("Exp. 2",y = 1.02)
g.savefig(os.path.join(figure_dir,
                       f'{experiment}_aggregate_features.png'),
        dpi = 400,
        bbox_inches = 'tight')

df_att_plot['attr'] = df_att_plot['Attributes'].apply(get_attr)
df_att_plot['window'] = df_att_plot['Attributes'].apply(get_window)

for model_name,df_sub in df_att_plot.groupby(['model_name']):
    aovrm = AnovaRM(df_sub, 
                  'value', 
                  'sub_name', 
                  within = ['attr','window'])
    res = aovrm.fit()
    anova_table = res.anova_table
    anova_table.round(5).to_csv(os.path.join(
            saving_dir,
            f'{experiment}_{model_name}.csv'),index=True)


ttest_results = dict(
        model_name = [],
        window = [],
        attribute = [],
        ps_mean = [],
        ps_std = [],
        value_mean = [],
        value_std = [],
        baseline = [],
        )
for (model_name,window,attribute),df_sub in df_att_plot.groupby([
                        'model_name',
                        'window',
                        'attr']):
    print(model_name,window,attribute,df_sub['value'].values.mean())
    if model_name == 'LogisticRegression':
        baseline = 1
        ps = resample_ttest(df_sub['value'].values,baseline = baseline,
                            n_ps = 100,n_permutation = int(1e6),one_tail = False)
    elif model_name == 'RandomForest':
        baseline = 0
        ps = resample_ttest(df_sub['value'].values,baseline = baseline,
                            n_ps = 100,n_permutation = int(1e6),one_tail = False)
    ttest_results['model_name'].append(model_name)
    ttest_results['window'].append(window)
    ttest_results['attribute'].append(attribute)
    ttest_results['ps_mean'].append(ps.mean())
    ttest_results['ps_std'].append(ps.std())
    ttest_results['value_mean'].append(df_sub['value'].values.mean())
    ttest_results['value_std'].append(df_sub['value'].values.std())
    ttest_results['baseline'].append(baseline)
ttest_results = pd.DataFrame(ttest_results)
temp = []
for model_name, df_sub in ttest_results.groupby(['model_name']):
    df_sub = df_sub.sort_values(['ps_mean'])
    converter = MCPConverter(pvals = df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
ttest_results = pd.concat(temp)
ttest_results = ttest_results.sort_values(['model_name','window','attribute'])
ttest_results['stars'] = ttest_results['ps_corrected'].apply(stars)
ttest_results.to_csv(os.path.join(
        saving_dir,
        f't_test_{experiment}.csv'),index=False)




