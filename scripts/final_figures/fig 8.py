#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:17:50 2018

@author: nmei

plot cross experiment generalization

"""

import os
import pandas as pd
import numpy as np
os.chdir('..')
import utils
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
import matplotlib.pyplot as plt
title_map = {'RandomForestClassifier':'Supplementary Fig','LogisticRegression':'Figure'}
figure_dir = '../figures/final_figures'

working_dir = '../results/cross_experiment_generalization'
df = pd.read_csv(os.path.join(working_dir,'cross experiment generalization (individual level).csv'))
df_plot = df.groupby(['window',
                      'model',
                      'participant',
                      'experiment_test',
                      'experiment_train']).mean().reset_index()
df_plot = df_plot[(df_plot['window'] > 0) & (df_plot['window'] < 5)]

cols = ['model', 'pval', 'score_mean', 'score_std', 'train', 'window',]
df_post = {name:[] for name in cols}
for (train,model_name,window),df_sub in df_plot.groupby(['experiment_train','model','window']):
    df_sub
    ps = utils.resample_ttest(df_sub['score'].values,
                              0.5,
                              n_ps = 100,
                              n_permutation = int(1e4),
                              one_tail = True)
    df_post['model'].append(model_name)
    df_post['window'].append(window)
    df_post['train'].append(train)
    df_post['score_mean'].append(df_sub['score'].values.mean())
    df_post['score_std'].append(df_sub['score'].values.std())
    df_post['pval'].append(ps.mean())
df_post = pd.DataFrame(df_post)
temp = []
for (train,model_name), df_sub in df_post.groupby(['train','model']):
    df_sub = df_sub.sort_values(['pval'])
    ps = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = ps)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
df_post = pd.concat(temp)

df_post['star'] = df_post['p_corrected'].apply(utils.stars)
df_post = df_post.sort_values(['window','train','model'])
df_post = df_post[df_post['window'] > 0]

df_plot['experiment_train'] = df_plot['experiment_train'].map({'POS':"Trained on Exp.1/ Tested on Exp.2",
                                                               'ATT':"Trained on Exp.2/ Tested on Exp.1"})
df_plot = df_plot[df_plot['window'] > 0]

for model_name, df_sub in df_plot.groupby(['model']):
    df_post_sub = df_post[df_post['model'] == model_name]
    g = sns.catplot(x = 'window',
                    y = 'score',
                    row = 'experiment_train',
                    data = df_sub,
                    kind = 'bar',
                    aspect = 2.5,
                    row_order = ['Trained on Exp.1/ Tested on Exp.2',
                                 'Trained on Exp.2/ Tested on Exp.1'],
                   legend_out = False)

    for idx_ax, exp in enumerate(['POS','ATT']):
        ax = g.axes.flatten()[idx_ax]
        df_exp_sub = df_post_sub[df_post_sub['train'] == exp]
        print(df_exp_sub)
        for iii,text in enumerate(df_exp_sub['star'].values):
            ax.annotate(text,xy=(iii-0.075,0.75))
    (g.set_axis_labels('Trial back','ROC AUC')
      .set(ylim=(0.45,0.8))
      .despine(left=True)
      .set_titles("{row_name}"))
    [ax.axhline(0.5,linestyle='--',color='black',alpha=0.5) for ax in g.fig.axes]
    g.savefig(os.path.join(figure_dir,'{} 8.jpeg'.format(title_map[model_name])),
              dpi=500,bbox_inches='tight')



























