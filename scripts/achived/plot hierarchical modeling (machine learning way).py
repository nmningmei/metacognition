#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 20:09:30 2018

@author: nmei
"""

import pandas as pd
import seaborn as sns
sns.set_context('poster')
sns.set_style('whitegrid')
import os
import matplotlib.pyplot as plt
results_dir = '../results/'
working_dir = '../data/'
figures_dir = '../figures/'


att = pd.read_csv(os.path.join(results_dir,
                               'att_uni_multi.csv'))
pos = pd.read_csv(os.path.join(results_dir,
                               'pos_uni_multi.csv'))
df = pos.copy()
fig,axes = plt.subplots(figsize=(15,18),
                        nrows=2,
                        ncols=2,
                        sharex=True,
                        sharey=True,)
for ax,(var,df_sub) in zip(axes.flatten()[:2],df.groupby('variation')):
    if var == 'univariate':
        ax = sns.barplot(x='window',
                         y='r2_mean',
                         hue='feature',
                         data=df_sub,
                         ax=ax)
        ax.set(xlabel='',
               ylabel='',
               title='Predicted by Single Variable (POS)',
               xticks=range(5),
               xticklabels=range(5))
    else:
        ax = sns.barplot(x='window',
                         y='r2_mean',
                         data=df_sub,
                         color='deepskyblue',
                         ax=ax)
        ax.set(xlabel='',
               ylabel='Variance Explained ($r^2$)',
               title='Predicted by all Variables (POS)',
               xticks=range(5),
               xticklabels=range(5))
df = att.copy()
for ax,(var,df_sub) in zip(axes.flatten()[2:],df.groupby('variation')):
    if var == 'univariate':
        ax = sns.barplot(x='window',
                         y='r2_mean',
                         hue='feature',
                         data=df_sub,
                         ax=ax)
        ax.set(xlabel='Trials look back',
               ylabel='',
               title='Predicted by Single Variable (ATT)',
               )
    else:
        ax = sns.barplot(x='window',
                         y='r2_mean',
                         data=df_sub,
                         color='deepskyblue',
                         ax=ax)
        ax.set(xlabel='Trials look back',
               ylabel='Variance Explained ($r^2$)',
               title='Predicted by all Variables (ATT)')
fig.savefig(os.path.join(figures_dir,'variance explained experiment comparison logistic regression.png'),
            bbox_inches='tight',
            dpi=500)

df = pos.copy()
fig,axes = plt.subplots(figsize=(15,18),
                        nrows=2,
                        ncols=2,
                        sharex=True,
                        sharey=True,)
for ax,(var,df_sub) in zip(axes.flatten()[:2],df.groupby('variation')):
    if var == 'univariate':
        ax = sns.barplot(x='window',
                         y='weight',
                         hue='feature',
                         data=df_sub,
                         ax=ax)
        ax.set(xlabel='',
               ylabel='',
               title='Predicted by Single Variable (POS)',
               xticks=range(5),
               xticklabels=range(5))
    else:
        ax = sns.barplot(x='window',
                         y='weight',
                         hue='feature',
                         data=df_sub,
#                         color='deepskyblue',
                         ax=ax)
        ax.set(xlabel='',
               ylabel=r'Weight ($\beta$)',
               title='Predicted by all Variables (POS)',
               xticks=range(5),
               xticklabels=range(5))
df = att.copy()
for ax,(var,df_sub) in zip(axes.flatten()[2:],df.groupby('variation')):
    if var == 'univariate':
        ax = sns.barplot(x='window',
                         y='weight',
                         hue='feature',
                         data=df_sub,
                         ax=ax)
        ax.set(xlabel='',
               ylabel='',
               title='Predicted by Single Variable (ATT)',
               xticks=range(5),
               xticklabels=range(5))
    else:
        ax = sns.barplot(x='window',
                         y='weight',
                         hue='feature',
                         data=df_sub,
#                         color='deepskyblue',
                         ax=ax)
        ax.set(xlabel='',
               ylabel=r'Weight ($\beta$)',
               title='Predicted by all Variables (ATT)',
               xticks=range(5),
               xticklabels=range(5))
fig.savefig(os.path.join(figures_dir,'weights experiment comparison logistic regression.png'),
                         bbox_inches='tight',
                         dpi=500)
















































