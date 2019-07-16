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
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
import matplotlib.pyplot as plt


working_dir = '../results/cross_experiment_generalization'
df = pd.read_csv(os.path.join(working_dir,'cross experiment generalization.csv'))
df_full = pd.read_csv(os.path.join(working_dir,'cross experiment generalization (folds).csv'))
df_corrected = pd.read_csv(os.path.join(working_dir,'cross experimnet validation post test.csv'))

df_plot = df.copy()
resample = []
n = 500
for (model_name,experiment_name),df_sub in df_plot.groupby(['model','test']):
    df_sub
    temp_ = []
    for window, df_sub_sub in df_sub.groupby(['window']):
        temp = pd.concat([df_sub_sub]*n,ignore_index=True)
        np.random.seed(12345)
        scores = np.random.normal(loc = df_sub_sub['score_mean'].values[0],
                                  scale = df_sub_sub['score_std'].values[0],
                                  size = n)
        temp['scores'] = scores
        temp_.append(temp)
    resample.append(pd.concat(temp_))
resample = pd.concat(resample)

np.random.seed(12345)
g = sns.factorplot(x           = 'window',
                y           = 'score',
                hue         = 'test',
                col         = 'model',
                data        = df_full,
                kind        = 'bar',
                ci          = 'sd',
                )
[ax.axhline(0.5,linestyle='--',color='black',alpha=0.5) for ax in g.fig.axes]

g = sns.factorplot(x           = 'window',
                y           = 'p_corrected',
                hue         = 'test',
                col         = 'model',
                data        = df_corrected,
                kind        = 'bar')
g.set(ylim=(0,0.005))
