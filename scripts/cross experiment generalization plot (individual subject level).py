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
import utils
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
import matplotlib.pyplot as plt

working_dir = '../results/cross_experiment_generalization'
df = pd.read_csv(os.path.join(working_dir,'cross experiment generalization (individual level).csv'))
df_plot = df.groupby(['window',
                      'model',
                      'participant',
                      'experiment_test',
                      'experiment_train']).mean().reset_index()
#df_plot = pd.concat([df_plot,df_same])
df_plot = df_plot[(df_plot['window'] > 0) & (df_plot['window'] < 5)]
df_plot.to_csv('../results/for_spss/auc as by window and model.csv',index=False)

df_plot['experiment_train'] = df_plot['experiment_train'].map({'POS':"Trained on Exp.1/ Tested on Exp.2",
                                                               'ATT':"Trained on Exp.2/ Tested on Exp.1"})
g = sns.factorplot(x = 'window',
                   y = 'score',
                   hue = 'model',
                   row = 'experiment_train',
#                   col = 'experiment_test',
                   data = df_plot,
                   kind = 'bar',
                   aspect = 2,
                   hue_order = ['RandomForestClassifier',
                                'LogisticRegression'],
                   row_order = ['Trained on Exp.1/ Tested on Exp.2',
                                'Trained on Exp.2/ Tested on Exp.1'],
#                   col_order = ['POS','ATT'],
                   legend_out = False)
(g.set_axis_labels('Trial back','ROC AUC')
  .set(ylim=(0.45,0.8))
  .set_titles('{row_name}')
  .despine(left=True))
g.axes[0][0].get_legend().set_title("")
[ax.axhline(0.5,linestyle='--',color='black',alpha=0.5) for ax in g.fig.axes]
g.savefig('../figures/within cross experiment CV results_3_1_features.png',dpi=500,bbox_inches='tight')




















