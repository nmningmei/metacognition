#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:15:41 2018

@author: nmei
"""
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
from utils import post_processing2,eta_squared,omega_squared,multiple_pairwise_comparison
import os
working_dir = '../results/'
pd.options.mode.chained_assignment = None
pos = pd.read_csv('../results/Pos_control.csv')
att = pd.read_csv('../results/ATT_control.csv')

# exp 1: awareness, 
df = pos.copy()
df = df[(df.window < 4) & (df.window > 0)]
for model_, df_sub in df.groupby('model'):
    df_sub = post_processing2(df_sub,names=['success','awareness','confidence'])
    formula = 'value ~ C(Attributions) + C(window) + C(Attributions):C(window)'
    model = ols(formula,df_sub).fit()
    aov_table = anova_lm(model,typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    if model_ == 'LogisticRegression':
        aov_table.index = ['Weights','Trials','Interaction','Residual']
        print('\n',model_,'\n',aov_table.round(4))
    elif model_ == 'DecisionTreeClassifier':
        aov_table.index = ['Feature Importance','Trials','Interaction','Residual']
        print('\n',model_,'\n',aov_table.round(4))
    

# exp 2
df = att.copy()
df = df[(df.window < 4) & (df.window > 0)]
for model_, df_sub in df.groupby('model'):
    df_sub = post_processing2(df_sub,names=['attention','awareness','confidence'])
    formula = 'value ~ C(Attributions) + C(window) + C(Attributions):C(window)'
    model = ols(formula,df_sub).fit()
    aov_table = anova_lm(model,typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    if model_ == 'LogisticRegression':
        aov_table.index = ['Weights','Trials','Interaction','Residual']
        print('\n',model_,'\n',aov_table.round(4))
    elif model_ == 'DecisionTreeClassifier':
        aov_table.index = ['Feature Importance','Trials','Interaction','Residual']
        print('\n',model_,'\n',aov_table.round(4))

#######################################################################
#################################################################
##########################
df = pos.copy()
df = pd.concat([df[df.window == 1],
                df[df.window == 2],
                df[df.window == 3]])
df = df[['awareness', 'confidence', 'success', 'model', 'score', 'sub',
       'window']]
# this is Benjamini & Hochberg (1995) (BH or alias fdr)
# giving you a pairwise comparison between variables
# and it is one-tailed
compar_pos = multiple_pairwise_comparison(df,1000,10000,method='bonferroni')
compar_pos.to_csv(os.path.join(working_dir,'pairwise_comparision_pos_control (1,2,3).csv'),index=False)
compar_pos = multiple_pairwise_comparison(df[df.window<3],1000,10000,method='bonferroni')
compar_pos.to_csv(os.path.join(working_dir,'pairwise_comparision_pos_control (1,2).csv'),index=False)

df = att.copy()
df = pd.concat([df[df.window == 1],
                df[df.window == 2],
                df[df.window == 3]])
compar_att = multiple_pairwise_comparison(df,1000,10000,method='bonferroni')
compar_att.to_csv(os.path.join(working_dir,'pairwise_comparison_att_control (1,2,3).csv'),index=False)
compar_att = multiple_pairwise_comparison(df[df.window<3],1000,10000,method='bonferroni')
compar_att.to_csv(os.path.join(working_dir,'pairwise_comparison_att_control (1,2).csv'),index=False)














































































