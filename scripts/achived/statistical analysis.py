#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:55:10 2018

@author: nmei
"""
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
sns.set_style('whitegrid')
sns.set_context('poster')
from utils import multiple_pairwise_comparison,post_processing,eta_squared,omega_squared
working_dir = '../results/'


if __name__ == '__main__':
    pos = pd.read_csv(os.path.join(working_dir,'Pos.csv'))
    att = pd.read_csv(os.path.join(working_dir,'ATT.csv'))
    
        ###################################################################################
    ##################### test of weights #############################################
    ###################################################################################

#    # exp 1: awareness, confidence,correct by 1,2,3
#    df = pos.copy()
#    df = df[(df.window < 4) & (df.window > 0)]
#    for model_, df_sub in df.groupby('model'):
#        df_sub = post_processing(df_sub)
#        formula = 'value ~ C(Attributions) + C(window) + C(Attributions):C(window)'
#        model = ols(formula,df_sub).fit()
#        aov_table = anova_lm(model,typ=2)
#        eta_squared(aov_table)
#        omega_squared(aov_table)
#        if model_ == 'LogisticRegression':
#            aov_table.index = ['Weights','Trials','Interaction','Residual']
#            print('\n',model_,'\n',aov_table.round(4))
#        elif model_ == 'DecisionTreeClassifier':
#            aov_table.index = ['Feature Importance','Trials','Interaction','Residual']
#            print('\n',model_,'\n',aov_table.round(4))
#    # exp 1: awareness, confidence, correct by 1,2
#    df = pos.copy()
#    df = df[(df.window < 3) & (df.window > 0)]
#    for model_, df_sub in df.groupby('model'):
#        df_sub = post_processing(df_sub)
#        formula = 'value ~ C(Attributions) + C(window) + C(Attributions):C(window)'
#        model = ols(formula,df_sub).fit()
#        aov_table = anova_lm(model,typ=2)
#        eta_squared(aov_table)
#        omega_squared(aov_table)
#        if model_ == 'LogisticRegression':
#            aov_table.index = ['Weights','Trials','Interaction','Residual']
#            print('\n',model_,'\n',aov_table.round(4))
#        elif model_ == 'DecisionTreeClassifier':
#            aov_table.index = ['Feature Importance','Trials','Interaction','Residual']
#            print('\n',model_,'\n',aov_table.round(4))
#    
    ############################################################################
    ########## multiple comparison #############################################
    ############################################################################
    df = pos.copy()
    df = pd.concat([df[df.window == 1],
                    df[df.window == 2],
                    df[df.window == 3]])
    # this is Benjamini & Hochberg (1995) (BH or alias fdr)
    # giving you a pairwise comparison between variables
    # and it is one-tailed
    compar_pos = multiple_pairwise_comparison(df,1000,10000,method='bonferroni')
    compar_pos.to_csv(os.path.join(working_dir,'pairwise_comparision_pos (1,2,3).csv'),index=False)
    compar_pos = multiple_pairwise_comparison(df[df.window<3],1000,10000,method='bonferroni')
    compar_pos.to_csv(os.path.join(working_dir,'pairwise_comparision_pos (1,2).csv'),index=False)
    
    df = att.copy()
    df = pd.concat([df[df.window == 1],
                    df[df.window == 2],
                    df[df.window == 3]])
    compar_att = multiple_pairwise_comparison(df,1000,10000,method='bonferroni')
    compar_att.to_csv(os.path.join(working_dir,'pairwise_comparison_att (1,2,3).csv'),index=False)
    compar_att = multiple_pairwise_comparison(df[df.window<3],1000,10000,method='bonferroni')
    compar_att.to_csv(os.path.join(working_dir,'pairwise_comparison_att (1,2).csv'),index=False)
    
    ##############################################
    # plot the results of Exp 1#####################
    ########################################
    compar_pos = pd.read_csv(os.path.join(working_dir,'pairwise_comparison_att (1,2,3).csv'))
    df = compar_pos.copy()
    # this is for plotting the axices labels
    axis_map = dict(awareness=0,confidence=1,correct=2,
                    RT_awareness=3,RT_confidence=4,RT_correct=5)
    fig,axes = plt.subplots(figsize=(25,20),nrows=2,ncols=3)
    for ii,(((model,window),df_sub),ax )in enumerate(zip(df.groupby(['model','window']),axes.flatten())):
        temp = np.zeros((6,6))
        for idx,row in df_sub.iterrows():
            # positive means row variable is greater than column variable
            temp[axis_map[row.larger],axis_map[row.less]] = row['ps_corrected']
            # symmetrical to above
            temp[axis_map[row.less],axis_map[row.larger]] = -row['ps_corrected']
        # for heatmap
        temp = pd.DataFrame(temp,columns=['awareness','confidence','correct',
                                          'RT_awareness','RT_confidence','RT_correct'])
        # also for heatmap
        temp.index = temp.columns
        # plot the heatmap
        mask = np.zeros_like(temp.values)
        mask[np.triu_indices_from(mask)] = True
        ax=sns.heatmap(temp,center=True,annot=True,cbar=False,ax=ax,
                       square=True,mask=mask)
        ax.set(title='{},trials {}'.format(model,window),
               )
    fig.suptitle('Probability of Success\nAttribute Comparison\nCorrected P values\nrow - column',y=1.0)
    fig.savefig('../figures/Probability of Success, attribute comparison.png',
                bbox_inches='tight',dpi=500)
    
    # plot the results of Exp 2
    compar_att = pd.read_csv(os.path.join(working_dir,'pairwise_comparison_att (1,2,3).csv'))
    df = compar_att.copy()
    axis_map = dict(awareness=0,confidence=1,correct=2,
                    RT_awareness=3,RT_confidence=4,RT_correct=5)
    fig,axes = plt.subplots(figsize=(25,20),nrows=2,ncols=3)
    for ii,(((model,window),df_sub),ax )in enumerate(zip(df.groupby(['model','window']),axes.flatten())):
        temp = np.zeros((6,6))
        for idx,row in df_sub.iterrows():
            # positive means row variable is greater than column variable
            temp[axis_map[row.larger],axis_map[row.less]] = row['ps_corrected']
            # symmetrical to above
            temp[axis_map[row.less],axis_map[row.larger]] = -row['ps_corrected']
        # for heatmap
        temp = pd.DataFrame(temp,columns=['awareness','confidence','correct',
                                          'RT_awareness','RT_confidence','RT_correct'])
        temp.index = temp.columns
        mask = np.zeros_like(temp.values)
        mask[np.triu_indices_from(mask)] = True
        ax=sns.heatmap(temp,center=True,annot=True,cbar=False,ax=ax,
                       square=True,mask=mask)
        ax.set(title='{},trials {}'.format(model,window),
               )
    fig.suptitle('Attention\nAttribute Comparison\nCorrected P values\nrow - column',y=1.0)
    fig.savefig('../figures/Attention, attribute comparison.png',
                bbox_inches='tight',dpi=500)
    
    
    