#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:46:24 2018

@author: nmei
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')
sns.set_context('poster')
saving_dir = '../figures/'
def post_processing(df):
    feature_names = [name for name in df.columns if 'coef' in name]
    feature_name_wk = feature_names[1:]
    working_df = df[feature_name_wk]
    for name in feature_name_wk:
        working_df[name] = working_df[name].apply(np.exp)
    new_col_names = {name:name[:-5] for name in feature_name_wk}
    working_df['model'] = 'logistic'
    working_df['window'] = df['window']
    working_df = working_df.rename(new_col_names,axis='columns')
    df_plot = pd.melt(working_df,id_vars = ['model','window'],
                 value_vars = new_col_names.values())
    df_plot.columns = ['Model','Window','Coefficients','Odd Ratio']
    return df_plot
"""
Take the exponential of each of the coefficients to generate the odds ratios. 
This tells you how a 1 unit increase or decrease in a variable affects the odds of being high POS.
"""
if __name__ == '__main__':
    pos = pd.read_csv('../results/pos_logistic_statsmodel_6_features.csv')
    att = pd.read_csv('../results/att_logistic_statsmodel_6_features.csv')
    
    df = pos.copy()
    df_plot = post_processing(df)
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
    
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
#                       palette = 'RdBu',
                       data = df_plot,
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                   'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nLogistic Regression by all 6 as Features',
                y             = 1.08,
                x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Probability of success logistic_regression_6_features.png'),
              dpi = 500, bbox_inches = 'tight')
    
    df = att.copy()
    df_plot = post_processing(df)
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nLogistic Regression by all 6 as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Attention logistic_regression_6_features.png'),
              dpi = 500, bbox_inches = 'tight')
    ###################  3 judgement features #########################################
    pos = pd.read_csv('../results/pos_logistic_statsmodel_3_1_features.csv')
    att = pd.read_csv('../results/att_logistic_statsmodel_3_1_features.csv')
    
    df = pos.copy()
    df_plot = post_processing(df)
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ]:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nLogistic Regression by correct, awareness, confidence as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Probability of success logistic_regression_3_1_features.png'),
              dpi = 500, bbox_inches = 'tight')
    
    df = att.copy()
    df_plot = post_processing(df)
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ]:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nLogistic Regression by correct, awareness, confidence as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Attention logistic_regression_3_1_features.png'),
              dpi = 500, bbox_inches = 'tight')
    #############################  RT as features #################################################
    pos = pd.read_csv('../results/pos_logistic_statsmodel_RT_features.csv')
    att = pd.read_csv('../results/att_logistic_statsmodel_RT_features.csv')
    
    df = pos.copy()
    df_plot = post_processing(df)
    for n in [1,2,3]:
        for name in [
                                    
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nLogistic Regression by RT of correct, awareness, confidence Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Probability of success logistic_regression_RT_features.png'),
              dpi = 500, bbox_inches = 'tight')
    
    df = att.copy()
    df_plot = post_processing(df)
    for n in [1,2,3]:
        for name in [
                                    
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nLogistic Regression by RT of correct, awareness, confidence as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Attention logistic_regression_RT_features.png'),
              dpi = 500, bbox_inches = 'tight')
    ##########################################################################################################################################
    pos = pd.read_csv('../results/pos_logistic_statsmodel_6_features.csv')
    att = pd.read_csv('../results/att_logistic_statsmodel_6_features.csv')
    
    df = pos.copy()
    df_plot = post_processing(df)
    df_plot = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
    
    g = sns.factorplot(x = 'Window',
                   y = 'Odd Ratio',
                   hue = 'Coefficients',
                   hue_order = [
                                'correct',
                                'awareness',
                                'confidence',
                                'RT_correct',
                                'RT_awareness',
                                'RT_confidence'],
                   data = df_plot,
#                   palette = 'RdBu',
                   aspect = 3,
                   dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nLogistic Regression by all 6 as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Probability of success logistic_regression_6_features (1,2,3).png'),
              dpi = 500, bbox_inches = 'tight')
    
    df = att.copy()
    df_plot = post_processing(df)
    df_plot = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nLogistic Regression by all 6 as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Attention logistic_regression_6_features (1,2,3).png'),
              dpi = 500, bbox_inches = 'tight')
    ###################  3 judgement features #########################################
    pos = pd.read_csv('../results/pos_logistic_statsmodel_3_1_features.csv')
    att = pd.read_csv('../results/att_logistic_statsmodel_3_1_features.csv')
    
    df = pos.copy()
    df_plot = post_processing(df)
    df_plot = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ]:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nLogistic Regression by correct, awareness, confidence as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Probability of success logistic_regression_3_1_features (1,2,3).png'),
              dpi = 500, bbox_inches = 'tight')
    
    df = att.copy()
    df_plot = post_processing(df)
    df_plot = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    for n in [1,2,3]:
        for name in [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ]:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'correct',
                                    'awareness',
                                    'confidence',
                                    ],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nLogistic Regression by correct, awareness, confidence as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Attention logistic_regression_3_1_features (1,2,3).png'),
              dpi = 500, bbox_inches = 'tight')
    #############################  RT as features #################################################
    pos = pd.read_csv('../results/pos_logistic_statsmodel_RT_features.csv')
    att = pd.read_csv('../results/att_logistic_statsmodel_RT_features.csv')
    
    df = pos.copy()
    df_plot = post_processing(df)
    df_plot = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    for n in [1,2,3]:
        for name in [
                                    
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nLogistic Regression by RT of correct, awareness, confidence Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Probability of success logistic_regression_RT_features (1,2,3).png'),
              dpi = 500, bbox_inches = 'tight')
    
    df = att.copy()
    df_plot = post_processing(df)
    df_plot = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    for n in [1,2,3]:
        for name in [
                                    
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence']:
        
            std = df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].std()['Odd Ratio']
            sqrt = len(df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)])
            print('win {},{},{:.2f} {:.2f}'.format(n,name,
                  df_plot[(df_plot['Coefficients'] == name) & (df_plot['Window'] == n)].mean()['Odd Ratio'],
                  std/sqrt))
            
    g = sns.factorplot(x = 'Window',
                       y = 'Odd Ratio',
                       hue = 'Coefficients',
                       hue_order = [
                                    'RT_correct',
                                    'RT_awareness',
                                    'RT_confidence'],
                       data = df_plot,
#                       palette = 'RdBu',
                       aspect = 3,
                       dodge   = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Odd Ratio')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nLogistic Regression by RT of correct, awareness, confidence as Features',
                    y             = 1.08,
                    x             = 0.45,))
    g.savefig(os.path.join(saving_dir,'Attention logistic_regression_RT_features (1,2,3).png'),
              dpi = 500, bbox_inches = 'tight')



















































