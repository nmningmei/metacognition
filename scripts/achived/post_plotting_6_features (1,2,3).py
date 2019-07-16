# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:48:24 2018

@author: ning
"""
import os
from glob import glob
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold,permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
sns.set_style('whitegrid')
sns.set_context('poster')
from utils import post_processing
saving_dir = '../figures/'
chance_map = {'DecisionTreeClassifier':'DTC_chance','LogisticRegression':'LR_chance'}


##############################################################################
############################## plotting part #################################
##############################################################################
############################## using 3 features ##########################
if __name__ == '__main__':
    pos = pd.read_csv('../results/Pos_6_features.csv')
    att = pd.read_csv('../results/ATT_6_features.csv')
    # don't work on the loaded data frame, make a copy of it
    df  = pos.copy()
    df_chance = pd.concat([pd.read_csv(f) for f in glob('../results/chance/Pos_6*.csv')])
    df_chance = df_chance.groupby(['sub','model','window',]).mean().reset_index()
    df_chance['model'] = df_chance['model'].map(chance_map)
    c   = df.groupby(['sub','model','window']).mean().reset_index()
    c   = pd.concat([c,df_chance])
    c   = c[(c['window']>0) & (c['window'] < 4)]
    g   = sns.catplot(   x          = 'window',
                         y          = 'score',
                         hue        = 'model',
                         data       = c,
                         aspect     = 3,
                         kind       = 'point',
                         hue_order  = ['DecisionTreeClassifier',
                                       'DTC_chance',
                                       'LogisticRegression',
                                       'LR_chance'],
                         palette    = "RdBu",
                         ci         = 95,
                         dodge      = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Clasifi.Score (AUC ROC)')
     .set(ylim=(0.45,0.75))
     )
    g.fig.suptitle('Model Comparison of Decoding Probability of Success\nall 6 as Features',
     y=1.02,x=0.45)
    g.fig.savefig(os.path.join(saving_dir,
                               'Model Comparison of Decoding Probability of Success_6_features(1,2,3).png'),
              dpi                 = 500,
              bbox_inches         = 'tight')
    
    
    id_vars         = ['model',
                       'score',
                       'sub',
                       'window',]
    value_vars      =[
                      'correct',
                      'awareness',
                      'confidence',
                      'RT_correct',
                      'RT_awareness',
                      'RT_confidence',
                      ]
    df_post                       = post_processing(df,id_vars,value_vars)
    c                             = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                             = c[(c['Window']>0) & (c['Window'] < 4)]
    g               = sns.catplot(   x       = 'Window',
                                     y       = 'Values',
                                     hue     = 'Attributes',
                                     row     = 'Models',
                                     row_order=['DecisionTreeClassifier','LogisticRegression'],
                                     hue_order=value_vars,
#                                     palette  = "RdBu",
                                     data    = c,
                                     aspect  = 3,
                                     sharey  = False,
                                     dodge   = 0.1,
                                     kind    = 'point')
    (g.set_axis_labels('Trials look back',
                       '')
      .set_titles('{row_name}')
      .fig.suptitle('Probability of Success\nall 6 as Features',
                    y             = 1.05,
                    x             = 0.45,))
    g.fig.axes[0].set(ylabel='Feature Importance')
    g.fig.axes[1].set(ylabel='Coefficients')
    g.savefig(os.path.join(saving_dir,'Weights plot of Probability of Success_6_features(1,2,3).png'),
               dpi                = 500,
               bbox_inches        = 'tight',)
    
    
    df = att.copy()
    df_chance = pd.concat([pd.read_csv(f) for f in glob('../results/chance/att_6*.csv')])
    df_chance = df_chance.groupby(['sub','model','window',]).mean().reset_index()
    df_chance['model'] = df_chance['model'].map(chance_map)
    c   = df.groupby(['sub','model','window']).mean().reset_index()
    c   = pd.concat([c,df_chance])
    c   = c[(c['window']>0) & (c['window'] < 4)]
    g = sns.catplot(      x         = 'window',
                          y         = 'score',
                          hue       = 'model',
                          data      = c,
                          aspect    = 3,
                          kind      = 'point',
                          palette   = "RdBu",
                          hue_order = ['DecisionTreeClassifier',
                                       'DTC_chance',
                                       'LogisticRegression',
                                       'LR_chance'],
                          ci        = 95,
                          dodge     = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Clasifi.Score (AUC ROC)')
     .set(ylim=(0.45,0.75))
     )
    g.fig.suptitle('Model Comparison of Decoding Attention\nall 6 Features',
     y=1.02,x=0.45)
    g.savefig(os.path.join(saving_dir,'Model Comparison of Decoding Attention_6_features(1,2,3).png'),
              dpi                 = 500,
              bbox_inches         = 'tight')
    
    id_vars         = ['model',
                       'score',
                       'sub',
                       'window',]
    value_vars      =[
                      'correct',
                      'awareness',
                      'confidence',
                      'RT_correct',
                      'RT_awareness',
                      'RT_confidence'
                      ]
    df_post                         = post_processing(df,id_vars,value_vars)
    c                               = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                               = c[(c['Window']>0) & (c['Window'] < 4)]
    g = sns.catplot(       x        = 'Window',
                           y        = 'Values',
                           hue      = 'Attributes',
                        hue_order   = value_vars,
                           row      = 'Models',
                           data     = c,
                           aspect   = 3,
#                           palette  = "RdBu",
                           row_order=['DecisionTreeClassifier','LogisticRegression'],
                           sharey   = False,
                           kind     = 'point',
                           ci       = 95,
                           dodge    = 0.1)
    (g.set_axis_labels('Trials look back',
                       '')
      .set_titles('{row_name}')
      .fig.suptitle('Attention\nall 6 Features',
                    y             = 1.05,
                    x             = 0.45,))
    g.fig.axes[0].set(ylabel='Feature Importance')
    g.fig.axes[1].set(ylabel='Coefficients')
    g.savefig(os.path.join(saving_dir,'Weights plot of Attention_6_features(1,2,3).png'),
              dpi                  = 500,
              bbox_inches          = 'tight',)
    
    
    
    

        
    
    
    
    
    
    
    
    
    
