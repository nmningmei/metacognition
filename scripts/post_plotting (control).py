# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:48:24 2018

@author: ning
"""
import os
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
from utils import post_processing2
saving_dir = '../figures/control'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)


##############################################################################
############################## plotting part #################################
##############################################################################
if __name__ == '__main__':
    pos = pd.read_csv('../results/Pos_control.csv')
    att = pd.read_csv('../results/ATT_control.csv')
    # don't work on the loaded data frame, make a copy of it
    df  = pos.copy()
    g   = sns.factorplot(x         = 'window',
                         y         = 'score',
                         hue       = 'model',
                         data      = df,
                         hue_order = ['DecisionTreeClassifier','LogisticRegression'],
                         aspect    = 3,
                         dodge     = 0.1)
    # for seaborn 0.9.0
    #g = sns.catplot(     x       = 'window',
    #                     y       = 'score',
    #                     hue     = 'model',
    #                     data    = df,
    #                     aspect  = 3,
    #                     kind    = 'point',
#    hue_order = ['DecisionTreeClassifier','LogisticRegression'],
    #                     ci      = 95
#                          dodge     = 0.1)
    (g.set_axis_labels('Trials look back',
                       'Clasifi.Score (AUC ROC)')
     .fig.suptitle('Model Comparison of Decoding Trial Correctness\n Exp: Probability of Success',
                   y=1.05))
    g.fig.savefig(os.path.join(saving_dir,
                               'Model Comparison of Decoding Probability of Success.png'),
              dpi                 = 500,
              bbox_inches         = 'tight')
    ###### process the data a little bit
    df                            = post_processing2(df,names=['success','awareness','confidence'])
    g = sns.factorplot(   x       = 'window',
                          y       = 'value',
                          hue     = 'Attributions',
                          row     = 'model',
                          row_order=['DecisionTreeClassifier','LogisticRegression'],
                          data    = df,
                          aspect  = 3,
                          sharey  = False,
                          dodge   = 0.1)
    # for seaborn 0.9.0
    #g = sns.catplot(      x       = 'window',
    #                      y       = 'value',
    #                      hue     = 'Attributions',
    #                      row     = 'model',
    #                      data    = df,
    #                      aspect  = 3,
    #                      sharey  = False,
    #                      kind    = 'point',
    #                      ci      = 95)
    (g.set_axis_labels('Trials look back',
                       '')
      .set_titles('{row_name}')
      .fig.suptitle('Predict: Trial Correctness by (awareness,confidence,POS)\nExp: Probability of Success',
                    y             = 1.05))
    g.fig.axes[0].set(ylabel='Feature Importance')
    g.fig.axes[1].set(ylabel='Coefficients')
    g.savefig(os.path.join(saving_dir,'Weights plot of Probability of Success.png'),
               dpi                = 500,
               bbox_inches        = 'tight',)
    
    
    df = att.copy()
    g = sns.factorplot(   x       = 'window',
                          y       = 'score',
                          hue     = 'model',
                          data    = df,
                          hue_order = ['DecisionTreeClassifier','LogisticRegression'],
                          aspect  = 3,
                          dodge   = 0.1)
    # for seaborn 0.9.0
    #g = sns.catplot(      x       = 'window',
    #                      y       = 'score',
    #                      hue     = 'model',
    #                      data    = df,
    #                      aspect  = 3,
    #                      kind    = 'point',
#    hue_order = ['DecisionTreeClassifier','LogisticRegression'],
    #                      ci      = 95)
    (g.set_axis_labels('Trials look back',
                       'Clasifi.Score (AUC ROC)')
     .fig.suptitle('Model Comparison of Decoding Trial Correctness\n Exp: Attention'))
    g.savefig(os.path.join(saving_dir,'Model Comparison of Decoding Attention.png'),
              dpi                 = 500,
              bbox_inches         = 'tight')
    
    df                            = post_processing2(df,names=['attention','awareness','confidence'])
    g = sns.factorplot(    x      = 'window',
                           y      = 'value',
                           hue    = 'Attributions',
                           row    = 'model',
                           row_order=['DecisionTreeClassifier','LogisticRegression'],
                           data   = df,
                           aspect = 3,
                           sharey = False,
                           dodge  = 0.1)
    # for seaborn 0.9.0
    #g = sns.catplot(       x      = 'window',
    #                       y      = 'value',
    #                       hue    = 'Attributions',
    #                       row    = 'model',
    #                       data   = df,
    #                       aspect = 3,
    #                       sharey = False,
    #                       kind   = 'point',
    #                       ci     = 95)
    (g.set_axis_labels('Trials look back',
                       '')
      .set_titles('{row_name}')
      .fig.suptitle('Predict: Trial Correctness by (awareness,confidence,attention)\nExp: Attention',
                    y              = 1.05))
    g.fig.axes[0].set(ylabel='Feature Importance')
    g.fig.axes[1].set(ylabel='Coefficients')
    g.savefig(os.path.join(saving_dir,'Weights plot of Attention.png'),
              dpi                  = 500,
              bbox_inches          = 'tight',)
    
    
    ###############################################################################
    ###################### plot the normalized weights ############################
    ###############################################################################
    
    pos_ttest = pd.read_csv('../results/Pos_ttest_control.csv')
    att_ttest = pd.read_csv('../results/ATT_ttest_control.csv')
    
    df        = pos_ttest.copy()
    g = sns.factorplot(x       = 'window',
                       y       = 'ps_mean',
                       hue     = 'model',
                       ci      = None,
                       kind    = 'bar',
                       data    = df,
                       hue_order = ['DecisionTreeClassifier','LogisticRegression'],
                       aspect  = 2.5,
#                       dodge   = 0.1
                       )
    # for seaborn 0.9.0
    #g = sns.catplot(   x       = 'window',
    #                   y       = 'ps_mean',
    #                   hue     = 'model',
    #                   ci      = None,
    #                   kind    = 'bar',
    #                   data    = df,
    #                   aspect  = 2.5,
    #                 )
    g.set_axis_labels('Trials look back',
                       'Mean of P values (corrected)')
    g.fig.axes[0].axhline(0.05,
                          color = 'red',
                    linestyle   = '--',
                    alpha       = 0.6)
    g.fig.suptitle('Predict: Trial Correctness by (awareness,confidence,POS)\nExp: Probability of Success\nBonferroni corrected P values',
                   y=1.15)
    
    g.savefig(os.path.join(saving_dir,'Significance test of Proabability of Success.png'),
              dpi               = 500,
              bbox_inches       = 'tight')
    
    df = att_ttest.copy()
    g = sns.factorplot(x        = 'window',
                       y        = 'ps_mean',
                       hue      ='model',
                       ci       = None,
                       kind     = 'bar',
                       data     = df,
                       hue_order = ['DecisionTreeClassifier','LogisticRegression'],
                       aspect   = 2.5,
#                       dodge    = 0.1
                       )
    # for seaborn 0.9.0
    #g = sns.catplot(   x        = 'window',
    #                   y        = 'ps_mean',
    #                   hue      = 'model',
    #                   ci       = None,
    #                   kind     = 'bar',
    #                   data     =df,
    #                   aspect   = 2.5,
    #                 )
    g.set_axis_labels('Trials look back',
                       'Mean of P values (corrected)')
    g.fig.axes[0].axhline(0.05,
              color              = 'red',
              linestyle          = '--',
              alpha              = 0.6)
    g.fig.suptitle('Predict: Trial Correctness by (awareness,confidence,attention)\nExp: Attention\nBonferroni corrected P values',
                   y             = 1.15)
    
    g.savefig(os.path.join(saving_dir,'Significance test of Attention.png'),
              dpi                = 500,
              bbox_inches        = 'tight')
    

        
    
    
    
    
    
    
    
    
    
