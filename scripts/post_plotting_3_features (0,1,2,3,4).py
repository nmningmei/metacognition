# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:48:24 2018

@author: ning
"""
from glob import glob
import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
sns.set_style('whitegrid')
sns.set_context('poster')
from utils import post_processing
saving_dir = '../figures/'
chance_map = {'RandomForestClassifier':'RF_chance','LogisticRegression':'LR_chance'}


##############################################################################
############################## plotting part #################################
##############################################################################
############################## using 3 features ##########################
if __name__ == '__main__':
    pos = pd.read_csv('../results/Pos_3_1_features.csv')
    att = pd.read_csv('../results/ATT_3_1_features.csv')
    # don't work on the loaded data frame, make a copy of it
    df  = pos.copy()
#    df_chance = pd.concat([pd.read_csv(f) for f in glob('../results/chance/Pos_3*.csv')])
#    df_chance = df_chance.groupby(['sub','model','window',]).mean().reset_index()
#    df_chance['model'] = df_chance['model'].map(chance_map)
    c   = df.groupby(['sub','model','window']).mean().reset_index()
#    c   = pd.concat([c,df_chance])
    c   = c[(c['window'] > 0) & (c['window'] < 5)]
    g   = sns.catplot(   x          = 'window',
                         y          = 'score',
                         hue        = 'model',
                         data       = c,
                         aspect     = 3,
                         kind       = 'bar',
                         hue_order  = ['RandomForestClassifier',
#                                       'RF_chance',
                                       'LogisticRegression',
#                                       'LR_chance'
                                       ],
#                         palette    = "RdBu",
                         ci         = 95,
                         dodge      = 0.1,
                         legend_out = False,)
    (g.set_axis_labels('Trial back',
                       'ROC AUC')
     .set(ylim=(0.45,0.8))
     .despine(left=True)
     .fig.axes[0].axhline(0.5,linestyle='--',color='black',alpha=0.5,))
#    g.fig.suptitle('Model Comparison of Decoding \nProbability of Success\nAwareness, Correctness, and Condifence as Features',
#     y=1.15,x=0.5)
    g.axes[0][0].get_legend().set_title('')
    g.fig.savefig(os.path.join(saving_dir,
                               'Model Comparison of Decoding Probability of Success_3_1_features (1,2,3,4).png'),
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
                      ]
    df_post                       = post_processing(df[(df['window'] >0) & (df['window'] < 5)],
                                                    id_vars,value_vars)
#    df_post.fillna(1)
    from sklearn.preprocessing import scale
    df_post.loc[df_post['Models'] == 'RandomForestClassifier','Values'] = scale(df_post.loc[df_post['Models'] == 'RandomForestClassifier','Values'].values)
    c                             = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
    g               = sns.catplot(   x       = 'Window',
                                     y       = 'Values',
                                     hue     = 'Attributes',
                                   hue_order = value_vars,
                                     row     = 'Models',
                                   row_order = ['RandomForestClassifier','LogisticRegression'],
                                     data    = c,
                                     aspect  = 3,
                                     sharey  = False,
                                     dodge   = 0.1,
                                     kind    = 'bar',
                                     )
    (g.set_axis_labels('Trial back',
                       '')
      .set_titles('{row_name}')
      .despine(left=True)
#      .fig.suptitle('Probability of Success\nAwareness, Correctness, and Condifence as Features',
#                    y             = 1.05,
#                    x             = 0.45,)
      )
    g.fig.axes[0].set(ylabel='Normalized Feature Importance',ylim=(0.3,0.36))
#    g.fig.axes[0].axhline(1./len(value_vars),color='red',linestyle='--',alpha=0.6)
    g.fig.axes[1].set(ylabel='Odd Ratio')
#    g.fig.axes[1].axhline(1.,color='red',linestyle='--',alpha=0.6)
    [ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for ax in g.fig.axes]
    g._legend.set_title('')
    g._legend.texts[0].set_text('correctness')
    g.savefig(os.path.join(saving_dir,'Weights plot of Probability of Success_3_1_features(1,2,3,4).png'),
               dpi                = 500,
               bbox_inches        = 'tight',)
    
    
    
    
    
    # exp 2
    df  = att.copy()
#    df_chance = pd.concat([pd.read_csv(f) for f in glob('../results/chance/att_3*.csv')])
#    df_chance = df_chance.groupby(['sub','model','window',]).mean().reset_index()
#    df_chance['model'] = df_chance['model'].map(chance_map)
    c   = df.groupby(['sub','model','window']).mean().reset_index()
#    c   = pd.concat([c,df_chance])
    c   = c[(c['window'] > 0) & (c['window'] < 5)]
    g   = sns.catplot(   x          = 'window',
                         y          = 'score',
                         hue        = 'model',
                         data       = c,
                         aspect     = 3,
                         kind       = 'bar',
                         hue_order  = ['RandomForestClassifier',
#                                       'RF_chance',
                                       'LogisticRegression',
#                                       'LR_chance'
                                       ],
#                         palette    = "RdBu",
                         ci         = 95,
                         dodge      = 0.1,
                         legend_out = False,)
    (g.set_axis_labels('Trial back',
                       'ROC AUC')
     .set(ylim=(0.45,0.8))
     .despine(left=True)
     .fig.axes[0].axhline(0.5,linestyle='--',color='black',alpha=0.5,))
#    g.fig.suptitle('Model Comparison of Decoding \nAttention\nAwareness, Correctness, and Condifence as Features',
#     y=1.15,x=0.5)
    g.axes[0][0].get_legend().set_title('')
    g.savefig(os.path.join(saving_dir,'Model Comparison of Decoding Attention_3_1_features(1,2,3,4).png'),
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
                      ]
    df_post                       = post_processing(df[(df['window'] >0) & (df['window'] < 5)],
                                                    id_vars,value_vars)
#    df_post.fillna(1)
    c                             = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    g               = sns.catplot(   x       = 'Window',
                                     y       = 'Values',
                                     hue     = 'Attributes',
                                   hue_order = value_vars,
                                     row     = 'Models',
                                   row_order = ['RandomForestClassifier','LogisticRegression'],
                                     data    = c,
                                     aspect  = 3,
                                     sharey  = False,
                                     dodge   = 0.1,
                                     kind    = 'bar',
                                     )
    (g.set_axis_labels('Trial back',
                       '')
      .set_titles('{row_name}')
      .despine(left=True)
#      .fig.suptitle('Attention\nAwareness, Correctness, and Condifence as Features',
#                    y             = 1.05,
#                    x             = 0.45,)
      )
    g.fig.axes[0].set(ylabel='Feature Importance',ylim=(0.3,0.36))
#    g.fig.axes[0].axhline(1./len(value_vars),color='red',linestyle='--',alpha=0.6)
    g.fig.axes[1].set(ylabel='Odd Ratio')
#    g.fig.axes[1].axhline(1.,color='red',linestyle='--',alpha=0.6)
    [ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for ax in g.fig.axes]
    g._legend.set_title('')
    g._legend.texts[0].set_text('correctness')
    g.savefig(os.path.join(saving_dir,'Weights plot of Attention_3_1_features(1,2,3,4).png'),
              dpi                  = 500,
              bbox_inches          = 'tight',)
    
    
    
    
    ###############################################################################
    ###################### plot the pos test statistics ###########################
    ###############################################################################
    
    pos_ttest = pd.read_csv('../results/Pos_ttest_3_1_features.csv')
    att_ttest = pd.read_csv('../results/ATT_ttest_3_1_features.csv')
    
    df        = pos_ttest.copy()
    g = sns.catplot(   x            = 'window',
                       y            = 'ps_corrected',
                       hue          = 'model',
                       ci           = 95,
                       kind         = 'bar',
                       data         = df,
                       hue_order    = ['RandomForestClassifier','LogisticRegression'],
                       aspect       = 3,
                       legend_out   = False,
                     )
    g.set_axis_labels('Trial back',
                       'P values (corrected)')
    g.fig.axes[0].axhline(0.05,
                          color = 'red',
                    linestyle   = '--',
                    alpha       = 0.6)
    g.fig.suptitle('Probability of Success\nBonferroni corrected P values\nAwareness, Correctness, and Condifence as Features',
                   y=1.14,
                   x=0.55)
    
    g.savefig(os.path.join(saving_dir,'Significance test of Proabability of Success_3_1_features (1,2,3,4).png'),
              dpi               = 500,
              bbox_inches       = 'tight')
    
    df = att_ttest.copy()
    g = sns.catplot(   x            = 'window',
                       y            = 'ps_corrected',
                       hue          = 'model',
                       ci           = 95,
                       kind         = 'bar',
                       data         = df,
                       hue_order    = ['RandomForestClassifier','LogisticRegression'],
                       aspect       = 3,
                       legend_out   = False,
                     )
    g.set_axis_labels('Trial',
                       'P values (corrected)')
    g.fig.axes[0].axhline(0.05,
              color              = 'red',
              linestyle          = '--',
              alpha              = 0.6)
    g.fig.suptitle('Attention\nBonferroni corrected P values\nAwareness, Correctness, and Condifence as Features',
                   y=1.14,
                   x=0.55)
    
    g.savefig(os.path.join(saving_dir,'Significance test of Attention_3_1_features (1,2,3,4).png'),
              dpi                = 500,
              bbox_inches        = 'tight')
    
    
    
    
    
    
    
    
    
