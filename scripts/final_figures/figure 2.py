# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:48:24 2018

@author: ning
"""
from glob import glob
import os
os.chdir('..')
working_dir = ''
import pandas as pd
import utils
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
sns.set_style('whitegrid')
sns.set_context('poster')
from utils import post_processing
figure_dir = '../figures/final_figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
title_map = {'RandomForestClassifier':'Supplementary Fig','LogisticRegression':'Figure'}


##############################################################################
############################## plotting part #################################
##############################################################################
############################## using 3 features ##########################
if __name__ == '__main__':
    pos = pd.read_csv('../results/Pos_3_1_features.csv')
    att = pd.read_csv('../results/ATT_3_1_features.csv')
    pos_ttest = pd.read_csv('../results/Pos_ttest_3_1_features.csv')
    att_ttest = pd.read_csv('../results/ATT_ttest_3_1_features.csv')
    pos_one_sample_ttest = pd.read_csv('../results/one sample t test POS_3_1_features.csv')
    att_one_sample_ttest = pd.read_csv('../results/one sample t test ATT_3_1_features.csv')
    # don't work on the loaded data frame, make a copy of it
    df  = pos.copy()
    df_ttest = pos_ttest.copy()
    c   = df.groupby(['sub','model','window']).mean().reset_index()
    c   = c[(c['window'] > 0) & (c['window'] < 5)]
    for model_name,df_sub in c.groupby(['model']):
        df_ttest_sub = df_ttest[df_ttest['model'] == model_name]
        df_ttest_sub['star'] = df_ttest_sub['ps_corrected'].apply(utils.stars)
        df_ttest_sub = df_ttest_sub.sort_values(['window'])
        g   = sns.catplot(   x          = 'window',
                             y          = 'score',
                             data       = df_sub,
                             aspect     = 2.5,
                             kind       = 'bar',
                             ci         = 95,
                             dodge      = 0.1,
                             legend_out = False,)
        (g.set_axis_labels('Trial back',
                           'ROC AUC')
         .set(ylim=(0.45,0.8))
         .despine(left=True)
         .fig.axes[0].axhline(0.5,linestyle='--',color='black',alpha=0.5,))
        ax = g.axes[0][0]
        for iii,text in enumerate(df_ttest_sub['star'].values):
            ax.annotate(text,xy=(iii-0.075,0.75))
        g.fig.savefig(os.path.join(figure_dir,
                                   '{} 2.jpeg'.format(title_map[model_name])),
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
    c                             = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
    step_map={'LogisticRegression':20,
             'RandomForestClassifier':1}
    for model_name,df_sub in c.groupby(['Models']):
        df_sub
        df_ttest_sub = pos_one_sample_ttest[pos_one_sample_ttest['model_name'] == model_name]
        results_temp = utils.posthoc_multiple_comparison_interaction(
                df_sub,
                depvar = 'Values',
                unique_levels = ['Window','Attributes'],
                n_ps = 100,
                n_permutation = int(1e4),
                selected = 0)
        results_temp = utils.strip_interaction_names(results_temp)
        results_temp['stars'] = results_temp['p_corrected'].apply(utils.stars)
        position_map = {0:-0.25,1:0,2:0.225,}
        hue_map = {name:ii for ii, name in enumerate(value_vars)}
        results_temp= utils.compute_xy(results_temp,position_map,hue_map)
        
        one_sample_ttest_position = []
        for ii,row in df_ttest_sub.iterrows():
            xtick = int(row['window']) - 1
            attribute1_x = xtick + position_map[hue_map[row['attribute']]]
            row['x_pos'] = attribute1_x
            one_sample_ttest_position.append(row.to_frame().T)
        one_sample_ttest_position = pd.concat(one_sample_ttest_position)
        
        g               = sns.catplot(   x       = 'Window',
                                         y       = 'Values',
                                         hue     = 'Attributes',
                                       hue_order = value_vars,
                                         data    = df_sub,
                                         aspect  = 2.5,
                                         dodge   = 0.1,
                                         kind    = 'bar',
                                         )
        ax = g.axes[0][0]
        y_start = df_sub['Values'].mean() + df_sub['Values'].std()*3 + 0.005
        for iii,row in results_temp.iterrows():
            if "*" in row['stars']:
                ax.hlines(y_start,row['x1'],row['x2'])
                ax.vlines(row['x1'],y_start-0.005*step_map[model_name],y_start)
                ax.vlines(row['x2'],y_start-0.005*step_map[model_name],y_start)
                ax.annotate(row['stars'],
                            xy = ((row['x1']+row['x2'])/2-0.005,y_start+0.001))
                y_start += 0.05 * step_map[model_name]
        y_pos = one_sample_ttest_position['value_mean'].mean() /2
        for iii,row in one_sample_ttest_position.iterrows():
            if "*" in row['stars']:
                ax.annotate(row['stars'],
                            xy = (row['x_pos']-0.035,
                                  y_pos),
                            color = 'red')
        (g.set_axis_labels('Trial back','')
          .despine(left=True)
          )
        if model_name == 'RandomForestClassifier':
            g.set_axis_labels('Trial back','Feature Importance')
            g.set(ylim=(0.,y_start + 0.05))
        else:
            g.set_axis_labels('Trial back','Odd Ratio')
            g.set(ylim=(0.,y_start + 0.05))
        [ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for ax in g.fig.axes]
        g._legend.texts[0].set_text('correctness')
        g.savefig(os.path.join(figure_dir,'{} 3.jpeg'.format(title_map[model_name])),
                   dpi                = 500,
                   bbox_inches        = 'tight',)
    
    
    df  = att.copy()
    df_ttest = att_ttest.copy()
    c   = df.groupby(['sub','model','window']).mean().reset_index()
    c   = c[(c['window'] > 0) & (c['window'] < 5)]
    
    for model_name,df_sub in c.groupby(['model']):
        df_ttest_sub = df_ttest[df_ttest['model'] == model_name]
        df_ttest_sub['star'] = df_ttest_sub['ps_corrected'].apply(utils.stars)
        df_ttest_sub = df_ttest_sub.sort_values(['window'])
        g   = sns.catplot(   x          = 'window',
                             y          = 'score',
                             data       = df_sub,
                             aspect     = 2.5,
                             kind       = 'bar',
                             ci         = 95,
                             dodge      = 0.1,
                             legend_out = False,)
        (g.set_axis_labels('Trial back',
                           'ROC AUC')
         .set(ylim=(0.45,0.8))
         .despine(left=True)
         .fig.axes[0].axhline(0.5,linestyle='--',color='black',alpha=0.5,))
        ax = g.axes[0][0]
        for iii,text in enumerate(df_ttest_sub['star'].values):
            ax.annotate(text,xy=(iii-0.075,0.75))
        g.savefig(os.path.join(figure_dir,'{} 5.jpeg'.format(title_map[model_name])),
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
    c                             = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
    step_map={'LogisticRegression':20,
             'RandomForestClassifier':1}
    for model_name,df_sub in c.groupby(['Models']):
        df_sub
        df_ttest_sub = pos_one_sample_ttest[att_one_sample_ttest['model_name'] == model_name]
        results_temp = utils.posthoc_multiple_comparison_interaction(
                df_sub,
                depvar = 'Values',
                unique_levels = ['Window','Attributes'],
                n_ps = 100,
                n_permutation = int(1e4),
                selected = 0)
        results_temp = utils.strip_interaction_names(results_temp)
        results_temp['stars'] = results_temp['p_corrected'].apply(utils.stars)
#        position_map = {0:-0.25,1:0,2:0.2,}
        hue_map = {name:ii for ii, name in enumerate(value_vars)}
        results_temp= utils.compute_xy(results_temp,position_map,hue_map)
        
        one_sample_ttest_position = []
        for ii,row in df_ttest_sub.iterrows():
            xtick = int(row['window']) - 1
            attribute1_x = xtick + position_map[hue_map[row['attribute']]]
            row['x_pos'] = attribute1_x
            one_sample_ttest_position.append(row.to_frame().T)
        one_sample_ttest_position = pd.concat(one_sample_ttest_position)
        
        g               = sns.catplot(   x       = 'Window',
                                         y       = 'Values',
                                         hue     = 'Attributes',
                                       hue_order = value_vars,
                                         data    = df_sub,
                                         aspect  = 2.5,
                                         dodge   = 0.1,
                                         kind    = 'bar',
                                         )
        ax = g.axes[0][0]
        y_start = df_sub['Values'].mean() + df_sub['Values'].std()*3 + 0.005
        for iii,row in results_temp.iterrows():
            if "*" in row['stars']:
                ax.hlines(y_start,row['x1'],row['x2'])
                ax.vlines(row['x1'],y_start-0.005*step_map[model_name],y_start)
                ax.vlines(row['x2'],y_start-0.005*step_map[model_name],y_start)
                ax.annotate(row['stars'],
                            xy = ((row['x1']+row['x2'])/2-0.005,y_start+0.001))
                y_start += 0.05 * step_map[model_name]
        y_pos = one_sample_ttest_position['value_mean'].mean() / 2
        for iii,row in one_sample_ttest_position.iterrows():
            if "*" in row['stars']:
                ax.annotate(row['stars'],
                            xy = (row['x_pos']-0.035,
                                  y_pos),
                            color = 'red')
        (g.set_axis_labels('Trial back','')
          .despine(left=True)
          )
        if model_name == 'RandomForestClassifier':
            g.set_axis_labels('Trial back','Feature Importance')
            g.set(ylim=(0.,y_start + 0.05))
        else:
            g.set_axis_labels('Trial back','Odd Ratio')
            g.set(ylim=(0.,y_start + 0.05))
        [ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for ax in g.fig.axes]
        g._legend.texts[0].set_text('correctness')
        g.savefig(os.path.join(figure_dir,'{} 6.jpeg'.format(title_map[model_name])),
                  dpi                  = 500,
                  bbox_inches          = 'tight',)
    
    
    
    
    ###############################################################################
    ###################### plot the pos test statistics ###########################
    ###############################################################################
    
#    pos_ttest = pd.read_csv('../results/Pos_ttest_3_1_features.csv')
#    att_ttest = pd.read_csv('../results/ATT_ttest_3_1_features.csv')
#    
#    df        = pos_ttest.copy()
#    g = sns.catplot(   x            = 'window',
#                       y            = 'ps_corrected',
#                       hue          = 'model',
#                       ci           = 95,
#                       kind         = 'bar',
#                       data         = df,
#                       hue_order    = ['RandomForestClassifier','LogisticRegression'],
#                       aspect       = 3,
#                       legend_out   = False,
#                     )
#    g.set_axis_labels('Trial back',
#                       'P values (corrected)')
#    g.fig.axes[0].axhline(0.05,
#                          color = 'red',
#                    linestyle   = '--',
#                    alpha       = 0.6)
#    g.fig.suptitle('Probability of Success\nBonferroni corrected P values\nAwareness, Correctness, and Condifence as Features',
#                   y=1.14,
#                   x=0.55)
#    
#    g.savefig(os.path.join(saving_dir,'Significance test of Proabability of Success_3_1_features (1,2,3,4).jpeg'),
#              dpi               = 500,
#              bbox_inches       = 'tight')
#    
#    df = att_ttest.copy()
#    g = sns.catplot(   x            = 'window',
#                       y            = 'ps_corrected',
#                       hue          = 'model',
#                       ci           = 95,
#                       kind         = 'bar',
#                       data         = df,
#                       hue_order    = ['RandomForestClassifier','LogisticRegression'],
#                       aspect       = 3,
#                       legend_out   = False,
#                     )
#    g.set_axis_labels('Trial',
#                       'P values (corrected)')
#    g.fig.axes[0].axhline(0.05,
#              color              = 'red',
#              linestyle          = '--',
#              alpha              = 0.6)
#    g.fig.suptitle('Attention\nBonferroni corrected P values\nAwareness, Correctness, and Condifence as Features',
#                   y=1.14,
#                   x=0.55)
#    
#    g.savefig(os.path.join(saving_dir,'Significance test of Attention_3_1_features (1,2,3,4).jpeg'),
#              dpi                = 500,
#              bbox_inches        = 'tight')
    
    
    
    
    
    
    
    
    
