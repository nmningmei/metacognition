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
from statannot import add_stat_annotation
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
    df_both = []
    df_both_test = []
    # don't work on the loaded data frame, make a copy of it
    df  = pos.copy()
    df_ttest = pos_ttest.copy()
    c   = df.groupby(['sub','model','window']).mean().reset_index()
    c   = c[(c['window'] > 0) & (c['window'] < 5)]
    c['experiment'] = 'Exp.1.'
    df_ttest['star']    = df_ttest['ps_corrected'].apply(utils.stars)
    df_ttest            = df_ttest.sort_values(['model','window'])
    df_ttest['experiment'] = 'Exp.1.'
    df_both.append(c)
    df_both_test.append(df_ttest)
    
    df  = att.copy()
    df_ttest = att_ttest.copy()
    c   = df.groupby(['sub','model','window']).mean().reset_index()
    c   = c[(c['window'] > 0) & (c['window'] < 5)]
    c['experiment'] = 'Exp.2.'
    df_ttest['star']    = df_ttest['ps_corrected'].apply(utils.stars)
    df_ttest            = df_ttest.sort_values(['model','window'])
    df_ttest['experiment'] = 'Exp.2.'
    df_both.append(c)
    df_both_test.append(df_ttest)
    
    df_both = pd.concat(df_both)
    df_both_test = pd.concat(df_both_test)
    
    g   = sns.catplot(   x          = 'window',
                         y          = 'score',
                         hue        = 'model',
                         row        = 'experiment',
                         data       = df_both,
                         aspect     = 2.5,
                         kind       = 'bar',
                         ci         = 95,
                         dodge      = 0.1,
                         legend_out = True,)
    (g.set_axis_labels('Trial back',
                       'ROC AUC')
      .set_titles("{row_name}")
     .set(ylim=(0.45,0.8))
     .despine(left=True))
    [ax.axhline(0.5,linestyle='--',color='black',alpha=0.5,) for ax in g.axes.flatten()]
    
    ax = g.axes[0][0]
    df_ttest = df_both_test[df_both_test['experiment'] == 'Exp.1.']
    for _, df_sub in df_ttest.groupby(['model']):
        for iii,text in enumerate(df_sub['star'].values):
            ax.annotate(text,xy=(iii-0.075,0.75))
    
    ax = g.axes[1][0]
    df_ttest = df_both_test[df_both_test['experiment'] == 'Exp.2.']
    for _, df_sub in df_ttest.groupby(['model']):
        for iii,text in enumerate(df_sub['star'].values):
            ax.annotate(text,xy=(iii-0.075,0.75))
    
    g.fig.savefig(os.path.join(figure_dir,
                               'ccn fig1.png'),
#              dpi                 = 500,
              bbox_inches         = 'tight')
    
    
    pos = pd.read_csv('../results/Pos_3_1_features.csv')
    att = pd.read_csv('../results/ATT_3_1_features.csv')
    pos_ttest = pd.read_csv('../results/Pos_ttest_3_1_features.csv')
    att_ttest = pd.read_csv('../results/ATT_ttest_3_1_features.csv')
    df_both,df_both_test = [],[]
    df = pos.copy()
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
    c['experiment'] = 'Exp.1.'
    temp = []
    for model_name,df_sub in c.groupby(['Models']):
        df_sub
        results_temp = utils.posthoc_multiple_comparison_interaction(
                df_sub,
                depvar = 'Values',
                unique_levels = ['Window','Attributes'],
                n_ps = 100,
                n_permutation = int(1e4),
                selected = 0)
        results_temp = utils.strip_interaction_names(results_temp)
        results_temp['stars'] = results_temp['p_corrected'].apply(utils.stars)
        results_temp['model'] = model_name
        temp.append(results_temp)
    temp = pd.concat(temp)
    df_stats = temp.sort_values(['model','window'])
    df_stats['experiment'] = 'Exp.1.'
    
    df_both.append(c)
    df_both_test.append(df_stats)
    
    
    df = att.copy()
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
    c['experiment'] = 'Exp.2.'
    temp = []
    for model_name,df_sub in c.groupby(['Models']):
        df_sub
        results_temp = utils.posthoc_multiple_comparison_interaction(
                df_sub,
                depvar = 'Values',
                unique_levels = ['Window','Attributes'],
                n_ps = 100,
                n_permutation = int(1e4),
                selected = 0)
        results_temp = utils.strip_interaction_names(results_temp)
        results_temp['stars'] = results_temp['p_corrected'].apply(utils.stars)
        results_temp['model'] = model_name
        temp.append(results_temp)
    temp = pd.concat(temp)
    df_stats = temp.sort_values(['model','window'])
    df_stats['experiment'] = 'Exp.2.'
    
    df_both.append(c)
    df_both_test.append(df_stats)
    
    df_both = pd.concat(df_both)
    df_both_test = pd.concat(df_both_test)
    
    df_both = df_both.sort_values(['experiment','Models','Window','Attributes'])
    df_both_test = df_both_test.sort_values(['experiment','model','window'])
    
    step_map={'LogisticRegression':20,
             'RandomForestClassifier':1}
    
    g               = sns.catplot(   x       = 'Window',
                                     y       = 'Values',
                                     hue     = 'Attributes',
                                     row     = 'Models',
                                   hue_order = value_vars,
                                     col     = 'experiment',
                                   col_order = ['Exp.1.','Exp.2.'],
                                     data    = df_both,
                                     aspect  = 2.5,
                                     dodge   = 0.1,
                                     kind    = 'bar',
                                     sharey  = False,
                                     )
    (g.set_axis_labels('Trial back','')
      .set_titles("{col_name} | {row_name}")
      .despine(left=True)
      )
    ax = g.axes[0][0]
    c = df_both[df_both['experiment'] == 'Exp.1.']
    df_stats = df_both_test[df_both_test['experiment'] == 'Exp.1.']
    ax.set(ylabel='Odd Ratio')
    position_map = {0:-0.25,1:0,2:0.25,}
    hue_map = {name:ii for ii, name in enumerate(value_vars)}
    results_temp = utils.compute_xy(df_stats[df_stats['model'] == 'LogisticRegression'],
                                    position_map,hue_map)
    df_sub = c[c['Models'] == 'LogisticRegression']
    y_start = df_sub['Values'].mean() + df_sub['Values'].std()*3 + 0.005
    for iii,row in results_temp.iterrows():
        if "*" in row['stars']:
            ax.hlines(y_start,row['x1'],row['x2'])
            ax.vlines(row['x1'],y_start-0.05*step_map[model_name],y_start)
            ax.vlines(row['x2'],y_start-0.05*step_map[model_name],y_start)
            ax.annotate(row['stars'],
                        xy = ((row['x1']+row['x2'])/2-0.005,y_start+0.001))
            y_start += 0.5 * step_map[model_name]
    
    ax = g.axes[1][0]
    ax.set(ylabel='Feature Importance')
    c = df_both[df_both['experiment'] == 'Exp.1.']
    df_stats = df_both_test[df_both_test['experiment'] == 'Exp.1.']
    position_map = {0:-0.25,1:0,2:0.25,}
    hue_map = {name:ii for ii, name in enumerate(value_vars)}
    results_temp = utils.compute_xy(df_stats[df_stats['model'] == 'RandomForestClassifier'],
                                    position_map,hue_map)
    df_sub = c[c['Models'] == 'RandomForestClassifier']
    y_start = df_sub['Values'].mean() + df_sub['Values'].std()*3 + 0.005
    for iii,row in results_temp.iterrows():
        if "*" in row['stars']:
            ax.hlines(y_start,row['x1'],row['x2'])
            ax.vlines(row['x1'],y_start-0.005*step_map[model_name],y_start)
            ax.vlines(row['x2'],y_start-0.005*step_map[model_name],y_start)
            ax.annotate(row['stars'],
                        xy = ((row['x1']+row['x2'])/2-0.005,y_start+0.001))
            y_start += 0.05 * step_map[model_name]
    
    ax = g.axes[0][1]
    c = df_both[df_both['experiment'] == 'Exp.2.']
    df_stats = df_both_test[df_both_test['experiment'] == 'Exp.2.']
    position_map = {0:-0.25,1:0,2:0.25,}
    hue_map = {name:ii for ii, name in enumerate(value_vars)}
    results_temp = utils.compute_xy(df_stats[df_stats['model'] == 'LogisticRegression'],
                                    position_map,hue_map)
    df_sub = c[c['Models'] == 'LogisticRegression']
    y_start = df_sub['Values'].mean() + df_sub['Values'].std()*3 + 0.005
    for iii,row in results_temp.iterrows():
        if "*" in row['stars']:
            ax.hlines(y_start,row['x1'],row['x2'])
            ax.vlines(row['x1'],y_start-0.05*step_map[model_name],y_start)
            ax.vlines(row['x2'],y_start-0.05*step_map[model_name],y_start)
            ax.annotate(row['stars'],
                        xy = ((row['x1']+row['x2'])/2-0.005,y_start+0.001))
            y_start += 0.05 * step_map[model_name]
    
    ax = g.axes[1][1]
    c = df_both[df_both['experiment'] == 'Exp.2.']
    df_stats = df_both_test[df_both_test['experiment'] == 'Exp.2.']
    position_map = {0:-0.25,1:0,2:0.25,}
    hue_map = {name:ii for ii, name in enumerate(value_vars)}
    results_temp = utils.compute_xy(df_stats[df_stats['model'] == 'RandomForestClassifier'],
                                    position_map,hue_map)
    df_sub = c[c['Models'] == 'RandomForestClassifier']
    y_start = df_sub['Values'].mean() + df_sub['Values'].std()*3 + 0.005
    for iii,row in results_temp.iterrows():
        if "*" in row['stars']:
            ax.hlines(y_start,row['x1'],row['x2'])
            ax.vlines(row['x1'],y_start-0.005*step_map[model_name],y_start)
            ax.vlines(row['x2'],y_start-0.005*step_map[model_name],y_start)
            ax.annotate(row['stars'],
                        xy = ((row['x1']+row['x2'])/2-0.005,y_start+0.001))
            y_start += 0.05 * step_map[model_name]
    
    
    [ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for ax in g.fig.axes]
    g._legend.texts[0].set_text('correctness')
    g.savefig(os.path.join(figure_dir,'ccn fig2.png'),
#                   dpi                = 500,
                   bbox_inches        = 'tight',)
    
    
    working_dir = '../results/cross_experiment_generalization'
    df = pd.read_csv(os.path.join(working_dir,'cross experiment generalization (individual level).csv'))
    df_plot = df.groupby(['window',
                          'model',
                          'participant',
                          'experiment_test',
                          'experiment_train']).mean().reset_index()
    df_plot = df_plot[(df_plot['window'] > 0) & (df_plot['window'] < 5)]
    
    cols = ['model', 'pval', 'score_mean', 'score_std', 'train', 'window',]
    df_post = {name:[] for name in cols}
    for (train,model_name,window),df_sub in df_plot.groupby(['experiment_train','model','window']):
        df_sub
        ps = utils.resample_ttest(df_sub['score'].values,
                                  0.5,
                                  n_ps = 100,
                                  n_permutation = int(1e4),
                                  one_tail = True)
        df_post['model'].append(model_name)
        df_post['window'].append(window)
        df_post['train'].append(train)
        df_post['score_mean'].append(df_sub['score'].values.mean())
        df_post['score_std'].append(df_sub['score'].values.std())
        df_post['pval'].append(ps.mean())
    df_post = pd.DataFrame(df_post)
    temp = []
    for (train,model_name), df_sub in df_post.groupby(['train','model']):
        df_sub = df_sub.sort_values(['pval'])
        ps = df_sub['pval'].values
        converter = utils.MCPConverter(pvals = ps)
        d = converter.adjust_many()
        df_sub['p_corrected'] = d['bonferroni'].values
        temp.append(df_sub)
    df_post = pd.concat(temp)
    
    df_post['star'] = df_post['p_corrected'].apply(utils.stars)
    df_post = df_post.sort_values(['model','train','window',])
    df_post = df_post[df_post['window'] > 0]
    
    df_plot['experiment_train'] = df_plot['experiment_train'].map({'POS':"Trained on Exp.1/ Tested on Exp.2",
                                                                   'ATT':"Trained on Exp.2/ Tested on Exp.1"})
    df_plot = df_plot[df_plot['window'] > 0]
    
    
    g = sns.catplot(x = 'window',
                    y = 'score',
                    row = 'experiment_train',
                    hue = 'model',
                    data = df_plot,
                    kind = 'bar',
                    aspect = 2.5,
                    row_order = ['Trained on Exp.1/ Tested on Exp.2',
                                 'Trained on Exp.2/ Tested on Exp.1'],
                   legend_out = True)
    
    (g.set_axis_labels('Trial back','ROC AUC')
      .set(ylim=(0.45,0.65))
      .despine(left=True)
      .set_titles("{row_name}"))
    [ax.axhline(0.5,linestyle='--',color='black',alpha=0.5) for ax in g.fig.axes]
    
    for idx_ax, exp in enumerate(['POS','ATT']):
        ax = g.axes.flatten()[idx_ax]
        
        df_sub = df_post[df_post['train'] == exp]
        df_sub['position'] = np.concatenate([np.arange(4)-0.25,np.arange(4)+0.15])
        for ii,row in df_sub.iterrows():
            ax.annotate(row['star'],xy=(row['position'],0.6))
    g.savefig(os.path.join(figure_dir,'fig4.png'),
              dpi = 500,
              bbox_inches = 'tight')
    
    
    
    
    
