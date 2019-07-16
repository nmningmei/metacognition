# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:18:10 2018
@author: ning
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from glob import glob
import utils
sns.set_style('whitegrid')
sns.set_context('poster')
import matplotlib.pyplot as plt
working_dir = '../results/dprime'

def split(x):
    a,b = x.split('_')
    awareness,pos = a.split('aw')
    if len(awareness) == 0:
        awareness = 0
    else:
        awareness = 1
    if 'high' in pos:
        pos = 1
    else:
        pos = 0
    return awareness,pos,b
df = []
for f in glob(os.path.join(working_dir,'*.csv')):
    df_sub = pd.read_csv(f)
    columns = list(df_sub.columns)
    df_sub['sub'] = df_sub[columns[0]]
    temp = pd.melt(df_sub,id_vars='sub',
                   value_vars=columns[1:])
    awareness, pos, dprimetype = [],[],[]
    for ii,row in temp.iterrows():
        awareness_,pos_,dprime_ = split(row['variable'])
        awareness.append(awareness_)
        pos.append(pos_)
        dprimetype.append(dprime_)
    temp['awareness'] = awareness
    temp['Rating'] = pos
    temp['dprime_type'] = dprimetype
    temp['Experiment'] = f.split('_')[0].split('\\')[-1][:3]
    df.append(temp)
df = pd.concat(df)
df['Experiment'] = df['Experiment'].map({'ATT':"Attention decision",
                                         'PoS':"Belief of success"})

df_plot = []
for experiment in pd.unique(df['Experiment']):
    df_sub = df[df['Experiment'] == experiment]
    df_sub['awareness'] = df_sub['awareness'].map({1:'Unaware',0:'Aware'})
    df_sub['Rating'] = df_sub['Rating'].map({0:'Low {}'.format(experiment.lower()),
                                             1:'High {}'.format(experiment.lower())})
    df_sub['legend'] = df_sub['awareness'] + ', ' + df_sub['Rating']
    df_plot.append(df_sub)
df_plot = pd.concat(df_plot)
df_plot['dprime_type'] = df_plot['dprime_type'].map({'dprime':"d'",
                                                    'metadprime':"Meta-d'"})

results = dict(
        dprime_type = [],
        experiment = [],
        ps_mean = [],
        ps_std = [],
        high_aware = [],
        low_aware = [],
        )
for (dpr,exp),df_sub in df_plot.groupby(['dprime_type','Experiment']):
    df_sub
    df_high_awe = df_sub[df_sub['awareness'] == 'Aware'].sort_values(['sub','variable','Rating',])
    df_low_awe = df_sub[df_sub['awareness'] == 'Unaware'].sort_values(['sub','variable','Rating',])
    ps = utils.resample_ttest_2sample(df_high_awe['value'].values,
                                      df_low_awe['value'].values,
                                      n_ps = 100,
                                      n_permutation = 5000)
    results['dprime_type'].append(dpr)
    results['experiment'].append(exp)
    results['ps_mean'].append(ps.mean())
    results['ps_std'].append(ps.std())
    results['high_aware'].append(df_high_awe['value'].values.mean())
    results['low_aware'].append(df_low_awe['value'].values.mean())
results = pd.DataFrame(results)

results = results.sort_values('ps_mean')
converter = utils.MCPConverter(pvals = results['ps_mean'].values)
d = converter.adjust_many()
results['p_corrected'] = d['bonferroni'].values

results['stars'] = results['p_corrected'].apply(utils.stars)


for experiment in pd.unique(df_plot['Experiment']):
    results_sub = results[results['experiment'] == experiment]
    g = sns.catplot(x = 'awareness',
                       y = 'value',
                       hue = 'Rating',
                       col = 'dprime_type',
#                       row = 'Experiment',
                       data = df_plot[df_plot['Experiment'] == experiment],
                       aspect = 2,
                       kind = 'bar',
                       legend_out = False,
                       
                       )
    (g.set_axis_labels("Awareness","Sensitivity")
      .set_titles("{col_name}")
#      .fig.suptitle('Experiment {}'.format(experiment),
#                    y=1.03)
      )
    g.axes.flatten()[0].get_legend().set_title('')
    g.set(ylim=(0,7))
    
    ax_d = g.axes.flatten()[0]
    ax_metad = g.axes.flatten()[1]
    
    y_start = 3
    ax_d.hlines(y_start,0,1)
    ax_d.vlines(0,y_start-0.5,y_start)
    ax_d.vlines(1,y_start-0.5,y_start)
    ax_d.annotate(results_sub[results_sub['dprime_type'] == "d'"]['stars'].values[0],
                xy = ((0+1)/2-0.005,y_start+0.001))
    
    if experiment == 'Attention decision':
        ax_metad.hlines(y_start,-0.2,0.8)
        ax_metad.vlines(-0.2,y_start-0.5,y_start)
        ax_metad.vlines(0.80,y_start-0.5,y_start)
        ax_metad.annotate(results_sub[results_sub['dprime_type'] == "Meta-d'"]['stars'].values[0],
                    xy = ((-0.25+0.75)/2-0.005,y_start+0.1))
    else:
        ax_metad.hlines(y_start,0,1)
        ax_metad.vlines(0,y_start-0.5,y_start)
        ax_metad.vlines(1,y_start-0.5,y_start)
        ax_metad.annotate(results_sub[results_sub['dprime_type'] == "Meta-d'"]['stars'].values[0],
                    xy = ((0+1)/2-0.005,y_start+0.1))
        
    
    g.savefig('../figures/dprime,metadprime,{}.png'.format(experiment),
              dpi = 500, bbox_inches='tight')

df_plot['Experiment'] = df_plot['Experiment'].map({'Attention decision':'Exp.2.',
       'Belief of success':'Exp.1.'})
results['experiment'] = results['experiment'].map({'Attention decision':'Exp.2.',
       'Belief of success':'Exp.1.'})

fig,axes = plt.subplots(figsize=(15,10),nrows = 2, ncols = 2,sharey = True)
df_plot = df_plot.sort_values(['Experiment','dprime_type'])
y_start = 3
for ii,(((experiment,dtype),df_sub),ax) in enumerate(zip(df_plot.groupby(['Experiment','dprime_type']),axes.flatten())):
    ax = sns.barplot(x = 'awareness',
                     y = 'value',
                     hue = 'Rating',
                     data = df_sub,
                     ax = ax)
    idx = np.logical_and(results['experiment'] == experiment,
                         results['dprime_type'] == dtype)
    df_stat = results[idx]
    if ii == 0:
        ax.set(xlabel='',ylabel='Sensitivity',ylim = (0,6),
               title = f'{experiment} | {dtype}')
        ax.get_legend().set_title('')
        ax.hlines(y_start,0,1)
        ax.vlines(0,y_start-0.5,y_start)
        ax.vlines(1,y_start-0.5,y_start)
        ax.annotate(df_stat['stars'].values[0],
                    xy = ((0+1)/2-0.005,y_start+0.001))
    elif ii == 1:
        ax.set(xlabel='',ylabel='',
               title = f'{experiment} | {dtype}')
        ax.legend_.remove()
        ax.hlines(y_start,0,1)
        ax.vlines(0,y_start-0.5,y_start)
        ax.vlines(1,y_start-0.5,y_start)
        ax.annotate(df_stat['stars'].values[0],
                    xy = ((0+1)/2-0.005,y_start+0.001))
    elif ii == 2:
        ax.set(xlabel='Awareness',ylabel='Sensitivity',
               title = f'{experiment} | {dtype}')
        ax.get_legend().set_title('')
        ax.hlines(y_start,0,1)
        ax.vlines(0,y_start-0.5,y_start)
        ax.vlines(1,y_start-0.5,y_start)
        ax.annotate(df_stat['stars'].values[0],
                    xy = ((0+1)/2-0.005,y_start+0.001))
    elif ii == 3:
        ax.set(xlabel='Awareness',ylabel='',
               title = f'{experiment} | {dtype}')
        ax.legend_.remove()
        ax.hlines(y_start,0,1)
        ax.vlines(0,y_start-0.5,y_start)
        ax.vlines(1,y_start-0.5,y_start)
        ax.annotate(df_stat['stars'].values[0],
                    xy = ((0+1)/2-0.005,y_start+0.001))
fig.tight_layout()
figure_dir = '../figures/final_figures'
fig.savefig(os.path.join(figure_dir,'fig5.png'),
            dpi = 500,
            bbox_inches = 'tight')









