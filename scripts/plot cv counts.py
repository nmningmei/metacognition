# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:58:20 2019

@author: ning
"""

import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')
import statsmodels.api as sm
from statsmodels.formula.api import ols

working_dir = '../results/cv_counts'
figure_dir = '../figures'

working_data = glob(os.path.join(working_dir,'*3_1*.csv'))

def strip(x):
    return x.split('_')[-1]

# pos
pos = pd.concat([pd.read_csv(f) for f in working_data if ('Pos' in f)])
pos = pos.groupby(['sub','window']).mean().reset_index()
pos['Experiment'] = 'Exp.1'
pos_melt = pd.melt(pos,
                   id_vars = ['sub','window','Experiment'],
                   value_vars = [
                                  'success_high_cond_correct_low',
                                 'success_high_cond_correct_high', 
                                 'success_high_cond_awareness_low',
                                 'success_high_cond_awareness_high', 
                                 'success_high_cond_confidence_low',
                                 'success_high_cond_confidence_high',
                                 ],
                   var_name = 'Attributes',
                   value_name = 'Conditioned_Prob',)
pos_melt['D'] = pos_melt['Attributes'].apply(strip)
pos_m = pos_melt[pos_melt['window'] == 1]
for HL,df_sub in pos_m.groupby(['D']):
    mod = ols('Conditioned_Prob ~ Attributes',
                    data=df_sub).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(f"{HL}\n",aov_table)

rename_map = {
              'success_high_cond_correct_low':'incorrect',
              'success_high_cond_correct_high':'correct', 
              'success_high_cond_awareness_low':'low awareness',
              'success_high_cond_awareness_high':'high awareness', 
              'success_high_cond_confidence_low':'low confidence',
              'success_high_cond_confidence_high':'high confidence',
              }
pos_melt['Attributes'] = pos_melt['Attributes'].map(rename_map)

# att
att = pd.concat([pd.read_csv(f) for f in working_data if ('att' in f)])
att = att.groupby(['sub','window']).mean().reset_index()
att['Experiment'] = 'Exp.2'
att_melt = pd.melt(att,
                   id_vars = ['sub','window','Experiment'],
                   value_vars = ['attention_high_cond_correct_low',
                                 'attention_high_cond_correct_high', 
                                 'attention_high_cond_awareness_low',
                                 'attention_high_cond_awareness_high', 
                                 'attention_high_cond_confidence_low',
                                 'attention_high_cond_confidence_high'],
                   var_name = 'Attributes',
                   value_name = 'Conditioned_Prob',)
att_melt['D'] = att_melt['Attributes'].apply(strip)
att_m = att_melt[att_melt['window'] == 1]
for HL,df_sub in att_m.groupby(['D']):
    mod = ols('Conditioned_Prob ~ Attributes',
                    data=df_sub).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(f"{HL}\n",aov_table)
rename_map = {'attention_high_cond_correct_low':'incorrect',
              'attention_high_cond_correct_high':'correct', 
              'attention_high_cond_awareness_low':'low awareness',
              'attention_high_cond_awareness_high':'high awareness', 
              'attention_high_cond_confidence_low':'low confidence',
              'attention_high_cond_confidence_high':'high confidence',}
att_melt['Attributes'] = att_melt['Attributes'].map(rename_map)

pos_melt['x'] = 0
att_melt['x'] = 0

fig,axes = plt.subplots(figsize=(15,15),nrows=2,sharey=True,sharex=True)
xlim = (-0.5,1.)
ylim = (0,1.)
ax = axes[0]
hue_order = np.sort(pd.unique(pos_melt[pos_melt['window']==1]['Attributes']))
ax = sns.barplot(x = 'x',
                 y = 'Conditioned_Prob',
                 hue = 'Attributes',
                 hue_order = hue_order,
                 data = pos_melt[pos_melt['window']==1],
                 ax = ax)
ax.legend(bbox_to_anchor=(1.05,0.1),loc=2,borderaxespad=0.)
ax.set(ylim=ylim,
       xticklabels=[],
       xlabel = '',
       ylabel = 'P(High POS)')
ax.axhline(0.8,xmin=0.15,xmax=0.45,color='black',linewidth=5)
ax.annotate('n.s.',xy=(-.225,.82))
ax.axhline(0.8,xmin=0.55,xmax=0.85,color='black',linewidth=5)
ax.annotate('n.s.',xy=(.2,0.82))
ax = axes[1]
hue_order = np.sort(pd.unique(att_melt[att_melt['window']==1]['Attributes']))
ax = sns.barplot(x = 'x',
                 y = 'Conditioned_Prob',
                 hue = 'Attributes',
                 hue_order = hue_order,
                 data = att_melt[att_melt['window']==1],
                 ax = ax)
#ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
ax.get_legend().remove()
ax.set(ylim=ylim,
       xticklabels=[],
       xlabel = 'Data from 1 Trial back',
       ylabel = 'P(High ATT)',)
ax.axhline(0.8,xmin=0.15,xmax=0.45,color='black',linewidth=5)
ax.annotate('n.s.',xy=(-.225,.82))
ax.axhline(0.8,xmin=0.55,xmax=0.85,color='black',linewidth=5)
ax.annotate('n.s.',xy=(.2,0.82))
fig.savefig(os.path.join(figure_dir,'CV_counts.png'),
            dpi = 400,
            bbox_inches = 'tight')
fig.savefig(os.path.join(figure_dir,'CV_counts (light).png'),
            bbox_inches = 'tight')

#def make_plot_attr(kk):
#    pos_plot_mean = kk[kk['window'] == 1].groupby(['Attributes']).mean().reset_index()
#    pos_plot_std = pos_melt[pos_melt['window'] == 1].groupby(['Attributes']).std().reset_index()
#    pos_plot_std['Conditioned_Prob'] = pos_plot_std['Conditioned_Prob'] / np.sqrt(len(pd.unique(pos['sub'])))
#    pos_plot_mean = pos_plot_mean.sort_values(['Attributes'])
#    pos_plot_std = pos_plot_std.sort_values(['Attributes'])
#    return pos_plot_mean,pos_plot_std
#pos_plot_mean,pos_plot_std = make_plot_attr(pos_melt)
#att_plot_mean,att_plot_std = make_plot_attr(att_melt)
#
#fig,axes = plt.subplots(figsize=(15,15),nrows=2,sharey=True)
#ax = axes[0]
#ax.bar(np.arange(6),
#       pos_plot_mean['Conditioned_Prob'].values,
#       color = 'blue',
#       alpha = 0.5)
#ax.errorbar(np.arange(6),
#            pos_plot_mean['Conditioned_Prob'].values,
#            pos_plot_std['Conditioned_Prob'].values,
#            linestyle='',
#            color = 'black',
#            alpha = 1.,
#            capsize = 5,)
#ax.set(ylabel='P(High POS)',ylim=(0,1.),title='Exp. 1')
#ax.set_xticks(np.arange(6))
#ax.set_xticklabels(pos_plot_mean['Attributes'].values,
#                   rotation = 45)
#ax.axhline(0.8,xmin=0.08,xmax=0.45,color='black',linewidth=5)
#ax.annotate('n.s.',xy=(1,.82))
#ax.axhline(0.8,xmin=0.5,xmax=0.95,color='black',linewidth=5)
#ax.annotate('n.s.',xy=(4.,0.82))
#
#ax = axes[1]
#ax.bar(np.arange(6),
#       att_plot_mean['Conditioned_Prob'].values,
#       color = 'blue',
#       alpha = 0.5)
#ax.errorbar(np.arange(6),
#            att_plot_mean['Conditioned_Prob'].values,
#            att_plot_std['Conditioned_Prob'].values,
#            linestyle='',
#            color = 'black',
#            alpha = 1.,
#            capsize = 5,)
#ax.set(ylabel='P(High ATT)',title='Exp. 2')
#ax.set_xticks(np.arange(6))
#ax.set_xticklabels(att_plot_mean['Attributes'].values,
#                   rotation = 45)
#ax.axhline(0.8,xmin=0.08,xmax=0.45,color='black',linewidth=5)
#ax.annotate('n.s.',xy=(1,.82))
#ax.axhline(0.8,xmin=0.5,xmax=0.95,color='black',linewidth=5)
#ax.annotate('n.s.',xy=(4.,0.82))
#fig.tight_layout()

#fig.savefig(os.path.join(figure_dir,'CV_counts.png'),
#            dpi = 400,
#            bbox_inches = 'tight')
#fig.savefig(os.path.join(figure_dir,'CV_counts (light).png'),
#            bbox_inches = 'tight')














