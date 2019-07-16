# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:56:23 2019

@author: ning
"""

import os
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')


working_dir = '../results/conditional probability factoring'
figure_dir = '../figures/conditional probability factoring'

working_data = [os.path.join(working_dir,'pos_for_plot.csv'),os.path.join(working_dir,'att_for_plot.csv')]
targets = {'pos':'success','att':'attention'}
def load(x):
    df = pd.read_csv(x)
    experiment = x.split('/')[-1].split("\\")[-1].split('.')[0].split('_')[0]
    df['experiment'] = experiment
    df['target'] = df[targets[experiment]]
    return df
df = pd.concat([load(f) for f in working_data])
df['pick'] = df['target'].apply(lambda x:x.split(' ')[0])
df = df[df['pick'] == 'high']
df = df.sort_values(['awareness','confidence','correctness'])

g = sns.catplot(x = 'awareness',
                y = 'prob',
                hue = 'confidence',
                col = 'correctness',
                row = 'experiment',
                row_order = ['pos','att'],
                data = df,
                kind = 'bar',
                aspect = 2,)
(g.set_axis_labels('awareness','Probability')
  .set_titles("{col_name}")
  .set(ylim=(0,0.85))
  .despine(left=True))
g.axes[0][0].set(ylabel = 'P(high belief of success)')
g.axes[1][0].set(ylabel = 'P(decision to focus attention)')
g.savefig(os.path.join(figure_dir,'supp fig 1.png'),
          dpi = 500,
          bbox_inches = 'tight')






























