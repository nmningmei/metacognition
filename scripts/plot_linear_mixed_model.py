# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:52:14 2019

@author: ning
"""

import os
from glob import glob

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_context('poster')
sns.set_style('whitegrid')

working_dir = '../results/linear_mixed'
figure_dir = '../figures/linear_mixed'
working_data = glob(os.path.join(working_dir,'*fit.csv'))
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

df_plot = []
for f in working_data:
    f = f.replace('\\','/')
    df = pd.read_csv(f)
    attr = df['Unnamed: 0'].apply(lambda x:x.split('_')[0])
    time = df['Unnamed: 0'].apply(lambda x:int(x.split('_')[1]))
    df = df.iloc[:,1:]
    df['Attributes'] = attr
    df['time'] = time
    df['experiment'] = f.split('/')[-1].split('_')[0]
    df_plot.append(df)
df_plot = pd.concat(df_plot)
def stars(x):
    if x < 0.001:
        return '***'
    elif x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return 'n.s.'
df_plot['star'] = df_plot['sign'].apply(stars)

y_pos = {'POS':0.55,
         'ATT': 0.17}
for ii,exp in enumerate(['POS','ATT']):
    fig,ax = plt.subplots(figsize = (12,6))

    df_sub = df_plot[df_plot['experiment'] == exp]
    df_sub['subtract'] = np.concatenate([np.arange(-0.27,3,),
                                         np.arange(0,4),
                                         np.arange(0.27,4)])
    ax = sns.barplot(x = 'time',
                     y = 'Estimate',
                     hue = 'Attributes',
                     data = df_sub,
                     ax = ax,
                     alpha = 0.8,)
    df_sub = pd.concat([df_sub[df_sub['Attributes'] == attr] for attr in ['awareness',
                                  'confidence',
                                  'correct',]])
    df_sub = df_sub.sort_values(['time'])
    ax.errorbar(df_sub['subtract'].values,
                df_sub['Estimate'].values,
                yerr = [np.abs(df_sub['lwr'].values),
                        df_sub['upr'].values],
                linestyle = '',
                color = 'black',
                alpha = 1.,
                capsize = 10,)
    exp_title = "Exp. 1" if exp == 'POS' else "Exp. 2"
    ax.set(title = exp_title,
           xlabel = 'Trial back',
           ylabel = 'Coefficients',
           ylim = (-0.06,y_pos[exp] + 0.1 * y_pos[exp]))
    ax.legend(loc = 'upper right',bbox_to_anchor=(1.,0.8))
    for kk,row in df_sub.iterrows():
        if row['star'] != 'n.s.':
            ax.text(row['subtract'] - 0.05,
                    y_pos[exp],
                    row['star'])
    fig.savefig(os.path.join(figure_dir,
                             f'{exp}.png'),
    dpi = 300,
    bbox_inches = 'tight')






