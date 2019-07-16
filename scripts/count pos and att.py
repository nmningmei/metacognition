# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:52:21 2019

@author: ning
"""
import os
import pandas as pd
import numpy as np
from glob import glob
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')
from scipy import stats

working_dir = '../data'
def load(x):
    df = pd.read_csv(x)
    df = df[df.columns[1:]]
    df.columns = ['participant',
                  'blocks',
                  'trials',
                  'firstgabor',
                  'target',
                  'tilted',
                  'correct',
                  'RT_correct',
                  'awareness',
                  'RT_awareness',
                  'confidence',
                  'RT_confidence']
    df['experiment'] = x.split('\\')[-1][:3].lower()
    return df
working_data = pd.concat(load(f) for f in glob(os.path.join(working_dir,"*.csv")))

results = dict(
        experiment = [],
        sub = [],
        high = [],
        low = [])
for (experiment,sub),df_sub in working_data.groupby(['experiment','participant']):
    events = dict(Counter(df_sub['target']))
    results['experiment'].append(experiment)
    results['sub'].append(sub)
    results['high'].append(events[2] / (events[1] + events[2]))
    results['low'].append(events[1] / (events[1] + events[2]))
results = pd.DataFrame(results)

df_plot = pd.melt(results,id_vars=['experiment','sub'],
                  value_vars=['high','low'],
                  )
df_plot = df_plot.sort_values('experiment',ascending = False)

fig,ax = plt.subplots(figsize=(8,8))
ax = sns.violinplot(x = 'experiment',
                    y = 'value',
                    hue = 'variable',
                    data = df_plot,
                    split = True,
                    cut = 0,
                    inner = 'quartile',
                    ax = ax)
legend = ax.legend()
legend.texts[0].set_text("low")
ax.set(xlabel = 'Experiment',ylabel = 'Probability',
       xticklabels = ['Exp 1.', 'Exp. 2'])
fig.savefig('../figures/prob count pos and att.png',
            dpi = 500,
            bbox_inches = 'tight')

temp = dict(
        experiment = [],
        level = [],
        t = [],
        p = [],
        mean = [],
        std = [],
        N = [],)
for (experiment,level),df_sub in df_plot.groupby(['experiment','variable']):
    a = df_sub['value'].values
    a_transform = np.arctanh(a)
    t,p = stats.ttest_1samp(a_transform,np.arctanh(0.5),)
    temp['experiment'].append(experiment)
    temp['level'].append(level)
    temp['t'].append(t)
    temp['p'].append(p)
    temp['mean'].append(a.mean())
    temp['std'].append(a.std())
    temp['N'].append(len(a))
df_stats = pd.DataFrame(temp)


















