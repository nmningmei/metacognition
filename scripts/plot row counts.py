# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:13:07 2019

@author: ning
"""

import os
import pandas as pd
from glob import glob
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt

working_dir = '../results/raw counts'
pos = pd.read_csv(os.path.join(working_dir,'pos.csv'))
att = pd.read_csv(os.path.join(working_dir,'att.csv'))


fig,axes = plt.subplots(figsize=(15,20),nrows=2,sharey=True)
ax = axes[0]
hue_order = pd.unique(pos['N-1 --> N'])
hue_order.sort()
sns.barplot(x = 'x',
            y = 'Probability',
            hue = 'N-1 --> N',
            hue_order = hue_order,
            data = pos,
            ax = ax,
            )
ax.set(ylim=(0,1.),xlabel='',xticks=[],
       title = "Exp.1")

ax = axes[1]
hue_order = pd.unique(att['N-1 --> N'])
hue_order.sort()
sns.barplot(x = 'x',
            y = 'Probability',
            hue = 'N-1 --> N',
            hue_order = hue_order,
            data = att,
            ax = ax
            )
ax.set(xlabel='',xticks=[],title = "Exp.2")

fig.savefig('../figures/raw counts.png',
            dpi = 400,
            bbox_inches = 'tight')
fig.savefig('../figures/raw counts.png',
#            dpi = 400,
            bbox_inches = 'tight')















