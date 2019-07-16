# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:02:16 2018

@author: ning
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from collections import Counter
from utils import (preprocessing)
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt
saving_dir = '../results/raw counts'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

# Exp 1
experiment = 'pos'
df         = pd.read_csv(os.path.join(working_dir,'../data/PoSdata.csv'))
df         = df[df.columns[1:]]
df.columns = ['participant',
              'blocks',
              'trials',
              'firstgabor',
              'success',
              'tilted',
              'correct',
              'RT_correct',
              'awareness',
              'RT_awareness',
              'confidence',
              'RT_confidence']
# make sure all the attributes are either 0 or 1
df.loc[:,'success'   ] = df.loc[:,'success'   ].values - 1
df.loc[:,'awareness' ] = df.loc[:,'awareness' ].values - 1
df.loc[:,'confidence'] = df.loc[:,'confidence'].values - 1
# use all 3 rating features
feature_names = [
                 'correct',
                 'awareness',
                 'confidence',]
target_name = 'success'
results = {}
for participant, df_sub in df.groupby('participant'):
    X,y,groups = preprocessing(
            df_sub,
            participant,
            n_back = 1,
            names = feature_names,
            independent_variables = feature_names,
            dependent_variable = target_name,
                                             )
    
    df_cat = pd.DataFrame(X,columns = feature_names)
    df_cat[target_name] = y.T
    df_cat['sub'] = groups
    def text(x):
        aware = {0:'low awareness',
                 1:'high awareness',}[x['awareness']]
        confidence = {0:'low confidence',
                      1:'high confidence'}[x['confidence']]
        success = {0:'low success',
                   1:'high success'}[x['success']]
        return f"{aware} and {confidence} --> {success}"
    temp = [text(x) for ii,x in df_cat.iterrows()]
    df_cat['text'] = temp
    
    counts = dict(Counter(temp))
    
    for name in counts.keys():
        if name not in results.keys():
            results[name] = []
        
        results[name].append(counts[name]/len(temp))
results = pd.DataFrame(results)
res_melt = pd.melt(results,id_vars=None,value_vars=results.columns,
                   var_name = 'N-1 --> N',
                   value_name = 'Probability')
res_melt['x'] = 0
hue_order = pd.unique(res_melt['N-1 --> N'])
hue_order.sort()
fig,ax = plt.subplots(figsize=(15,10))
sns.barplot(x = 'x',
            y = 'Probability',
            hue = 'N-1 --> N',
            hue_order = hue_order,
            data = res_melt,
            ax = ax,
            )
ax.set(ylim=(0,1.))

res_melt['experiment'] = experiment
res_melt.to_csv(os.path.join(saving_dir,'pos.csv'),index=False)












