#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:47:07 2018

@author: nmei
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import re
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from matplotlib import pyplot as plt
import utils

figure_dir = 'figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_dir = {ii+1:'batch/correlation/results_e{}'.format(ii+1) for ii in range(3)}
working_data = [glob(os.path.join(d,'*.csv')) for _,d in working_dir.items()]
working_data = np.concatenate(working_data)
df = []
for f in working_data:
    exp, = re.findall('\d',f)
    df_temp = pd.read_csv(f)
    df_temp['experiment'] = exp
    df.append(df_temp)
df = pd.concat(df)

decode_dir = {ii+1:'batch/results_e{}'.format(ii+1) for ii in range(3)}
decode_data = [glob(os.path.join(d,'*.csv')) for _,d in decode_dir.items()]
decode_data = np.concatenate(decode_data)
df_decode = []
for f in decode_data:
    exp, = re.findall('\d',f)
    df_temp = pd.read_csv(f).iloc[:,1:]
    df_temp['experiment'] = exp
    df_decode.append(df_temp)
df_decode = pd.concat(df_decode)
df_decode = df_decode[df_decode['chance']==False]

df = df.sort_values(['sub','model','window','experiment',])
df_decode = df_decode.sort_values(['sub','model','window','experiment',])

for col_name in df_decode.columns:
    if col_name not in df.columns:
       df[col_name] = df_decode[col_name].values

col_to_plot = ['p(correct|awareness)',
               'p(correct|unawareness)',
               'p(incorrect|awareness)',
               'p(incorrect|unawareness)']
fig,axes = plt.subplots(figsize=(20,25),
                        nrows=4,
                        ncols=2,
                        sharey=True)
for col_name,ax in zip(col_to_plot,axes):
    for model,a in zip(utils.make_clfs().keys(),ax):
        df_work = df[df['model'] == model]
        sns.barplot(x = 'window',
                    y = col_name,
                    hue = 'experiment',
                    data = df_work,
                    ax = a)




















































