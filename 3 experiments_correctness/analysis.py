#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:07:58 2018

@author: nmei

exp1 (e1) there were 2 possible awareness ratings (unaware and aware of the orientation)

exp2 (e2) there were 3 possible awareness ratings ( (
e.g. 1- no experience, 2 brief glimpse 3 almost clear or clear perception)
BUT if can make a binary classification by focussing on 1 and 2 which are 
the majority of the trials.

exp3 (e3) there were 2 possible awareness ratings (unaware and aware of the orientation)

"""

import pandas as pd
import numpy as np
import utils

df1 = pd.read_csv('e1.csv').iloc[:,1:]
df2 = pd.read_csv('e2.csv').iloc[:,1:]
df3 = pd.read_csv('e3.csv').iloc[:,1:]

# exp 1
df = df1.copy()
df = df[['blocks.thisRepN',
         'trials.thisIndex',
         'key_resp_2.keys',
         'resp.corr',
         'resp_mrating.keys',
         'participant',]]
df.columns = ['blocks',
              'trials',
              'awareness',
              'correctness',
              'confidence',
              'participant',]
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correctness      = [],
               awareness        = [],
               confidence       = [],
               )
# use success, awareness, and confidence as features
np.random.seed(12345)
# use judgement features
feature_names = [
                 'correctness',
                 'awareness',
                 'confidence',]
target_name = 'awareness'
experiment = 'e1'
name_for_scale = ['awareness']
for participant,df_sub in df.groupby(['participant']):
    for n_back in np.arange(1,5):
        results          = utils.classification(
                                      df_sub.dropna(),
                                      feature_names,
                                      target_name,
                                      results,
                                      participant,
                                      experiment,
                                      window = n_back,
                                      chance = False,
                                      name_for_scale = name_for_scale
                                      ) 















































