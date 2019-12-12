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
from utils import (classification)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
saving_dir = '../results/linear_mixed'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
n_permu = 500
n_back = 4

# Exp 2
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
df_to_save = []
for participant in ['AC', 'CL', 'FW', 'HB', 'KK', 'LM', 'MC', 'MP1', 'MP2', 'NN', 'RP','SD', 'TJ', 'TS', 'WT']:
#participant = 'AC'
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'success' ] = df_sub.loc[:,'success' ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
    
    feature_names = ['awareness','confidence','correct']
    target_name = 'success'
    for block,df_block in df_sub.groupby(['blocks']):
        features_ = df_block[feature_names].values
        targets_ = df_block[target_name].values
        
        # define how to generate the sequences
        data_gen = TimeseriesGenerator(features_, # the 3 features
                                       targets_, # the target column
                                       length = 4, # the width of the window
                                       batch_size = 1, # so that the generator will output 1 trial per time
                                       )
    
        features,targets = [],[]
        for (feature,target) in data_gen: # let the generator output 1 trial per loop
            # this will make a 4 rows by 3 columns dataframe
            feature = pd.DataFrame(feature[0],columns = [feature_names])
            # let's flatten the dataframe so that each column contains 
            # value per feature and time back
            temp = {}
            for col in feature_names:
                for ii in range(n_back): # the smaller index of the row the further back in time (more negative in value, i.e. -4 is row 0)
                    temp[f'{col}_{n_back-ii}'] = [feature.loc[ii,col].values[0]]
            # convert the dictionary to dataframe
            feature = pd.DataFrame(temp)
            target = pd.DataFrame(target,columns = [target_name])
            
            features.append(feature)
            targets.append(target)
        
        features = pd.concat(features)
        targets = pd.concat(targets)
        temp = pd.concat([features,targets],axis = 1)
        temp['sub_name'] = participant
        temp['blocks'] = block
        df_to_save.append(temp)
df_to_save = pd.concat(df_to_save)
df_to_save.to_csv(os.path.join(saving_dir,'POS.csv'),
                  index = False)




























