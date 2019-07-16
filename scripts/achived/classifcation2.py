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
from sklearn.utils import shuffle
from utils import classification,resample_ttest,MCPConverter

graph_dir = os.path.join(working_dir,'graph')
dot_dir = os.path.join(working_dir,'dot')
if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)
if not os.path.exists(dot_dir):
    os.mkdir(dot_dir)

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

np.random.seed(12345)
results = dict(sub        = [],
               model      = [],
               score      = [],
               window     = [],
               correct    = [],
               awareness  = [],
               confidence = [])
for n_back in range(11): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        features                   = []
        targets                    = []
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        for block, df_block in df_sub.groupby('blocks'):
            feature = (df_block[[
#                                 'trials',
                                 'correct',
                                 'awareness',
                                 'confidence']].shift(-n_back) # shift upward so that the last n_back rows are gone
                                               .dropna() # since some of rows are gone, so they are nans
                                               .values # I only need the matrix not the data frame
                       )
            target = (df_block[[
#                                'trials',
                                'success']].shift(n_back) # same thing for the target, but shifting downward 
                                           .dropna()
                                           .values
                     )
            features.append(feature)
            targets.append(target)
        features = np.concatenate(features)
        targets  = np.concatenate(targets)
        features,targets = shuffle(features,targets)
        # this is the part that is redundent and the code is long
        results          = classification(features,
                                          targets,
                                          results,
                                          participant,
                                          experiment,
                                          window=n_back)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
c.to_csv('../results/Pos.csv',index=False) # save as a csv
c = pd.read_csv('../results/Pos.csv')
# now it is the nonparametric t test with random resampling 
ttest = dict(model    =[],
             window   =[],
             ps_mean  =[],
             ps_std   =[])
for (model,window), c_sub in c.groupby(['model','window']):
    ps = resample_ttest(c_sub['score'].values,# numpy-array
                        baseline=0.5,# the value we want to compare against to
                        n_ps=500,# estimate the p value 500 times
                        n_permutation=int(5e4) # use 50000 resamplings to estimate 1 p value
                        )
    ttest['model'  ].append(model      )
    ttest['window' ].append(window     )
    ttest['ps_mean'].append(np.mean(ps))
    ttest['ps_std' ].append(np.std(ps) )
    print('{} window {} {:.3f}'.format(model,window,np.mean(ps)))
d = pd.DataFrame(ttest) # transform a dictionary object to data frame
# now it is the p value correction for multiple comparison
# note that the correction is done within each model along the number of windows
# , and we have 3 models
temp = []
for model,d_ in d.groupby('model'): # for each model
    idx_sort = np.argsort(d_['ps_mean'].values)
    for name in d_.columns:
        d_[name] = d_[name].values[idx_sort]
    coverter            = MCPConverter(d_['ps_mean'].values # numpy array 
                                       ) # initialize the functional object
    df                  = coverter.adjust_many() # run the method in the object
    d_['ps_corrected'] = df['bonferroni'].values # here is one of the 4 correction methods
    temp.append(d_)
d    = pd.concat(temp)
d.to_csv('../results/Pos_ttest.csv',index = False)

#############################################################################
################################## Exp 2 ####################################
#############################################################################
experiment = 'att'
df         = pd.read_csv(os.path.join(working_dir,'../data/ATTfoc.csv'))
df         = df[df.columns[1:]]
df.columns = ['participant',
              'blocks',
              'trials',
              'firstgabor',
              'attention',
              'tilted',
              'correct',
              'RT_correct',
              'awareness',
              'RT_awareness',
              'confidence',
              'RT_confidence']

np.random.seed(12345)
results = dict(sub        =[],
               model      =[],
               score      =[],
               window     =[],
               correct    =[],
               awareness  =[],
               confidence =[])
for n_back in range(11):# loop through the number of trials you want to look back
    for participant,df_sub in df.groupby('participant'):# loop through each subject
        # make sure each attribute have the values bring either 0 or 1
        df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        features = []
        targets  = []
        for block, df_block in df_sub.groupby('blocks'): # for each block within each subject
            feature = (df_block[['correct',
                                'awareness',
                                'confidence']].shift(-n_back)
                                              .dropna()
                                              .values
                                              )
            target  = (df_block['attention'].shift(n_back)
                                            .dropna()
                                            .values)
            features.append(feature)
            targets.append(target)
        features         = np.concatenate(features)
        targets          = np.concatenate(targets)
        features,targets = shuffle(features,targets)
        results          = classification(features,
                                          targets,
                                          results,
                                          participant,
                                          experiment,
                                          window=n_back)
c = pd.DataFrame(results)
c.to_csv('../results/ATT.csv',index = False)
c = pd.read_csv('../results/ATT.csv')
ttest = dict(model   =[],
             window  =[],
             ps_mean =[],
             ps_std  =[])
for (model,window), c_sub in c.groupby(['model','window']):
    ps = resample_ttest(c_sub['score'].values,
                        baseline      = 0.5,
                        n_ps          = 500,
                        n_permutation = int(5e4))
    ttest['model'  ].append(model      )
    ttest['window' ].append(window     )
    ttest['ps_mean'].append(np.mean(ps))
    ttest['ps_std' ].append(np.std(ps) )
    print('{} window {} {:.3f}'.format(model,window,np.mean(ps)))
d            = pd.DataFrame(ttest)
temp         = []
for model,d_ in d.groupby('model'):
    idx_sort = np.argsort(d_['ps_mean'].values)
    for name in d_.columns:
        d_[name] = d_[name].values[idx_sort]
    coverter            = MCPConverter(d_['ps_mean'].values)
    df                  = coverter.adjust_many()
    d_['ps_corrected'] = df['bonferroni'].values
    temp.append(d_)
d            = pd.concat(temp)
d.to_csv('../results/ATT_ttest.csv',index = False)


    
