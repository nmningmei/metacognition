# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:02:16 2018

@author: ning

leave one participant out

"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.utils import shuffle
from utils import MCPConverter,classification_leave_one_sub_out
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_context('poster')
sns.set_style('whitegrid')

results_dir = '../results/'
working_dir = '../data/'
figure_dir = '../figures'
dot_dir = os.path.join(working_dir,'dot')
if not os.path.exists(dot_dir):
    os.mkdir(dot_dir)

if __name__ == '__main__':
    # Exp 1
    experiment = 'pos'
    df         = pd.read_csv(os.path.join(working_dir,'PoSdata.csv'))
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
    results = dict(model      = [],
                   score      = [],
                   window     = [],
                   correct    = [],
                   awareness  = [],
                   confidence = [],
                   p_val      = [],)
    for n_back in range(4): # loop through the number of trials looking back
        X,y,groups = [],[],[]
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
            subs     = np.array([participant] * len(targets))
            features,targets = shuffle(features,targets)
            X.append(features)
            y.append(targets)
            groups.append(subs)
        X = np.concatenate(X)
        y = np.concatenate(y)
        groups = np.concatenate(groups)
        
        results = classification_leave_one_sub_out(X,
                                                   y,
                                                   results,
                                                   groups,
                                                   experiment,
                                                   dot_dir,
                                                   window=n_back)
    
    c = pd.DataFrame(results) # tansform a dictionary object to a data frame
    c.to_csv(os.path.join(results_dir,'Pos_leave_one_sub_out.csv'),index=False) # save as a csv
    
    
    # Exp 2
    experiment = 'att'
    df         = pd.read_csv(os.path.join(working_dir,'ATTfoc.csv'))
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
    results = dict(model      = [],
                   score      = [],
                   window     = [],
                   correct    = [],
                   awareness  = [],
                   confidence = [],
                   p_val      = [],)
    for n_back in range(11): # loop through the number of trials looking back
        X,y,groups = [],[],[]
        for participant,df_sub in df.groupby('participant'):# for each subject
            # make sure all the attributes are either 0 or 1
            df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
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
                                    'attention']].shift(n_back) # same thing for the target, but shifting downward 
                                               .dropna()
                                               .values
                         )
    
                features.append(feature)
                targets.append(target)
            features = np.concatenate(features)
            targets  = np.concatenate(targets)
            subs     = np.array([participant] * len(targets))
            features,targets = shuffle(features,targets)
            X.append(features)
            y.append(targets)
            groups.append(subs)
        X = np.concatenate(X)
        y = np.concatenate(y)
        groups = np.concatenate(groups)
        
        results = classification_leave_one_sub_out(X,
                                                   y,
                                                   results,
                                                   groups,
                                                   experiment,
                                                   dot_dir,
                                                   window=n_back)
    
    c = pd.DataFrame(results) # tansform a dictionary object to a data frame
    c.to_csv(os.path.join(results_dir,'Att_leave_one_sub_out.csv'),index=False) # save as a csv
    
    ###########################################################################
    #############################################################################
    ############################################################################
    
    pos = pd.read_csv(os.path.join(results_dir,'Att_leave_one_sub_out.csv'))
    att = pd.read_csv(os.path.join(results_dir,'Att_leave_one_sub_out.csv'))
    
    combimed = []
    n_back = 4
    df = pos.copy()
    df = df[df['window'] <= 4]
    temp = []
    for model,df_sub in df.groupby('model'):
        df_temp = df_sub.sort_values(by='p_val')
        converter = MCPConverter(df_temp['p_val'].values)
        df_pvals = converter.adjust_many()
        df_temp['p_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_temp)
    df = pd.concat(temp)
    df['experiment'] = 'Probability of Success'
    combimed.append(df)
    
    df = att.copy()
    df = df[df['window'] <= 4]
    temp = []
    for model,df_sub in df.groupby('model'):
        df_temp = df_sub.sort_values(by='p_val')
        converter = MCPConverter(df_temp['p_val'].values)
        df_pvals = converter.adjust_many()
        df_temp['p_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_temp)
    df = pd.concat(temp)
    df['experiment'] = 'Attention'
    combimed.append(df)
    
    cdf = pd.concat(combimed)
    # plot
    
    fig,axes = plt.subplots(figsize=(12,6),ncols=2)
    for ax,ex in zip(axes,['Probability of Success','Attention']):
        ax=sns.barplot(x='window',
                      y='p_corrected',
                      data=cdf[cdf.experiment == ex],
                      ci=99,
                      ax=ax)
        ax.axhline(0.05,linestyle='--',alpha=0.6,color='red',
                   label='0.05 alpha level')
        ax.set(xlabel="Trials look back",
               ylabel="P value (Corrected)",
               title='Experiment: {}'.format(ex))
        ax.legend(loc='best')
    fig.suptitle('Leave One Subject Out',y=1.0)
    fig.savefig(os.path.join(figure_dir,'leave one out.png'),bbox_inches='tight',
              dpi=400)
    

































