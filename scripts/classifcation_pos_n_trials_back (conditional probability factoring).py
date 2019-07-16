# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:57:48 2019

@author: ning
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
saving_dir = '../results/conditional probability factoring'
figure_dir = '../figures/conditional probability factoring'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

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
temp = []
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
#            for d in [0,1]:
            temp.append([a,b,c])
unique_events = np.array(temp)
key_for_results = {'{},{},{}'.format(*k):[] for k in unique_events}
results = dict(
        prob = [],
        correctness = [],
        awareness = [],
        confidence = [],
        success = [],
        sub = [],
        window = [],)
for participant in ['AC', 'CL', 'FW', 'HB', 'KK', 'LM', 'MC', 'MP1', 'MP2', 'NN', 'RP','SD', 'TJ', 'TS', 'WT']:
    
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
    
    ##################################################################
    np.random.seed(12345)
    # use all 6 possible features
    feature_names = [
                     'correct',
                     'awareness',
                     'confidence',
                     ]
    target_name = 'success'
    df_ = df_sub.copy()
    window = 1
    features, targets = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
        """
        Important note:
        the following shifting is not to directly perform the mismatching between the feature trials and
        the target trials, but it is to delete the last N rows of the feature trials and the first N rows
        of the target trials. In such case, when we put them together, the mismatching would be exactly 
        what we want: first trial of feature to predict the next trial of target.
        """
        feature       = (df_block[feature_names].shift(window) # shift downward so that the last n_back rows are gone
                                                .dropna() # since some of rows are gone, so they are nans
                                                .values # I only need the matrix not the data frame
                )
        target        = (df_block[target_name].shift(-window) # same thing for the target, but shifting upward, and the first n_back rows are gone
                                              .dropna()
                                              .values
                     )
        features.append(feature)
        targets.append(target)
    features          = np.concatenate(features)
    targets           = np.concatenate(targets)
    
    df_cpf = pd.DataFrame(features,columns = feature_names)
    df_cpf[target_name] = targets
    j = 0
    for keys in key_for_results:
        correct,aware,conf = [int(k) for k in keys.split(',')]
        idx_pick = (df_cpf['correct'] == correct) & \
                   (df_cpf['awareness'] == aware) & \
                   (df_cpf['confidence'] == conf)
#        print(np.sum(idx_pick))
        df_pick = df_cpf[idx_pick]
        if np.sum(idx_pick) > 0:
            for success in [0.,1.]:
                j += 1
                idx_pick = df_pick['success'] == success
                df_pick_sub = df_pick[idx_pick]
                results['sub'].append(participant)
                results['window'].append(window)
                results['prob'].append(df_pick_sub.shape[0]/df_pick.shape[0])
                results['success'].append(success)
                results['correctness'].append(correct)
                results['awareness'].append(aware)
                results['confidence'].append(conf)
        else:
            for success in [0.,1.]:
                j += 1
                idx_pick = df_pick['success'] == success
                df_pick_sub = df_pick[idx_pick]
                results['sub'].append(participant)
                results['window'].append(window)
                results['prob'].append(np.nan)
                results['success'].append(success)
                results['correctness'].append(correct)
                results['awareness'].append(aware)
                results['confidence'].append(conf)
#    print(j)
results_to_save = pd.DataFrame(results)
results_to_save.to_csv(os.path.join(saving_dir,'pos.csv'),index=False)


df_plot = results_to_save.copy()
df_plot['awareness'] = df_plot['awareness'].map({1.:'aware',0.:'unaware'})
df_plot['confidence'] = df_plot['confidence'].map({1.:'high confidence',0.:'low confidence'})
df_plot['correctness'] = df_plot['correctness'].map({1.:'correct',0.:'incorrect'})
df_plot['success'] = df_plot['success'].map({1.:'high pos',0.:'low pos'})
df_plot['level'] = df_plot['correctness'] + ', '+df_plot['awareness'] +', '+ df_plot['confidence']

#temp = []
#for (target,subject),df_sub in df_plot.groupby(['success','sub']):
#    df_sub['prob'] = df_sub['count'] / df_sub['count'].sum()
#    temp.append(df_sub)
#df_plot = pd.concat(temp)
df_plot.to_csv(os.path.join(saving_dir,'pos_for_plot.csv'))
df_plot = df_plot.sort_values(['awareness'])

g = sns.catplot(x = 'awareness',
                y = 'prob',
                hue = 'confidence',
                col = 'correctness',
                row = 'success',
                row_order = ['high pos','low pos'],
                data = df_plot,
                kind = 'bar',
                aspect = 2,
                )
(g.set_axis_labels('Awareness','Probability')
  .set_titles("{row_name} | {col_name}")
  .set(ylim=(0.,0.85))
  .despine(left=True))
for ii,(target,df_sub) in enumerate(df_plot.groupby('success')):
#    formula = 'prob ~ C(correctness)*C(awareness)*C(confidence)'
#    model = ols(formula, df_sub).fit()
#    aov_table = anova_lm(model, typ=2)
#    s = f"{target}, F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}"
#    print(s)
#    g.axes[ii][0].annotate(s,xy=(-0.45,.8))
    g.axes[ii][0].set(ylabel=f'Probability | {target}')
g.savefig(os.path.join(figure_dir,'pos.png'),
          dpi = 400,
          bbox_inches = 'tight')


df_plot['level'] = df_plot['correctness'] + ', '+df_plot['awareness'] +', '+ df_plot['confidence']
for target,df_sub in df_plot.groupby(['success']):
    temp = {}
    df_sub = df_sub.sort_values(['sub','window','level'])
    for level,df_sub_sub in df_sub.groupby(['level']):
#        print(df_sub_sub.shape)
        temp[level] = df_sub_sub['prob'].values
    for_j = pd.DataFrame(temp)
    for_j.to_csv(os.path.join(saving_dir,f'{target} for jsp.csv'))#,na_rep='NAN')

    aovrm = AnovaRM(df_sub, 
                    'prob', 
                    'sub', 
                    within=['awareness','confidence','correctness'],
                    aggregate_func = np.nanmean)
    res = aovrm.fit().summary().tables[0]
    res.to_csv(os.path.join(saving_dir,f'ANVOA report {target}.csv'))















