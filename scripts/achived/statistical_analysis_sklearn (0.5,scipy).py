#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:46:24 2018

@author: nmei
"""
import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
from glob import glob
import seaborn as sns
from scipy import stats
import numpy as np
from statsmodels.formula.api import ols#,mixedlm
from statsmodels.stats.anova import anova_lm
from utils import eta_squared,omega_squared,resample_ttest,MCPConverter,post_processing
from itertools import combinations
sns.set_style('whitegrid')
sns.set_context('poster')
saving_dir = '../figures/'
df_dir = '../results/for_spss'
if not os.path.exists(df_dir):
    os.mkdir(df_dir)
def thresholding(value):
    if value < 0.001:
        return "***"
    elif value < 0.01:
        return "**"
    elif value < 0.05:
        return "*"
    else:
        return "ns"
def preparation(c):
    df_temp = {}
    for ((model,window,feature),df_sub) in c.groupby(['Models','Window','Attributes']):
        if 'Tree' in model:
            df_temp['{}_win{}_{}'.format(model,window,feature)] = df_sub['Values'].values
    df_temp = pd.DataFrame(df_temp)
    return df_temp
"""
Take the exponential of each of the coefficients to generate the odds ratios. 
This tells you how a 1 unit increase or decrease in a variable affects the odds of being high POS.
"""
if __name__ == '__main__':
    ##########################################################################################################################################
    pos_w               = glob('../results/experiment_score/Pos_6*.csv')
    pos                 = pd.concat([pd.read_csv(f) for f in pos_w])
    pos.to_csv('../results/Pos_6_features.csv',index=False)
    att_w               = glob('../results/experiment_score/att_6*.csv')
    att                 = pd.concat([pd.read_csv(f) for f in att_w])
    att.to_csv('../results/ATT_6_features.csv',index=False)
    
    df                  = pos.copy()
    #########################  compared against chance level ###############
    df                  = df[(0<df['window']) & (df['window']<5)]
    results = dict(
                   model=[],
                   window=[],
                   pval=[],
                   t=[],
                   df=[],
                    )
    for (model,window),df_sub in df.groupby(['model','window']):
        df_sub = df_sub.sort_values('sub')
        t,pval = stats.ttest_1samp(df_sub['score'].values,0.5,)
        
        results['model'].append(model)
        results['window'].append(window)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)-1)
    results = pd.DataFrame(results)
    temp = []
    for model, df_sub in results.groupby('model'):
        idx_sort = np.argsort(df_sub['pval'])
        for name in results.columns:
            df_sub[name] = df_sub[name].values[idx_sort]
        convert = MCPConverter(pvals=df_sub['pval'].values)
        df_pvals = convert.adjust_many()
        df_sub['ps_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    results.sort_values(['model','window']).to_csv('../results/Pos_ttest_6_features (scipy).csv',index=False)
    
    ##########################################################################
    id_vars             = ['model',
                           'score',
                           'sub',
                           'window',]
    value_vars          =[
                          'correct',
                          'awareness',
                          'confidence',
                          'RT_correct',
                          'RT_awareness',
                          'RT_confidence',
                          ]
    df_post             = post_processing(df,id_vars,value_vars)
#    c                   = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                   = df_post[df_post['Window']<5]
    df_temp             = preparation(c)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'pos,6 features,feature importance.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    df_temp.to_csv(os.path.join(df_dir,'pos,6 features,feature importance.csv'),index=False)
    
    ######
    #######
    #########
    df              = att.copy()
    #########################  compared against chance level ###############
    df                  = df[(0<df['window']) & (df['window']<5)]
    results = dict(
                   model=[],
                   window=[],
                   pval=[],
                   t=[],
                   df=[],
                    )
    for (model,window),df_sub in df.groupby(['model','window']):
        df_sub = df_sub.sort_values('sub')
        t,pval = stats.ttest_1samp(df_sub['score'].values,0.5,)
        
        results['model'].append(model)
        results['window'].append(window)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)-1)
    results = pd.DataFrame(results)
    temp = []
    for model, df_sub in results.groupby('model'):
        idx_sort = np.argsort(df_sub['pval'])
        for name in results.columns:
            df_sub[name] = df_sub[name].values[idx_sort]
        convert = MCPConverter(pvals=df_sub['pval'].values)
        df_pvals = convert.adjust_many()
        df_sub['ps_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    results.sort_values(['model','window']).to_csv('../results/ATT_ttest_6_features (scipy).csv',index=False)
    
    ##########################################################################
    id_vars             = ['model',
                           'score',
                           'sub',
                           'window',]
    value_vars          =[
                          'correct',
                          'awareness',
                          'confidence',
                          'RT_correct',
                          'RT_awareness',
                          'RT_confidence',
                          ]
    df_post             = post_processing(df,id_vars,value_vars)
#    c                   = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                   = df_post[df_post['Window']<5]
    df_temp             = preparation(c)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'att,6 features,feature importance.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    df_temp.to_csv(os.path.join(df_dir,'att,6 features,feature importance.csv'),index=False)
    
    ###################################################################################
    ###################  3 judgement features #########################################
    ###################################################################################
    pos_w           = glob('../results/experiment_score/Pos_3*.csv')
    pos             = pd.concat([pd.read_csv(f) for f in pos_w])
    pos.to_csv('../results/Pos_3_1_features.csv',index=False)
    att_w           = glob('../results/experiment_score/att_3*.csv')
    att             = pd.concat([pd.read_csv(f) for f in att_w])
    att.to_csv('../results/ATT_3_1_features.csv',index=False)
    
    df              = pos.copy()
    #########################  compared against chance level ###############
    df                  = df[(0<df['window']) & (df['window']<5)]
    results = dict(
                   model=[],
                   window=[],
                   pval=[],
                   t=[],
                   df=[],
                    )
    for (model,window),df_sub in df.groupby(['model','window']):
        df_sub = df_sub.sort_values('sub')
        t,pval = stats.ttest_1samp(df_sub['score'].values,0.5,)
        
        results['model'].append(model)
        results['window'].append(window)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)-1)
    results = pd.DataFrame(results)
    temp = []
    for model, df_sub in results.groupby('model'):
        idx_sort = np.argsort(df_sub['pval'])
        for name in results.columns:
            df_sub[name] = df_sub[name].values[idx_sort]
        convert = MCPConverter(pvals=df_sub['pval'].values)
        df_pvals = convert.adjust_many()
        df_sub['ps_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    results.sort_values(['model','window']).to_csv('../results/Pos_ttest_3_1_features (scipy).csv')
    
    ##########################################################################
    id_vars             = ['model',
                           'score',
                           'sub',
                           'window',]
    value_vars          =[
                          'correct',
                          'awareness',
                          'confidence',
                          ]
    df_post             = post_processing(df,id_vars,value_vars)
#    c                   = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                   = df_post[df_post['Window']<5]
    df_temp             = preparation(c)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'pos,judgment features,feature importance.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    df_temp.to_csv(os.path.join(df_dir,'pos,judgment features,feature importance.csv'),index=False)
    
    ######
    ########
    ##########
    df              = att.copy()
    #########################  compared against chance level ###############
    df                  = df[(0<df['window']) & (df['window']<5)]
    results = dict(
                   model=[],
                   window=[],
                   pval=[],
                   t=[],
                   df=[],
                    )
    for (model,window),df_sub in df.groupby(['model','window']):
        df_sub = df_sub.sort_values('sub')
        t,pval = stats.ttest_1samp(df_sub['score'].values,0.5,)
        
        results['model'].append(model)
        results['window'].append(window)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)-1)
    results = pd.DataFrame(results)
    temp = []
    for model, df_sub in results.groupby('model'):
        idx_sort = np.argsort(df_sub['pval'])
        for name in results.columns:
            df_sub[name] = df_sub[name].values[idx_sort]
        convert = MCPConverter(pvals=df_sub['pval'].values)
        df_pvals = convert.adjust_many()
        df_sub['ps_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    results.sort_values(['model','window']).to_csv('../results/ATT_ttest_3_1_features (scipy).csv')
    
    ##########################################################################
    id_vars             = ['model',
                           'score',
                           'sub',
                           'window',]
    value_vars          =[
                          'correct',
                          'awareness',
                          'confidence',
                          ]
    df_post             = post_processing(df,id_vars,value_vars)
#    c                   = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                   = df_post[df_post['Window']<5]
    df_temp             = preparation(c)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'att,judgment features,feature importance.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    df_temp.to_csv(os.path.join(df_dir,'att,judgment features,feature importance.csv'),index=False)
    
    ###############################################################################################
    #############################  RT as features #################################################
    ###############################################################################################
    pos_w           = glob('../results/experiment_score/Pos_RT*.csv')
    pos             = pd.concat([pd.read_csv(f) for f in pos_w])
    pos.to_csv('../results/Pos_RT_features.csv',index=False)
    att_w           = glob('../results/experiment_score/att_RT*.csv')
    att             = pd.concat([pd.read_csv(f) for f in att_w])
    att.to_csv('../results/ATT_RT_features.csv',index=False)
    
    df              = pos.copy()
    #########################  compared against chance level ###############
    df                  = df[(0<df['window']) & (df['window']<5)]
    results = dict(
                   model=[],
                   window=[],
                   pval=[],
                   t=[],
                   df=[],
                    )
    for (model,window),df_sub in df.groupby(['model','window']):
        df_sub = df_sub.sort_values('sub')
        t,pval = stats.ttest_1samp(df_sub['score'].values,0.5,)
        
        results['model'].append(model)
        results['window'].append(window)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)-1)
    results = pd.DataFrame(results)
    temp = []
    for model, df_sub in results.groupby('model'):
        idx_sort = np.argsort(df_sub['pval'])
        for name in results.columns:
            df_sub[name] = df_sub[name].values[idx_sort]
        convert = MCPConverter(pvals=df_sub['pval'].values)
        df_pvals = convert.adjust_many()
        df_sub['ps_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    results.sort_values(['model','window']).to_csv('../results/Pos_ttest_RT_features (scipy).csv')
    
    ##########################################################################
    id_vars             = ['model',
                           'score',
                           'sub',
                           'window',]
    value_vars          =[
                          'RT_correct',
                          'RT_awareness',
                          'RT_confidence',
                          ]
    df_post             = post_processing(df,id_vars,value_vars)
#    c                   = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                   = df_post[df_post['Window']<5]
    df_temp             = preparation(c)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'pos,RT features,feature importance.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    df_temp.to_csv(os.path.join(df_dir,'pos,RT features,feature importance.csv'),index=False)
    
    ######
    ########
    ##########
    df                  = att.copy()
    #########################  compared against chance level ###############
    df                  = df[(0<df['window']) & (df['window']<5)]
    results = dict(
                   model=[],
                   window=[],
                   pval=[],
                   t=[],
                   df=[],
                    )
    for (model,window),df_sub in df.groupby(['model','window']):
        df_sub = df_sub.sort_values('sub')
        t,pval = stats.ttest_1samp(df_sub['score'].values,0.5,)
        
        results['model'].append(model)
        results['window'].append(window)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)-1)
    results = pd.DataFrame(results)
    temp = []
    for model, df_sub in results.groupby('model'):
        idx_sort = np.argsort(df_sub['pval'])
        for name in results.columns:
            df_sub[name] = df_sub[name].values[idx_sort]
        convert = MCPConverter(pvals=df_sub['pval'].values)
        df_pvals = convert.adjust_many()
        df_sub['ps_corrected'] = df_pvals['bonferroni'].values
        temp.append(df_sub)
    results = pd.concat(temp)
    results.sort_values(['model','window']).to_csv('../results/ATT_ttest_RT_features (scipy).csv')
    
    ##########################################################################
    id_vars             = ['model',
                           'score',
                           'sub',
                           'window',]
    value_vars          =[
                          'RT_correct',
                          'RT_awareness',
                          'RT_confidence',
                          ]
    df_post             = post_processing(df,id_vars,value_vars)
#    c                   = df_post.groupby(['Subjects','Models','Window','Attributes']).mean().reset_index()
    c                   = df_post[df_post['Window']<5]
    df_temp             = preparation(c)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'att,RT features,feature importance.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    df_temp.to_csv(os.path.join(df_dir,'att,RT features,feature importance.csv'),index=False)


















































