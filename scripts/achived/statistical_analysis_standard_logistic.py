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
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols#,mixedlm
from statsmodels.stats.anova import anova_lm
from utils import eta_squared,omega_squared,resample_ttest_2sample,MCPConverter
from itertools import combinations
sns.set_style('whitegrid')
sns.set_context('poster')
saving_dir = '../figures/'
df_dir = '../results/for_spss'
def post_processing(df):
    feature_names = [name for name in df.columns if 'coef' in name] # the feature names with "coef" in them
    feature_name_wk = feature_names[1:] # take the intercept out
    working_df = df[feature_name_wk] # 
    for name in feature_name_wk:
        working_df[name] = working_df[name].apply(np.exp)
    new_col_names = {name:name[:-5] for name in feature_name_wk}
    working_df['model'] = 'logistic'
    working_df['window'] = df['window']
    working_df = working_df.rename(new_col_names,axis='columns')
    df_plot = pd.melt(working_df,id_vars = ['model','window'],
                 value_vars = new_col_names.values())
    df_plot.columns = ['Model','Window','Coefficients','Odd_Ratio']
    return df_plot
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
    for ((window,feature),df_sub) in c.groupby(['Window','Coefficients']):
        df_temp['{}_win{}_{}'.format('logistic',window,feature)] = df_sub['Odd_Ratio'].values
    df_temp = pd.DataFrame(df_temp)
    return df_temp
"""
Take the exponential of each of the coefficients to generate the odds ratios. 
This tells you how a 1 unit increase or decrease in a variable affects the odds of being high POS.
"""
if __name__ == '__main__':
    results = []
    aov_tables = []
    ##########################################################################################################################################
    pos                 = pd.read_csv('../results/pos_logistic_statsmodel_6_features.csv')
    att                 = pd.read_csv('../results/att_logistic_statsmodel_6_features.csv')
    
    df                  = pos.copy()
    df_plot             = post_processing(df) # process the dataframe with melt or sth
    df_plot             = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)] # get the window
    df_temp             = preparation(df_plot)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'pos,6 features,odd ratio.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    
    ######
    df              = att.copy()
    df_plot         = post_processing(df)
    df_plot         = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    df_temp             = preparation(df_plot)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'att,6 features,odd ratio.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    
    ###################################################################################
    ###################  3 judgement features #########################################
    ###################################################################################
    pos             = pd.read_csv('../results/pos_logistic_statsmodel_3_1_features.csv')
    att             = pd.read_csv('../results/att_logistic_statsmodel_3_1_features.csv')
    
    df              = pos.copy()
    df_plot         = post_processing(df)
    df_plot         = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    df_temp         = preparation(df_plot)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'pos,judgment features,odd ratio.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    
    ######
    df              = att.copy()
    df_plot         = post_processing(df)
    df_plot         = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    df_temp         = preparation(df_plot)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'att,judgment features,odd ratio.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    # no main effect of coefficients nor interaction
    ###############################################################################################
    #############################  RT as features #################################################
    ###############################################################################################
    pos             = pd.read_csv('../results/pos_logistic_statsmodel_RT_features.csv')
    att             = pd.read_csv('../results/att_logistic_statsmodel_RT_features.csv')
    
    df              = pos.copy()
    df_plot         = post_processing(df)
    df_plot         = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    df_temp         = preparation(df_plot)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'pos,RT features,odd ratio.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()
    ######
    df              = att.copy()
    df_plot         = post_processing(df)
    df_plot         = df_plot[(df_plot['Window']>0) & (df_plot['Window']<4)]
    df_temp         = preparation(df_plot)
    writer              = pd.ExcelWriter(os.path.join(df_dir,'att,RT features,odd ratio.xlsx'))
    df_temp.to_excel(writer,'sheet1',index=False);writer.save()



















































