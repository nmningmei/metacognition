# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:02:16 2018

@author: ning
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
result_dir = '../results/'


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
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correct          = [],
               awareness        = [],
               confidence       = [],
               RT_correct       = [],
               RT_awareness     = [],
               RT_confidence    = [],
               r2               = [],
               intercept        = [],
               )
# use all 6 possible features
for n_back in range(11): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        df_sub['intercept'] = 1
        feature_names = ['intercept',
                         'correct',
                         'awareness',
                         'confidence',
                         'RT_correct',
                         'RT_awareness',
                         'RT_confidence']
        target_name = 'success'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        df_  = pd.DataFrame(features,columns = feature_names)
        df_[target_name] = targets
        model = sm.Logit(df_[target_name],df_[feature_names])
        temp = model.fit(method='lbfgs')
        results['sub'].append(participant)
        results['model'].append('logistic')
        results['score'].append([temp.bic,temp.aic])
        results['window'].append(n_back)
        for name in feature_names:
            results[name].append([temp.params[name],
                                  temp.bse[name],
                                  temp.tvalues[name],
                                  temp.pvalues[name],
                                  temp.conf_int().loc[name][0],
                                  temp.conf_int().loc[name][1],
                                 ])
        results['r2'].append(temp.prsquared)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
for name in feature_names:
    temp = c[name].to_frame()
    temp[name+'_coef'] = np.vstack(temp[name].values)[:,0]
    temp[name+'_se'] = np.vstack(temp[name].values)[:,1]
    temp[name+'_tval'] = np.vstack(temp[name].values)[:,2]
    temp[name+'_pval'] = np.vstack(temp[name].values)[:,3]
    temp[name+'_lower'] = np.vstack(temp[name].values)[:,4]
    temp[name+'_upper'] = np.vstack(temp[name].values)[:,5]
    for k_name in temp.columns[1:]:
        c[k_name] = temp[k_name].values
    c = c.drop(name,axis=1)
c.to_csv(os.path.join(result_dir,'pos_logistic_statsmodel_6_features.csv'),index=False)
c = pd.read_csv(os.path.join(result_dir,'pos_logistic_statsmodel_6_features.csv'))
j = c.groupby('window').mean().reset_index()
j.to_csv(os.path.join(result_dir,'pos_logistic_statsmodel_mean_6_features.csv'),index=False)
##############################################################################################################
# 3 judgement features
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
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correct          = [],
               awareness        = [],
               confidence       = [],
               r2               = [],
               intercept        = [],
               )
# use all 6 possible features
for n_back in range(11): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        df_sub['intercept'] = 1
        feature_names = ['intercept',
                         'correct',
                         'awareness',
                         'confidence',]
        target_name = 'success'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        df_  = pd.DataFrame(features,columns = feature_names)
        df_[target_name] = targets
        model = sm.Logit(df_[target_name],df_[feature_names])
        temp = model.fit(method='lbfgs')
        results['sub'].append(participant)
        results['model'].append('logistic')
        results['score'].append([temp.bic,temp.aic])
        results['window'].append(n_back)
        for name in feature_names:
            results[name].append([temp.params[name],
                                  temp.bse[name],
                                  temp.tvalues[name],
                                  temp.pvalues[name],
                                  temp.conf_int().loc[name][0],
                                  temp.conf_int().loc[name][1],
                                 ])
        results['r2'].append(temp.prsquared)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
for name in feature_names:
    temp = c[name].to_frame()
    temp[name+'_coef'] = np.vstack(temp[name].values)[:,0]
    temp[name+'_se'] = np.vstack(temp[name].values)[:,1]
    temp[name+'_tval'] = np.vstack(temp[name].values)[:,2]
    temp[name+'_pval'] = np.vstack(temp[name].values)[:,3]
    temp[name+'_lower'] = np.vstack(temp[name].values)[:,4]
    temp[name+'_upper'] = np.vstack(temp[name].values)[:,5]
    for k_name in temp.columns[1:]:
        c[k_name] = temp[k_name].values
    c = c.drop(name,axis=1)
c.to_csv(os.path.join(result_dir,'pos_logistic_statsmodel_3_1_features.csv'),index=False)
c = pd.read_csv(os.path.join(result_dir,'pos_logistic_statsmodel_3_1_features.csv'))
j = c.groupby('window').mean().reset_index()
j.to_csv(os.path.join(result_dir,'pos_logistic_statsmodel_mean_3_1_features.csv'),index=False)
#####################################################################################################################################
# RT features
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
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               RT_correct       = [],
               RT_awareness     = [],
               RT_confidence    = [],
               r2               = [],
               intercept        = [],
               )
# use all 6 possible features
for n_back in range(11): # loop through the number of trials looking back
    for participant,df_sub in df.groupby('participant'):# for each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        df_sub['intercept'] = 1
        feature_names = ['intercept',
                         'RT_correct',
                         'RT_awareness',
                         'RT_confidence']
        target_name = 'success'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        df_  = pd.DataFrame(features,columns = feature_names)
        df_[target_name] = targets
        model = sm.Logit(df_[target_name],df_[feature_names])
        temp = model.fit(method='lbfgs')
        results['sub'].append(participant)
        results['model'].append('logistic')
        results['score'].append([temp.bic,temp.aic])
        results['window'].append(n_back)
        for name in feature_names:
            results[name].append([temp.params[name],
                                  temp.bse[name],
                                  temp.tvalues[name],
                                  temp.pvalues[name],
                                  temp.conf_int().loc[name][0],
                                  temp.conf_int().loc[name][1],
                                 ])
        results['r2'].append(temp.prsquared)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
for name in feature_names:
    temp = c[name].to_frame()
    temp[name+'_coef'] = np.vstack(temp[name].values)[:,0]
    temp[name+'_se'] = np.vstack(temp[name].values)[:,1]
    temp[name+'_tval'] = np.vstack(temp[name].values)[:,2]
    temp[name+'_pval'] = np.vstack(temp[name].values)[:,3]
    temp[name+'_lower'] = np.vstack(temp[name].values)[:,4]
    temp[name+'_upper'] = np.vstack(temp[name].values)[:,5]
    for k_name in temp.columns[1:]:
        c[k_name] = temp[k_name].values
    c = c.drop(name,axis=1)
c.to_csv(os.path.join(result_dir,'pos_logistic_statsmodel_RT_features.csv'),index=False)
c = pd.read_csv(os.path.join(result_dir,'pos_logistic_statsmodel_RT_features.csv'))
j = c.groupby('window').mean().reset_index()
j.to_csv(os.path.join(result_dir,'pos_logistic_statsmodel_mean_RT_features.csv'),index=False)
#####################################################################################################
#####################################################################################################
#####################################################################################################
################################       ATT                      #####################################
#####################################################################################################
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
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correct          = [],
               awareness        = [],
               confidence       = [],
               RT_correct       = [],
               RT_awareness     = [],
               RT_confidence    = [],
               r2               = [],
               intercept        = [],
               )
# use all 6 features
for n_back in range(11):# loop through the number of trials you want to look back
    for participant,df_sub in df.groupby('participant'):# loop through each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        df_sub['intercept'] = 1
        feature_names = ['intercept',
                         'correct',
                         'awareness',
                         'confidence',
                         'RT_correct',
                         'RT_awareness',
                         'RT_confidence']
        target_name = 'attention'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        df_  = pd.DataFrame(features,columns = feature_names)
        df_[target_name] = targets
        model = sm.Logit(df_[target_name],df_[feature_names])
        temp = model.fit(method='lbfgs')
        results['sub'].append(participant)
        results['model'].append('logistic')
        results['score'].append([temp.bic,temp.aic])
        results['window'].append(n_back)
        for name in feature_names:
            results[name].append([temp.params[name],
                                  temp.bse[name],
                                  temp.tvalues[name],
                                  temp.pvalues[name],
                                  temp.conf_int().loc[name][0],
                                  temp.conf_int().loc[name][1],
                                 ])
        results['r2'].append(temp.prsquared)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
for name in feature_names:
    temp = c[name].to_frame()
    temp[name+'_coef'] = np.vstack(temp[name].values)[:,0]
    temp[name+'_se'] = np.vstack(temp[name].values)[:,1]
    temp[name+'_tval'] = np.vstack(temp[name].values)[:,2]
    temp[name+'_pval'] = np.vstack(temp[name].values)[:,3]
    temp[name+'_lower'] = np.vstack(temp[name].values)[:,4]
    temp[name+'_upper'] = np.vstack(temp[name].values)[:,5]
    for k_name in temp.columns[1:]:
        c[k_name] = temp[k_name].values
    c = c.drop(name,axis=1)
c.to_csv(os.path.join(result_dir,'att_logistic_statsmodel_6_features.csv'),index=False)
c = pd.read_csv(os.path.join(result_dir,'att_logistic_statsmodel_6_features.csv'))
j = c.groupby('window').mean().reset_index()
j.to_csv(os.path.join(result_dir,'att_logistic_statsmodel_mean_6_features.csv'),index=False)
#######################################################################################################################################
# 3 judgement features
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
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               correct          = [],
               awareness        = [],
               confidence       = [],
               r2               = [],
               intercept        = [],
               )
# use all 6 features
for n_back in range(11):# loop through the number of trials you want to look back
    for participant,df_sub in df.groupby('participant'):# loop through each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        df_sub['intercept'] = 1
        feature_names = ['intercept',
                         'correct',
                         'awareness',
                         'confidence',
                         ]
        target_name = 'attention'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        df_  = pd.DataFrame(features,columns = feature_names)
        df_[target_name] = targets
        model = sm.Logit(df_[target_name],df_[feature_names])
        temp = model.fit(method='lbfgs')
        results['sub'].append(participant)
        results['model'].append('logistic')
        results['score'].append([temp.bic,temp.aic])
        results['window'].append(n_back)
        for name in feature_names:
            results[name].append([temp.params[name],
                                  temp.bse[name],
                                  temp.tvalues[name],
                                  temp.pvalues[name],
                                  temp.conf_int().loc[name][0],
                                  temp.conf_int().loc[name][1],
                                 ])
        results['r2'].append(temp.prsquared)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
for name in feature_names:
    temp = c[name].to_frame()
    temp[name+'_coef'] = np.vstack(temp[name].values)[:,0]
    temp[name+'_se'] = np.vstack(temp[name].values)[:,1]
    temp[name+'_tval'] = np.vstack(temp[name].values)[:,2]
    temp[name+'_pval'] = np.vstack(temp[name].values)[:,3]
    temp[name+'_lower'] = np.vstack(temp[name].values)[:,4]
    temp[name+'_upper'] = np.vstack(temp[name].values)[:,5]
    for k_name in temp.columns[1:]:
        c[k_name] = temp[k_name].values
    c = c.drop(name,axis=1)
c.to_csv(os.path.join(result_dir,'att_logistic_statsmodel_3_1_features.csv'),index=False)
c = pd.read_csv(os.path.join(result_dir,'att_logistic_statsmodel_3_1_features.csv'))
j = c.groupby('window').mean().reset_index()
j.to_csv(os.path.join(result_dir,'att_logistic_statsmodel_mean_3_1_features.csv'),index=False)
####################################################################################################################################
# RT features
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
results = dict(sub              = [],
               model            = [],
               score            = [],
               window           = [],
               RT_correct       = [],
               RT_awareness     = [],
               RT_confidence    = [],
               r2               = [],
               intercept        = [],
               )
# use all 6 features
for n_back in range(11):# loop through the number of trials you want to look back
    for participant,df_sub in df.groupby('participant'):# loop through each subject
        # make sure all the attributes are either 0 or 1
        df_sub.loc[:,'attention' ] = df_sub.loc[:,'attention' ].values - 1
        df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
        df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
        df_sub['intercept'] = 1
        feature_names = ['intercept',
                         'RT_correct',
                         'RT_awareness',
                         'RT_confidence']
        target_name = 'attention'
        features, targets = [],[]
        for block, df_block in df_sub.groupby('blocks'):
            # preparing the features and target by shifting the feature columns up
            # and shifting the target column down
            feature       = (df_block[feature_names].shift(n_back) # shift downward so that the first n_back rows are gone
                                                    .dropna() # since some of rows are gone, so they are nans
                                                    .values # I only need the matrix not the data frame
                    )
            target        = (df_block[target_name].shift(-n_back) # same thing for the target, but shifting upward, and the last n_back rows are gone
                                                  .dropna()
                                                  .values
                         )
            features.append(feature)
            targets.append(target)
        features          = np.concatenate(features)
        targets           = np.concatenate(targets)
        df_  = pd.DataFrame(features,columns = feature_names)
        df_[target_name] = targets
        model = sm.Logit(df_[target_name],df_[feature_names])
        temp = model.fit(method='lbfgs')
        results['sub'].append(participant)
        results['model'].append('logistic')
        results['score'].append([temp.bic,temp.aic])
        results['window'].append(n_back)
        for name in feature_names:
            results[name].append([temp.params[name],
                                  temp.bse[name],
                                  temp.tvalues[name],
                                  temp.pvalues[name],
                                  temp.conf_int().loc[name][0],
                                  temp.conf_int().loc[name][1],
                                 ])
        results['r2'].append(temp.prsquared)
c = pd.DataFrame(results) # tansform a dictionary object to a data frame
for name in feature_names:
    temp = c[name].to_frame()
    temp[name+'_coef'] = np.vstack(temp[name].values)[:,0]
    temp[name+'_se'] = np.vstack(temp[name].values)[:,1]
    temp[name+'_tval'] = np.vstack(temp[name].values)[:,2]
    temp[name+'_pval'] = np.vstack(temp[name].values)[:,3]
    temp[name+'_lower'] = np.vstack(temp[name].values)[:,4]
    temp[name+'_upper'] = np.vstack(temp[name].values)[:,5]
    for k_name in temp.columns[1:]:
        c[k_name] = temp[k_name].values
    c = c.drop(name,axis=1)
c.to_csv(os.path.join(result_dir,'att_logistic_statsmodel_RT_features.csv'),index=False)
c = pd.read_csv(os.path.join(result_dir,'att_logistic_statsmodel_RT_features.csv'))
j = c.groupby('window').mean().reset_index()
j.to_csv(os.path.join(result_dir,'att_logistic_statsmodel_mean_RT_features.csv'),index=False)




















































