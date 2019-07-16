#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:24:14 2018

@author: nmei
exp1 (e1) there were 2 possible awareness ratings (unaware and aware of the orientation)


"""
if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import utils
    # define result saving directory
    dir_saving      = 'hierarchical_regression'
    if not os.path.exists(dir_saving):
        os.makedirs(dir_saving)
    # preallocate the data frame structure
    results         = dict(sub              = [],
                           model            = [],
                           window           = [],
                           model1           = [],
                           model2           = [],
                           model3           = [],
                           model4           = [],
                           sig21            = [],
                           sig32            = [],
                           sig42            = [],
                           experiment       = [],)
    ############################################################################################
    df1         = pd.read_csv('e1.csv').iloc[:,1:]
    df              = df1.copy()
    # select the columns that I need
    df              = df[['blocks.thisN',
                          'trials.thisN',
                          'key_resp_2.keys',
                          'resp.corr',
                          'resp_mrating.keys',
                          'participant',]]
    # rename the columns
    df.columns      = ['blocks',
                       'trials',
                       'awareness',
                       'correctness',
                       'confidence',
                       'participant',]
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'awareness'
    experiment      = 'e1'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    # ['ah', 'av', 'bj', 'cm', 'db', 'ddb', 'fcm', 'kf', 'kk', 'ml', 'qa','sk', 'yv']
    # get one of the participants' data
    for participant,df_sub in df.groupby('participant'):
        # for 1-back to 4-back
        results = utils.hierarchical_regression(df_sub,
                            name_for_scale,
                            feature_names,
                            target_name,
                            participant,
                            results,
                            experiment,
                            n_backs = [1,4])
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(os.path.join(dir_saving,'hierarchical regression.csv'),
                               index=False)
    ######################################################################################################
    df2         = pd.read_csv('e2.csv').iloc[:,1:]
    df              = df2.copy()
    # select the columns that I need
    df              = df[['blocks.thisN',
                          'trials.thisN',
                          'key_resp_2.keys',
                          'resp.corr',
                          'resp_mrating.keys',
                          'participant',]]
    # rename the columns
    df.columns      = ['blocks',
                       'trials',
                       'awareness',
                       'correctness',
                       'confidence',
                       'participant',]
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'awareness'
    experiment      = 'e2'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    for participant,df_sub in df.groupby('participant'):
        df_sub          = df_sub[df_sub['awareness'] != 3]
        # for 1-back to 4-back
        results = utils.hierarchical_regression(df_sub,
                            name_for_scale,
                            feature_names,
                            target_name,
                            participant,
                            results,
                            experiment,
                            n_backs = [1,4])
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(os.path.join(dir_saving,'hierarchical regression.csv'),
                               index=False)
    ###############################################################################################
    df3         = pd.read_csv('e3.csv').iloc[:,1:]
    df              = df3.copy()
    # select the columns that I need
    df              = df[['blocks.thisN',
                          'trials.thisN',
                          'key_resp_2.keys',
                          'resp.corr',
                          'resp_mrating.keys',
                          'participant',]]
    # rename the columns
    df.columns      = ['blocks',
                       'trials',
                       'awareness',
                       'correctness',
                       'confidence',
                       'participant',]
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'awareness'
    experiment      = 'e3'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    for participant,df_sub in df.groupby('participant'):
        # for 1-back to 4-back
        results = utils.hierarchical_regression(df_sub,
                            name_for_scale,
                            feature_names,
                            target_name,
                            participant,
                            results,
                            experiment,
                            n_backs = [1,4])
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(os.path.join(dir_saving,'hierarchical regression.csv'),
                               index=False)






























