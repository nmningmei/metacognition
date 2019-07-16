#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:07:58 2018

@author: nmei

in exp2 (e2) there were 3 possible awareness ratings ( (e.g. 1- no experience, 2 brief glimpse 3 almost clear or clear perception)
BUT if can make a binary classification by focussing on 1 and 2 which are the majority of the trials.


"""
if __name__ == '__main__':
    import os
    import pandas as pd
    import numpy as np
    import utils
    # define result saving directory
    dir_saving      = 'results_e2'
    if not os.path.exists(dir_saving):
        os.mkdir(dir_saving)
    
    try:# the subject level processing
        df1         = pd.read_csv('e2.csv').iloc[:,1:]
    except: # when I test the script
        df1         = pd.read_csv('../e2.csv').iloc[:,1:]
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
    # preallocate the data frame structure
    results         = dict(sub              = [],
                           model            = [],
                           score            = [],
                           window           = [],
                           correctness      = [],
                           awareness        = [],
                           confidence       = [],
                           chance           = [],
                           )
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names   = [
                         'correctness',
                         'awareness',
                         'confidence',
                         ]
    target_name     = 'correctness'
    experiment      = 'e2'
    # for some of the variables, we need to rescale them to a more preferable range like 0-1
    name_for_scale  = ['awareness']
    # 'ack', 'cc', 'ck', 'cpj', 'em', 'es', 'fd', 'jmac', 'lidia', 'ls','mimi', 'pr', 'pss', 'sva', 'tj'
    # get one of the participants' data
    participant = 'cpj'
    df_sub          = df[df['participant'] == participant]
    # pick 1- no experience, 2 brief glimpse for binary classification
    df_sub          = df_sub[df_sub['awareness'] != 3]
    # for 1-back to 4-back
    for n_back in np.arange(1,5):
        # experiment score
        results     = utils.classification(
                                          df_sub.dropna(),                  # take out nan rows
                                          feature_names,                    # feature columns
                                          target_name,                      # target column
                                          results,                          # the saving structure
                                          participant,                      # participant's name
                                          experiment,                       # experiment name
                                          window = n_back,                  # N-back
                                          chance = False,                   # it is NOT estimating the chance level but the empirical classification experiment
                                          name_for_scale = name_for_scale   # scale some of the variables
                                          )
        # empirical chance level
        results     = utils.classification(
                                          df_sub.dropna(),
                                          feature_names,
                                          target_name,
                                          results,
                                          participant,
                                          experiment,
                                          window = n_back,
                                          chance = True,                    # it is to estimate the empirical chance level
                                          name_for_scale = name_for_scale
                                          )
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(os.path.join(dir_saving,'{}.csv'.format(participant)))















































