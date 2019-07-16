#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:06:07 2018

@author: nmei

correlate the predicted awareness with the correctness

"""

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import utils
    from sklearn.model_selection import LeaveOneOut,cross_val_predict
    from sklearn.utils import shuffle
    from scipy import stats
    # define result saving directory
    dir_saving      = 'results_e1'
    if not os.path.exists(dir_saving):
        os.makedirs(dir_saving)
    
    try:# the subject level processing
        df1         = pd.read_csv('e1.csv').iloc[:,1:]
    except: # when I test the script
        df1         = pd.read_csv('../../e1.csv').iloc[:,1:]
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
                           corre            = [],
                           window           = [],
                           pval             = [],
                           )
    results['p(correct|awareness)'] = []
    results['p(correct|unawareness)'] = []
    results['p(incorrect|awareness)'] = []
    results['p(incorrect|unawareness)'] = []
    results['p(correct)'] = []
    results['p(incorrect)'] = []
    results['p(aware)'] = []
    results['p(unaware)'] = []
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
    participant = 'fcm'
    df_sub          = df[df['participant'] == participant]
    # for 1-back to 4-back
    for n_back in np.arange(1,5):
        X,y,groups = utils.get_features_targets_groups(
                                df_sub.dropna(),
                                n_back                  = n_back,
                                names                   = name_for_scale,
                                independent_variables   = feature_names,
                                dependent_variable      = [target_name,'correctness'])
        X,y,groups = shuffle(X,y,groups)
        y,correctness = y[:,0],y[:,1]
        for model_name,model in utils.make_clfs().items():
            cv = LeaveOneOut()
            print('{}-back,{}'.format(n_back,model_name))
            preds = cross_val_predict(model,X,y,groups=groups,cv=cv,method='predict',verbose=2,n_jobs=4)
            df_pred_ = pd.DataFrame(np.vstack([preds,correctness]).T,columns = ['preds','correct'])
            p_correct = float(np.sum(correctness == 1)+1) / (len(correctness)+1)
            p_incorrect = float(np.sum(correctness == 0)+1) / (len(correctness)+1)
            p_aware = float(np.sum(preds == 1)+1) / (len(preds)+1)
            p_unaware = float(np.sum(preds == 0)+1) / (len(preds)+1)
            p_correct_aware = float(np.sum(np.logical_and(correctness == 1, preds == 1))+1) / (len(df_pred_)+1)
            p_correct_unaware = float(np.sum(np.logical_and(correctness == 1, preds == 0))+1) / (len(df_pred_)+1)
            p_incorrect_aware = float(np.sum(np.logical_and(correctness == 0, preds == 1))+1) / (len(df_pred_)+1)
            p_incorrect_unaware = float(np.sum(np.logical_and(correctness == 0, preds == 0))+1) / (len(df_pred_)+1)
            correlation,pval = stats.spearmanr(preds,correctness)
            results['sub'].append(participant)
            results['model'].append(model_name)
            results['corre'].append(correlation)
            results['pval'].append(pval)
            results['p(correct|awareness)'].append(p_correct_aware/p_aware)
            results['p(correct|unawareness)'].append(p_correct_unaware/p_unaware)
            results['p(incorrect|awareness)'].append(p_incorrect_aware/p_aware)
            results['p(incorrect|unawareness)'].append(p_incorrect_unaware/p_unaware)
            results['p(correct)'].append(p_correct)
            results['p(incorrect)'].append(p_incorrect)
            results['p(aware)'].append(p_aware)
            results['p(unaware)'].append(p_unaware)
            results['window'].append(n_back)
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(os.path.join(dir_saving,'{}.csv'.format(participant)),index=False)

































