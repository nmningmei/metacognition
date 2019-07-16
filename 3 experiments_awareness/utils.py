#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:30:19 2018

@author: nmei
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedShuffleSplit,
                                     LeaveOneGroupOut,
                                     cross_val_score)
from sklearn.metrics import roc_auc_score,accuracy_score
try:
    from tqdm import tqdm
except:
    print("why is tqdm not installed?")
try:
    import pymc3 as pm
except:
    print("you don't have pymc3 or you haven't set up the environment")
#import theano.tensor as t
from itertools import islice

def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def resample_ttest(x,baseline = 0.5,n_ps = 100,n_permutation = 5000,one_tail = False):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    """
    import numpy as np
    experiment = np.mean(x) # the mean of the observations in the experiment
    experiment_diff = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    # newexperiment = np.mean(experiment_diff) # just look at the new mean and make sure it is at the baseline
    # simulate/bootstrap null hypothesis distribution
    # 1st-D := number of sample same as the experiment
    # 2nd-D := within one permutation resamping, we perform resampling same as the experimental samples,
    # but also repeat this one sampling n_permutation times
    # 3rd-D := repeat 2nd-D n_ps times to obtain a distribution of p values later
    temp = np.random.choice(experiment_diff,size=(x.shape[0],n_permutation,n_ps),replace=True)
    temp = temp.mean(0)# take the mean over the sames because we only care about the mean of the null distribution
    # along each row of the matrix (n_row = n_permutation), we count instances that are greater than the observed mean of the experiment
    # compute the proportion, and we get our p values
    
    if one_tail:
        ps = (np.sum(temp >= experiment,axis=0)+1.) / (n_permutation + 1.)
    else:
        ps = (np.sum(np.abs(temp) >= np.abs(experiment),axis=0)+1.) / (n_permutation + 1.)
    return ps
def resample_ttest_2sample(a,b,n_ps=100,n_permutation=5000,one_tail=False,match_sample_size = True,):
    if match_sample_size:# when the N is matched
        difference = a - b
        ps = resample_ttest(difference,baseline=0,n_ps=n_ps,n_permutation=n_permutation,one_tail=one_tail)
        return ps
    else: # when the N is not matched
        difference              = np.mean(a) - np.mean(b)
        concatenated            = np.concatenate([a,b])
        np.random.shuffle(concatenated)
        temp                    = np.zeros((n_permutation,n_ps))
        try:
            iterator            = tqdm(range(n_ps),desc='ps')
        except:
            iterator            = range(n_ps)
        for n_p in iterator:
            for n_permu in range(n_permutation):
                idx_a           = np.random.choice([0,1],
                                                   size = (len(a)+len(b)),
                                                   p    = [float(len(a))/(len(a)+len(b)),
                                                           float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                idx_b           = np.logical_not(idx_a)
                d               = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                if np.isnan(d):
                    idx_a       = np.random.choice([0,1],
                                                   size     = (len(a)+len(b)),
                                                   p        = [float(len(a))/(len(a)+len(b)),
                                                               float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                    idx_b       = np.logical_not(idx_a)
                    d = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                temp[n_permu,n_p] = d
        if one_tail:
            ps = (np.sum(temp >= difference,axis=0)+1.) / (n_permutation + 1.)
        else:
            ps = (np.sum(np.abs(temp) >= np.abs(difference),axis=0)+1.) / (n_permutation + 1.)
        return ps
################################################################
### multi-comparison method, reference from internet!!!!!!!!!!!!################
    ################################################
import statsmodels as sms
class MCPConverter(object):
    """
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores=None):
        self.pvals = pvals
        self.zscores = zscores
        self.len = len(pvals)
        if zscores is not None:
            srted = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals = srted[:, 0]
            self.sorted_zscores = srted[:, 1]
        else:
            self.sorted_pvals = np.array(sorted(pvals.copy()))
        self.order = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method="holm"):
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method is "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method is "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method is "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method is "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods=["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method is not "lfdr":
                    df[method] = self.adjust(method)
        return df
# https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
def label_diff(i,j,text,X,Y,ax):
    x = (X[i]+X[j])/2
    y = 1.1*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'linewidth':2}
    ax.annotate(text, xy=(X[i],y+7), zorder=10)
    ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
################## from https://github.com/parrt/random-forest-importances/blob/master/src/rfpimp.py#L237 #############
################## reference http://explained.ai/rf-importance/index.html #############################################
def oob_classifier_accuracy(rf, X_train, y_train):
    """
    Adjusted... 
    Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
    classifier. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
    """
    try:
        X                   = X_train.values
    except:
        X                   = X_train.copy()
    try:
        y                   = y_train.values
    except: 
        y                   = y_train.copy()

    n_samples               = len(X)
    n_classes               = len(np.unique(y))
    # preallocation
    predictions             = np.zeros((n_samples, n_classes))
    for tree in rf.estimators_: # for each decision tree in the random forest - I have put 1 tree in the forest
        # Private function used to _parallel_build_trees function.
        unsampled_indices   = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds          = tree.predict_proba(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds

    predicted_class_indexes = np.argmax(predictions, axis=1)# threshold the probabilistic predictions
    predicted_classes       = [rf.classes_[i] for i in predicted_class_indexes] # use the thresholded indicies to obtain a binary prediction

    oob_score               = sum(y==predicted_classes) / float(len(y))
    return oob_score
def sample(X_valid, y_valid, n_samples):
    """
    Not sure what this is doing
    Only if the n_sample is less than the total number of samples, it subsamples the data???? Maybe?
    """
    if n_samples < 0: 
        n_samples                   = len(X_valid)
    n_samples                       = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix                          = np.random.choice(len(X_valid), n_samples)
        X_valid                     = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid                     = y_valid.iloc[ix].copy(deep=False)
    return X_valid, y_valid
def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=5000):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    X_sample, y_sample          = shuffle(X_train, y_train)
    # get a baseline out-of-bag sampled decoding score
    try:
        baseline                    = metric(rf, X_sample, y_sample)
    except:
        try:
            baseline                    = metric(y_sample,rf.predict_proba(X_sample)[:,-1])
        except:
            baseline                    = metric(y_sample,rf.predict_proba(X_sample)[:,-1]>0.5)
    # make suer that we work on the copy of the raw data
    X_train                     = X_sample.copy() # shallow copy
    X_train                     = pd.DataFrame(X_train)
    y_train                     = y_sample
    imp                         = []
#    for n_ in range(100):
#        imp_temp = []
#        for col in X_train.columns: # for each feature
#            save                = X_train[col].copy() # save the original
#            X_train[col]        = np.random.uniform(save.min(),save.max(),size=save.shape) # reorder
#            # oob score after reorder 1 and only 1 feature. 
#            # In orther words, how much information is gone when the feature becomes unimformative
#            try:
#                m                   = metric(rf, X_train, y_train)
#            except:
#                m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1])
#            X_train[col]        = save # restore the feature
#            imp_temp.append(baseline - m)
#        imp.append(imp_temp)
    
    for col in X_train.columns: # for each feature
        save                = X_train[col].copy() # save the original
        X_train[col]        = np.random.uniform(save.min(),save.max(),size=save.shape) # reorder
        # oob score after reorder 1 and only 1 feature. 
        # In orther words, how much information is gone when the feature becomes unimformative
        try:
            m                   = metric(rf, X_train, y_train)
        except:
            try:
                m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1])
            except:
                m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1]>0.5)
        X_train[col]        = save # restore the feature
        imp.append(baseline - m)
        
    return np.array(imp)#.mean(0)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def permutation_importances(rf, X_train, y_train, metric, n_samples = 5000,feature_names = None):
    """
    Call the function, and just to make it pretty
    """
    imp = permutation_importances_raw(rf, X_train, y_train, metric, n_samples)
    imp = softmax(imp)
    I = pd.DataFrame(data={'Feature':feature_names, 'Importance':imp})
    I = I.set_index('Feature')
    return I



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def make_clfs():
    """
    Generate a dictionary with initialized scikit-learn models with their names
    1. Logistic Regression
    2. Decision Tree
    """
    return dict(
        # Sklearn applies automatic regularization, 
        # so we’ll set the parameter C to a large value to emulate no regularization
        LogisticRegression      = LogisticRegression(  C                      = 1e100,
                                                       max_iter               = int(1e3),# default was 100
                                                       tol                    = 0.0001, # default value
                                                       solver                 = 'liblinear', # default solver
                                                       random_state           = 12345,
                                                       class_weight           = 'balanced',
                                                        ),# end of logistic
        # I need to use random forest but with 1 tree because I use an external function to estimate the 
        # feature importance.
        RandomForestClassifier  = RandomForestClassifier(n_estimators         = 500, 
                                                         criterion            = 'entropy',
                                                         max_depth            = 1,
                                                         min_samples_leaf     = 5, # control for minimum sample per node
                                                         random_state         = 12345,
                                                         oob_score            = True,
                                                         n_jobs               = -1,
                                                         class_weight         = 'balanced',
                                                         )# end of Random Forest
        )# end of dictionary
def classification(df_,
                   feature_names,
                   target_name,
                   results,
                   participant,
                   experiment,
                   window = 0,
                   n_splits = 100,
                   chance = False,
                   name_for_scale = [],
                   ):
    """
    Since the classification is redundent after the features and targets are 
    ready, I would rather to make a function for the redundent part
    
    Inputs:
        df_                 : dataframe of a given subject
        feature_names       : names of features
        target_name         : name of target
        results             : dictionary like object, update itself every cross validation
        participant         : string, for updating the results dictionary
        experiment          : for graph of the tree model
        window              : integer value, for updating the results dictionary
        n_splits            : number of cross validation folds
        chance              : whether to estiamte the experimental score or empirical chance level
        naem_for_scale      : if there are features that need to be scaled
    return:
        results
    """
    features,targets,groups = get_features_targets_groups( 
            df_,
            n_back                  = window,
            names                   = name_for_scale,
            independent_variables   = feature_names,
           dependent_variable       = target_name)
    if chance:
        # to shuffle the features independent from the targets
        features         = shuffle(features)
    else:
        features,targets = shuffle(features,targets)

    cv                = StratifiedShuffleSplit(
             n_splits = n_splits,# random partition for NN times
            test_size = 0.2,# split 20 % of the data as test data and the rest will be training data
         random_state = 12345 # for reproducible purpose
         )
    # this is not for initialization but for iteration
    clfs              = make_clfs()
    # for each classifier, we fit-test cross validate the classifier and save the
    # classification score and the weights of the attributes
    for model,c in clfs.items():
        scores        = []
        weights       = []
        try:
            iterator = tqdm(enumerate(cv.split(features,targets)),desc='{}'.format(model))
        except:
            iterator = enumerate(cv.split(features,targets))
        for fold, (train,test) in iterator:
            clf         = make_clfs()[model]
            # fit the training data
            clf.fit(features[train],targets[train].ravel())
            # make predictions on the test data
            pred      = clf.predict_proba(features[test])[:,-1]
            # score the predictions
            try:
                score     = roc_auc_score(targets[test],pred)
            except:
                score     = accuracy_score(targets[test],pred>0.5)
            scores.append(score)
            try:
                weights.append(clf.coef_[0])
            except:
                try:
                    feature_importance = permutation_importances(clf,features[train],targets[train],roc_auc_score)#clf.feature_importances_#
                except:
                    feature_importance = permutation_importances(clf,features[train],targets[train],accuracy_score)#clf.feature_importances_#
                weights.append(feature_importance.values.flatten())
        results['sub'       ].append(participant          )
        results['model'     ].append(model                )
        results['score'     ].append(np.mean(scores)      )
        results['window'    ].append(window               )
        results['chance'    ].append(chance               )
        for iii,name in enumerate(feature_names):
            results[name    ].append(np.mean(weights,0)[iii])
        
        print('sub {:3},model {:22},window {:1},score={:.3f}'.format(
                participant,
                model,
                window,
                np.mean(scores)
                )
            )
    return results


def predict(instances, weights,intercept):
    weights = weights.reshape(-1,1)
    """Predict gender given weight (w) and height (h) values."""
    v = intercept + np.dot(instances,weights)
    return np.exp(v)/(1+np.exp(v))
def bayesian_logistic(df_,
                   feature_names,
                   target_name,
                   results,
                   participant,
                   experiment,
                   dot_dir,
                   window=0,
                   ):
    """
    Since the classification is redundent after the features and targets are 
    ready, I would rather to make a function for the redundent part
    
    Inputs:
        df_                 : dataframe of a given subject
        feature_names       : names of features
        target_name         : name of target
        results             : dictionary like object, update itself every cross validation
        participant         : string, for updating the results dictionary
        experiment          : for graph of the tree model
        dot_dit             : directory of the tree plots
        window              : integer value, for updating the results dictionary
    return:
        results
    """
    features, targets   = [],[]
    for block, df_block in df_.groupby('blocks'):
        # preparing the features and target by shifting the feature columns up
        # and shifting the target column down
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
    features            = np.concatenate(features)
    targets             = np.concatenate(targets)
    features, targets   = shuffle(features,targets)
    # for each classifier, we fit-test cross validate the classifier and save the
    # classification score and the weights of the attributes
    model_name               = 'bayes_logistic'
   
    # this is for initialization
    df_train        = pd.DataFrame(features,columns=feature_names)
    df_train[target_name]   = targets
    scaler = StandardScaler()
    
    for name in feature_names:
       if 'RT' in name:
           df_train[name] = scaler.fit_transform(df_train[name].values.reshape(-1,1))
    niter                   = 1000
    formula                 = '{} ~'.format(target_name)
    for name in feature_names:
        formula += ' + {}'.format(name)
    
    with pm.Model() as model:
        pm.glm.GLM.from_formula(formula, 
                                df_train, 
                                family=pm.glm.families.Binomial(),
                                )
        start               = pm.find_MAP(progressbar=False)
        try:
            step                = pm.NUTS(scaling=start,)
        except:
            step       = pm.Metropolis()
        trace               = pm.sample(  niter, 
                                          start         =start, 
                                          step          =step, 
                                          njobs         =4, 
                                          random_seed   =12345,
                                          progressbar=0,
                                          )
    df_trace                = pm.trace_to_dataframe(trace[niter//2:])
    intercept               = df_trace['Intercept'].mean()
    df_test                 = pd.DataFrame(features,
                                           columns      =feature_names)
    weights = df_trace.iloc[:,1:].values.mean(0)
    preds                   = predict(df_test.values,weights,intercept)
    # score the predictions
    score                   = roc_auc_score(targets,preds)
    results['sub'       ].append(participant          )
    results['model'     ].append(model_name                )
    results['score'     ].append(score      )
    results['window'    ].append(window               )
    for iii,name in enumerate(feature_names):
        results[name        ].append(df_trace[name].values.mean())
    
    print('sub {},model {},window {},score={:.3f}'.format(
            participant,
            model_name,
            window,
            score
            )
        )
    return results,trace
def post_processing(df,id_vars=[],value_vars=[]):
    """
    After classification, we want to see how each model weights on the attributes,
    thus, we need some ways to nornalize the weights for each subject so that 
    we could have a fair comparison between subjects and models
    Here, I decide to normize the vector of 3 attributes for each subject so that
    all these weight vectors are norm vecoters. In other words, the weights can
    be interpret as the directions they are pointing in the 3D space
    """
    df_ = df.copy()
    for feature_name in value_vars:
        df_[feature_name][df_['model'] == 'LogisticRegression'] = df_[feature_name][df['model'] == 'LogisticRegression'].apply(np.exp)
    df_post               = pd.melt(df_,
                                    id_vars         =id_vars,
                                    value_vars      =value_vars)
    df_post.columns       = ['Models',
                             'Scores',
                             'Subjects',
                             'Window',
                             'Attributes',
                             'Values',
                             ]
    return df_post
def post_processing2(df,names = []):
    """

    """
    # To 
    temp1                 = df[['sub',
                                'window',
                                'score',
                                'model',
                                names[0]
                                ]]
    temp1.loc[:,'value']  = temp1[names[0]]
    temp1['Attributions'] = names[0]
    temp2                 = df[['sub',
                                'window',
                                'score',
                                'model',
                                names[1]
                                ]]
    temp2.loc[:,'value']  = temp2[names[1]]
    temp2['Attributions'] = names[1]
    temp3                 = df[['sub',
                                'window',
                                'score',
                                'model',
                                names[2]
                                ]]
    temp3.loc[:,'value']  = temp3[names[2]]
    temp3['Attributions'] = names[2]
    df                    = pd.concat([temp1,
                                       temp2,
                                       temp3]).dropna(axis=1)
    return df

def multiple_pairwise_comparison(df,n_ps=100,n_permutation=5000,method='bonferoni'):
    from itertools import combinations
    results = dict(model        =[],
                   window       =[],
                   ps_mean      =[],
                   ps_std       =[],
                   larger       =[],
                   less         =[],
                   diff         =[],
                   )
    for (model,window), df_sub in df.groupby(['model','window']):
        name = list(df_sub.columns[:6])
        names = combinations(name,2)
        for (name1,name2) in names:

                if np.mean(df_sub[name1] - df_sub[name2]) >= 0:
                    ps = resample_ttest_2sample(df_sub[name1].values,
                                        df_sub[name2].values,
                                        n_ps=n_ps,
                                        n_permutation=n_permutation)
                    results['model'].append(model)
                    results['window'].append(window)
                    results['ps_mean'].append(ps.mean())
                    results['ps_std'].append(ps.std())
                    results['larger'].append(name1)
                    results['less'].append(name2)
                    results['diff'].append(np.mean(df_sub[name1] - df_sub[name2]))
                else:
                    ps = resample_ttest_2sample(df_sub[name2].values,
                                        df_sub[name1].values,
                                        n_ps=n_ps,
                                        n_permutation=n_permutation)
                    results['model'].append(model)
                    results['window'].append(window)
                    results['ps_mean'].append(ps.mean())
                    results['ps_std'].append(ps.std())
                    results['larger'].append(name2)
                    results['less'].append(name1)
                    results['diff'].append(np.mean(df_sub[name2] - df_sub[name1]))
    compar = pd.DataFrame(results)
    temp = []
    for (model,window),compar_sub in compar.groupby(['model','window']):
        idx_sort = np.argsort(compar_sub['ps_mean'])
        for name in compar_sub.columns:
            compar_sub[name] = compar_sub[name].values[idx_sort]
        convert = MCPConverter(compar_sub['ps_mean'].values)
        df_pvals = convert.adjust_many()
        compar_sub['ps_corrected'] = df_pvals[method].values
        temp.append(compar_sub)
    compar = pd.concat(temp)
    return compar
#def tinvlogit(x):
#    return t.exp(x) / (1 + t.exp(x))
def logistic_regression(df_working,sample_size=3000):
    independent_variables = df_working.columns[:-1]
    dependent_variable = df_working.columns[-1]
    traces,models = {},{}
    for name in independent_variables:
        with pm.Model() as model:
            pm.glm.GLM.from_formula('{} ~ {}'.format(dependent_variable,name),
                                       df_working, family=pm.glm.families.Binomial())
            start = pm.find_MAP()
            step = pm.NUTS(scaling=start)
            trace = pm.sample(sample_size,step,start,chains=3, tune=1000)
        traces[name]=trace
        models[name]=model
    return traces,models
def compute_r2(df, ppc, ft_endog):
    
    sse_model = (ppc['y'] - df[ft_endog].values)**2
    sse_mean = (df[ft_endog].values - np.random.choice(df[ft_endog],
                size=(ppc['y'].shape[0],df[ft_endog].shape[0])
                ))**2
    
    return 1 - (sse_model.sum(1) / sse_mean.sum(1))
def plot_traces(traces, retain=1000):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[-retain:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.summary(traces[-retain:]).iterrows()})

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')
def compute_ppc(trc, mdl, samples=500, size=50):
    return pm.sample_ppc(trc[-1000:], samples=samples, model=mdl,)
def compute_mse(df, ppc, ft_endog):
    return np.sum((ppc['y'].mean(0).mean(0).T - df[[ft_endog]])**2)[0]/df.shape[0]
def classification_leave_one_sub_out(features,targets,results,groups,experiment,dot_dir,window=0,comparison='fast'):
    from sklearn.preprocessing import normalize
    features,targets,groups = shuffle(features,targets,groups)
    cv = LeaveOneGroupOut()
    clfs = make_clfs()
    
    for model,c in clfs.items():
        scores = []
        weights = []
        print('\ntrain-test')
        for fold,(train,test) in enumerate(cv.split(features,targets,groups=groups)):
            clf = make_clfs()[model]
            clf.fit(features[train],targets[train].ravel())
            pred = clf.predict_proba(features[test])[:,-1]
            score = roc_auc_score(targets[test],pred)
            scores.append(score)
            try:
                weights.append(normalize(clf.coef_)[0])
            except:
                weights.append(normalize(clf.feature_importances_.reshape(1, -1))[0])
        results['model'     ].append(model                )
        results['score'     ].append(np.mean(scores)      )
        results['window'    ].append(window               )
        results['correct'   ].append(np.mean(weights,0)[0])
        results['awareness' ].append(np.mean(weights,0)[1])
        results['confidence'].append(np.mean(weights,0)[2])
        # test against to chance level
        print('estimate chance level')
        if comparison == 'slow':
            random_score = []
            for _ in tqdm(range(5)):
                clf = make_clfs()[model]
                cv  = LeaveOneGroupOut()
                X_  = features
                y_  = shuffle(targets)
                random_score_ = cross_val_score(clf,X_,y_.ravel(),groups=groups,
                                               scoring='roc_auc',cv=cv)
                random_score.append(random_score_)
            random_score = np.mean(random_score,axis=0)
            ps = resample_ttest_2sample(np.array(scores),random_score,
                                        n_ps=500,n_permutation=10000)
            results['p_val'     ].append(np.mean(ps))
        elif comparison == 'fast':
            clf = make_clfs()[model]
            cv  = LeaveOneGroupOut()
            X_  = features
            y_  = shuffle(targets)
            random_score = cross_val_score(clf,X_,y_.ravel(),groups=groups,
                                           scoring='roc_auc',cv=cv)
            ps = resample_ttest_2sample(np.array(scores),random_score,
                                        n_ps=500,n_permutation=10000)
            results['p_val'     ].append(np.mean(ps))
        if model == 'DecisionTreeClassifier':
            clf.fit(features,targets)
            out_file = dot_dir+'/'+'{}_window_{}_LOG_tree.dot'.format(experiment,window)
            export_graphviz(decision_tree=clf,
                            out_file=out_file,
                            feature_names=['correct',
                                           'awareness',
                                           'confidence'],
                            class_names=['low {}'.format(experiment),
                                         'high {}'.format(experiment)])
        print('model {},window {},score={:.3f}-{:.4f}'.format(
                model,
                window,
                np.mean(scores),
                np.mean(ps)
                )
            )
    return results
from matplotlib import pyplot as plt
def errplot(x, y, yerr, **kwargs):
    ax      = plt.gca()
    data    = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
 
def omega_squared(aov):
    mse             = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov
def preprocessing(df,
                  participant,
                  n_back                    = 0,
                  names                     = [],
                  independent_variables     = [],
                  dependent_variable        = [],):
    for name in names:
        df.loc[:,name] = df.loc[:,name].values - 1
    features = []
    targets  = []
    for block,df_block in df.groupby('blocks'):
        feature = (df_block[independent_variables].shift(-n_back)
                                                  .dropna()
                                                  .values
                                                  )
        target  = (df_block[dependent_variable].shift(n_back)
                                               .dropna()
                                               .values)
        features.append(feature)
        targets.append(target)
    features    = np.concatenate(features)
    targets     = np.concatenate(targets)
    subs        = np.array([participant] * len(targets))
    return features,targets,subs

def get_features_targets_groups(df,
                                n_back                  = 0,
                                names                   = [],
                                independent_variables   = [],
                                dependent_variable      = []):
    X,y,groups                                          = [],[],[]
    
    for participant,pos_sub in df.groupby('participant'):# for each subject
        features,targets,subs = preprocessing(pos_sub,
                                              participant,
                                              n_back                = n_back,
                                              names                 = names,
                                              independent_variables = independent_variables,
                                              dependent_variable    = dependent_variable)
        X.append(features)
        y.append(targets)
        groups.append(subs)
    X = np.concatenate(X)
    y = np.concatenate(y)
    groups = np.concatenate(groups)
    return X,y,groups
def posthoc_multiple_comparison(df_sub,model_name = '',factor='',n_ps=100,n_permutation=5000):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            ps_mean = [],
            ps_std = [],
            model = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    unique_levels = pd.unique(df_sub[factor])
    pairs = combinations(unique_levels,2)
    for (level1,level2) in pairs:
        ps = resample_ttest_2sample(df_sub[df_sub[factor] == level1]['Values'].values,
                                    df_sub[df_sub[factor] == level2]['Values'].values,
                                    n_ps=n_ps,
                                    n_permutation=n_permutation)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['model'].append(model_name)
        results['level1'].append(level1)
        results['level2'].append(level2)
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['ps_mean'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['ps_mean'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_scipy(df_sub,model_name = '',factor='',):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            pval = [],
            t = [],
            df = [],
            model = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    unique_levels = pd.unique(df_sub[factor])
    pairs = combinations(unique_levels,2)
    for (level1,level2) in pairs:
        t,pval = stats.ttest_rel(df_sub[df_sub[factor] == level1]['Values'].values,
                                 df_sub[df_sub[factor] == level2]['Values'].values,)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)*2-2)
        results['model'].append(model_name)
        results['level1'].append(level1)
        results['level2'].append(level2)
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['pval'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['pval'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_interaction(df_sub,
                                            model_name = '',
                                            unique_levels = [],
                                            n_ps=100,
                                            n_permutation=5000,
                                            selected = True):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            ps_mean = [],
            ps_std = [],
            model = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    if selected:
        pairs = []
        for unique_window in np.unique(np.array(unique_levels)[:,0]):
            selected_window, = np.where(np.array(unique_levels)[:,0] == unique_window)
            selected_window = np.array(unique_levels)[selected_window]
            pairs_ = combinations(selected_window,2)
            for p in pairs_:
                pairs.append(p)
    else:
        pairs = combinations(unique_levels,2)
    try:
        iterator = tqdm(pairs,desc='interaction')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[(df_sub['Window'] == int(level1[0])) & (df_sub['Attributes'] == level1[1])]
        b = df_sub[(df_sub['Window'] == int(level2[0])) & (df_sub['Attributes'] == level2[1])]
        ps = resample_ttest_2sample(a['Values'].values,
                                    b['Values'].values,
                                    n_ps=n_ps,
                                    n_permutation=n_permutation)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['model'].append(model_name)
        results['level1'].append('window {}_{}'.format(level1[0],level1[1]))
        results['level2'].append('window {}_{}'.format(level2[0],level2[1]))
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['ps_mean'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['ps_mean'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_interaction_scipy(df_sub,
                                            model_name = '',
                                            unique_levels = [],
                                            selected = True):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            pval = [],
            t = [],
            df = [],
            model = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    if selected:
        pairs = []
        for unique_window in np.unique(np.array(unique_levels)[:,0]):
            selected_window, = np.where(np.array(unique_levels)[:,0] == unique_window)
            selected_window = np.array(unique_levels)[selected_window]
            pairs_ = combinations(selected_window,2)
            for p in pairs_:
                pairs.append(p)
    else:
        pairs = combinations(unique_levels,2)
    try:
        iterator = tqdm(pairs,desc='interaction')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[(df_sub['Window'] == int(level1[0])) & (df_sub['Attributes'] == level1[1])]
        b = df_sub[(df_sub['Window'] == int(level2[0])) & (df_sub['Attributes'] == level2[1])]
        t,pval = stats.ttest_rel(a['Values'].values,
                                 b['Values'].values,
                                    )
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(a)+len(b) - 2)
        results['model'].append(model_name)
        results['level1'].append('window {}_{}'.format(level1[0],level1[1]))
        results['level2'].append('window {}_{}'.format(level2[0],level2[1]))
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['pval'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['pval'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results

def hierarchical_regression(df_sub,
                            name_for_scale,
                            feature_names,
                            target_name,
                            participant,
                            results,
                            experiment,
                            n_backs = [1,4],
                            n_jobs = -1):
    for n_back in np.arange(n_backs[0],n_backs[1] +1):
        # some preprocessing the data
        X,y,groups = get_features_targets_groups(
                                df_sub.dropna(),
                                n_back                  = n_back,
                                names                   = name_for_scale,
                                independent_variables   = feature_names,
                                dependent_variable      = [target_name,'correctness'])
        # separate the target and the correctness of the currect trials
        y,correctness = y[:,0],y[:,1]
        for model_name,model in make_clfs().items():
            # the cross validation method that will be used repeatedly
            cv = StratifiedShuffleSplit(n_splits=100,
                                        test_size=0.2,
                                        random_state=12345)
            # put the features of the previous N-back trials together
            df_features = pd.DataFrame(X,columns=feature_names)
            # put the correctness of the currect trials to the feature data frame
            df_features['correct_current'] = correctness
            # put the target, aka awareness of the current trials to a data frame
            df_target = pd.DataFrame(y.reshape(-1,1),columns=['predictee'])
            # put everything together
            df_work = pd.concat([df_features,df_target],axis=1)
            
            np.random.seed(12345)
            # model 1: awareness_(current) ~ confidence_(n back)
            model1 = make_clfs()[model_name]
            X_ = df_work['confidence'].values.reshape(-1,1)
            y_ = df_work['predictee'].values
            X_,y_ = shuffle(X_,y_)
            scores1 = cross_val_score(model1,X_,y_,scoring='roc_auc',cv=cv,verbose=1,n_jobs=n_jobs)
            # model 2: awareness_(current) ~ confidence_(n back)+awareness_(n back)
            np.random.seed(12345)
            model2 = make_clfs()[model_name]
            X_ = df_work[['confidence','awareness']].values
            y_ = df_work['predictee'].values
            X_,y_ = shuffle(X_,y_)
            scores2 = cross_val_score(model2,X_,y_,scoring='roc_auc',cv=cv,verbose=1,n_jobs=n_jobs)
            # model 3: awareness_(current) ~ confidence_(n back)+awareness_(n back)+correctness_(current)
            np.random.seed(12345)
            model3 = make_clfs()[model_name]
            X_ = df_work[['confidence','awareness','correct_current']].values
            y_ = df_work['predictee'].values
            X_,y_ = shuffle(X_,y_)
            scores3 = cross_val_score(model3,X_,y_,scoring='roc_auc',cv=cv,verbose=1,n_jobs=n_jobs)
            # model 4: awareness_(current) ~ confidence_(n back)+awareness_(n back)+correctness_(n back)
            np.random.seed(12345)
            model4 = make_clfs()[model_name]
            X_ = df_work[['confidence','awareness','correctness']].values
            y_ = df_work['predictee'].values
            X_,y_ = shuffle(X_,y_)
            scores4 = cross_val_score(model4,X_,y_,scoring='roc_auc',cv=cv,verbose=1,n_jobs=n_jobs)
            # test the improvement of the classification scores
            ps21 = resample_ttest_2sample(scores2,scores1,
                                                n_ps = 100,
                                                n_permutation = 5000,
                                                one_tail = True,)
            ps32 = resample_ttest_2sample(scores3,scores2,
                                                n_ps = 100,
                                                n_permutation = 5000,
                                                one_tail = True,)
            ps42 = resample_ttest_2sample(scores4,scores2,
                                                n_ps = 100,
                                                n_permutation = 5000,
                                                one_tail = True,)
            print(scores1.mean(),scores2.mean(),scores3.mean(),scores4.mean())
            print(ps21.mean(),ps32.mean(),ps42.mean())
            
            results['sub'].append(participant)
            results['model'].append(model_name)
            results['window'].append(n_back)
            results['model1'].append(scores1.mean())
            results['model2'].append(scores2.mean())
            results['model3'].append(scores3.mean())
            results['model4'].append(scores4.mean())
            results['sig21'].append(ps21.mean())
            results['sig32'].append(ps32.mean())
            results['sig42'].append(ps42.mean())
            results['experiment'].append(experiment)
    return results