# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:03:46 2019

@author: ning
"""

from glob import glob
import os
working_dir = ''
import pandas as pd
import utils
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
#from statannot import add_stat_annotation
sns.set_style('whitegrid')
sns.set_context('poster')
from utils import post_processing
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
import statsmodels.api as sm
figure_dir = '../figures/final_figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
title_map = {'RandomForestClassifier':'Supplementary Fig','LogisticRegression':'Figure'}
model_order = ['LogisticRegression','RandomForestClassifier']


big_title = "# Metacognition Experiments"
with open('README.md','w') as f:
    f.write(big_title)
    f.close()

experiment_description = """## Exp1: {Probability of Success}(POS, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)
## Exp2: {Attention of the coming trial}(ATT, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(experiment_description)
    f.close()

goals = """# Goals:
* - [x] predict POS/ATT with correct, awareness, and confidence ratings
* - [X] cross POS-ATT experiment generalization
* - [x] cross POS-ATT AUC ANOVA, with between subject factor (Exp) and within subject factor (trial window)
* - [x] use features from the previous trials to predict POS/ATT in the next N trials, where 0 < N <= 4
* - [x] interpret the results and infer information processing
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(goals)
    f.close()
    
decoding = """# Decodings
1. RandomForest (n_estimators = 500) - to increase biases and avoid overfitting
2. Logistic Regression (C = 1e9) - to reduce regularization so that we can interpret the results
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(decoding)
    f.close()
    
windows = """# Windows
1. -- use the features from the previous trial to the target
2. -- features from 2 trials prior to the target
3. -- features from 3 trials prior to the target
4. -- features from 4 trials prior to the target
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(windows)
    f.close()
    
pos = pd.read_csv('../results/Pos_3_1_features.csv')
att = pd.read_csv('../results/ATT_3_1_features.csv')
pos_ttest = pd.read_csv('../results/Pos_ttest_3_1_features.csv')
att_ttest = pd.read_csv('../results/ATT_ttest_3_1_features.csv')
# results 1.1 - logistic regression decoding results
result11="""
# Result - 1.1 - Exp 1.logistic regression
## POS, correct, awareness and confidence as features, decoding scores of logistic regression
![pos-3-lr](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%202.jpeg)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals\*, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(result11)
    f.close()
    
reference="""
Awareness, correctness, and confidence carried lots of information of how participants' POS for the next trial. And these features in the 2-back, 3-back, and 4-back trials carried enough information of how participants' POS for the successive trials for the classifiers to learn and make predictions.

\*Reference:

DiCiccio and Efron, 1996. Bootstrap confidence intervals, Statistical Science, 11(3), 189 - 228
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(reference)
    f.close()

df  = pos.copy()
df_ttest = pos_ttest.copy()
c   = df.groupby(['sub','model','window']).mean().reset_index()
c   = c[(c['window'] > 0) & (c['window'] < 5)]
pvalues_temp = """
#### P values
1-back: {:.4f}, 2-back: {:.4f}, 3-back: {:.4f}, 4-back: {:.4f}"""
pvalues1 = {}
for model_name in model_order:
    df_ttest_sub = df_ttest[df_ttest['model'] == model_name]
    a,b,c,d = df_ttest_sub['ps_corrected'].values
    pvalues1[model_name] = pvalues_temp.format(a,b,c,d)
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(pvalues1['LogisticRegression'])
    f.close()

features = """
### odd ratios estimated by scitkit-learn logistic regression
![pos-lr-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%203.jpeg)
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(features)
    f.close()

posthoc1 = {}
id_vars         = ['model',
                   'score',
                   'sub',
                   'window',]
value_vars      =[
                  'correct',
                  'awareness',
                  'confidence',
                  ]
df_post                       = post_processing(df[(df['window'] >0) & (df['window'] < 5)],
                                                id_vars,value_vars)
c                             = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
multicomparisonMessages = {}
for model_name,df_sub in c.groupby(['Models']):
    text = ""
#    df_sub['Window'] = pd.Categorical(df_sub['Window'])
    aovrm = AnovaRM(df_sub,'Values','sub',within=['Window','Attributes'])
    res_anova = aovrm.fit()
#    model = ols('Values ~ C(Window)*C(Attributes)', df_sub).fit()
#    res_anova = sm.stats.anova_lm(model, typ= 2)
#    print(res_anova)
    # main effect of window
    text_temp = "There is {} main effect of window, F({:1},{:1}) = {:.4f},p = {:.8f}"
    sig = res_anova.anova_table['Pr > F']['Window']
    if res_anova.anova_table['F Value']['Window'] == np.inf: sig_text = 'no'
    elif sig < 0.05: sig_text = "a significant" 
    else: sig_text = "no"
    text_temp = text_temp.format(
                     sig_text,
                     res_anova.anova_table['Num DF']['Window'],
                     res_anova.anova_table['Den DF']['Window'],
                     res_anova.anova_table['F Value']['Window'],
                     res_anova.anova_table['Pr > F']['Window'],)
    text += text_temp
    main1 = utils.posthoc_multiple_comparison(
            df_sub,
            depvar = 'Values',
            factor = 'Window',
            n_ps = 100,
            n_permutation = int(1e4))
    
    # main effect of features
    text_temp = "\n\nThere is {} main effect of attributes, F({:1},{:1}) = {:.4f},p = {:.8f}"
    sig = res_anova.anova_table['Pr > F']['Attributes']
    if sig < 0.05: sig_text = "a significant" 
    else: sig_text = "no"
    text_temp = text_temp.format(
                     sig_text,
                     res_anova.anova_table['Num DF']['Attributes'],
                     res_anova.anova_table['Den DF']['Attributes'],
                     res_anova.anova_table['F Value']['Attributes'],
                     res_anova.anova_table['Pr > F']['Attributes'],
                     )
    text += text_temp
    main2 = utils.posthoc_multiple_comparison(
            df_sub,
            depvar = 'Values',
            factor = 'Attributes',
            n_ps = 100,
            n_permutation = int(1e4))
    
    text_temp = "\n\nA post hoc comparison reveal that:"
    text += text_temp
    for ii, row in main2.iterrows():
        text_temp = "\n\n{} is {} different from {}, p = {:.8f}"
        if row['p_corrected'] < 0.05:
            sig_text = "significantly"
        else:
            sig_text = "not"
        text_temp = text_temp.format(
                         row['level1'],
                         sig_text,
                         row['level2'],
                         row['p_corrected'])
        text += text_temp
    
    # interaction 
    text_temp = "\n\nThere is {} interaction between Window and Attributes,F({:1},{:1}) = {:.4f}, p = {:.8f}"
    sig = res_anova.anova_table['Pr > F']['Window:Attributes']
    if sig < 0.05: sig_text = "a significant" 
    else: sig_text = "no"
    text_temp = text_temp.format(
                     sig_text,
                     res_anova.anova_table['Num DF']['Window:Attributes'],
                     res_anova.anova_table['Den DF']['Window:Attributes'],
                     res_anova.anova_table['F Value']['Window:Attributes'],
                     res_anova.anova_table['Pr > F']['Window:Attributes'],
            )
    text += text_temp
    
    text_temp = "\n\nA post hoc multiple comparision reveal that:"
    text += text_temp
    interaction_temp = utils.posthoc_multiple_comparison_interaction(
            df_sub,
            depvar = 'Values',
            unique_levels = ['Window','Attributes'],
            n_ps = 100,
            n_permutation = int(1e4),
            selected = 0)
    interaction_temp = utils.strip_interaction_names(interaction_temp)
    interaction_temp = interaction_temp.sort_values(['p_corrected'])
    for ii, row in interaction_temp.iterrows():
        if row['p_corrected'] < 0.05:
            text_temp = "\n\n{} at {:1}-back is significantly different from {} at {:1}-back, p = {:.8f}"
            text_temp = text_temp.format(
                             row['attribute1'],
                             row['window'],
                             row['attribute2'],
                             row['window'],
                             row['p_corrected'])
            text += text_temp
        else:
            break
    text += "\n\nThe reset are not statitically significant, p > {:.4f}".format(
            row['p_corrected'])
    multicomparisonMessages[model_name] = text
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(multicomparisonMessages['LogisticRegression'])
    f.close()


# results 1.2 - RF decoding results
result22="""
# Result - 1.2 - Exp 1.Random Forest
## POS, correct, awareness and confidence as features, decoding scores of random forest
![pos-3-rf](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%202.jpeg)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The RF decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(result22)
    f.close()

with open('README.md','a') as f:
    f.write('\n\n')
    f.write(pvalues1['RandomForestClassifier'])
    f.close()

features = """
### feature importance estimated by scitkit-learn random forest
![pos-rf-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%203.jpeg)
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(features)
    f.close()
    
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(multicomparisonMessages['RandomForestClassifier'])
    f.close()



# results 2.1 - logistic regression decoding results
result21="""
# Result - 2.1 - Exp 2.logistic regression
## ATT, correct, awareness and confidence as features, decoding scores of logistic regression
![att-3-lr](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%205.jpeg)
Decoding decision of engagement with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(result21)
    f.close()

df  = att.copy()
df_ttest = att_ttest.copy()
c   = df.groupby(['sub','model','window']).mean().reset_index()
c   = c[(c['window'] > 0) & (c['window'] < 5)]
pvalues_temp = """
#### P values
1-back: {:.4f}, 2-back: {:.4f}, 3-back: {:.4f}, 4-back: {:.4f}"""
pvalues2 = {}
for model_name in model_order:
    df_ttest_sub = df_ttest[df_ttest['model'] == model_name]
    a,b,c,d = df_ttest_sub['ps_corrected'].values
    pvalues2[model_name] = pvalues_temp.format(a,b,c,d)
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(pvalues1['LogisticRegression'])
    f.close()

features = """
### Odd ratio estimated by scitkit-learn logistic regression
![att-rf-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%206.jpeg)
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(features)
    f.close()

posthoc2 = {}
id_vars         = ['model',
                   'score',
                   'sub',
                   'window',]
value_vars      =[
                  'correct',
                  'awareness',
                  'confidence',
                  ]
df_post                       = post_processing(df[(df['window'] >0) & (df['window'] < 5)],
                                                id_vars,value_vars)
c                             = df_post.groupby(['sub','Models','Window','Attributes']).mean().reset_index()
multicomparisonMessages = {}
for model_name,df_sub in c.groupby(['Models']):
    text = ""
#    df_sub['Window'] = pd.Categorical(df_sub['Window'])
    aovrm = AnovaRM(df_sub,'Values','sub',within=['Window','Attributes'])
    res_anova = aovrm.fit()
#    model = ols('Values ~ C(Window)*C(Attributes)', df_sub).fit()
#    res_anova = sm.stats.anova_lm(model, typ= 2)
#    print(res_anova)
    # main effect of window
    text_temp = "There is {} main effect of window, F({:1},{:1}) = {:.4f},p = {:.8f}"
    sig = res_anova.anova_table['Pr > F']['Window']
    if res_anova.anova_table['F Value']['Window'] == np.inf: sig_text = 'no'
    elif sig < 0.05: sig_text = "a significant" 
    else: sig_text = "no"
    text_temp = text_temp.format(
                     sig_text,
                     res_anova.anova_table['Num DF']['Window'],
                     res_anova.anova_table['Den DF']['Window'],
                     res_anova.anova_table['F Value']['Window'],
                     res_anova.anova_table['Pr > F']['Window'],
                     )
    text += text_temp
    main1 = utils.posthoc_multiple_comparison(
            df_sub,
            depvar = 'Values',
            factor = 'Window',
            n_ps = 100,
            n_permutation = int(1e4))
    
    # main effect of features
    text_temp = "\n\nThere is {} main effect of attributes, F({:1},{:1}) = {:.4f},p = {:.8f}"
    sig = res_anova.anova_table['Pr > F']['Attributes']
    if sig < 0.05: sig_text = "a significant" 
    else: sig_text = "no"
    text_temp = text_temp.format(
                     sig_text,
                     res_anova.anova_table['Num DF']['Attributes'],
                     res_anova.anova_table['Den DF']['Attributes'],
                     res_anova.anova_table['F Value']['Attributes'],
                     res_anova.anova_table['Pr > F']['Attributes'],
                     )
    text += text_temp
    
    main2 = utils.posthoc_multiple_comparison(
            df_sub,
            depvar = 'Values',
            factor = 'Attributes',
            n_ps = 100,
            n_permutation = int(1e4))
    
    text_temp = "\n\nA post hoc comparison reveal that:"
    text += text_temp
    for ii, row in main2.iterrows():
        text_temp = "\n\n{} is {} different from {}, p = {:.8f}"
        if row['p_corrected'] < 0.05:
            sig_text = "significantly"
        else:
            sig_text = "not"
        text_temp = text_temp.format(
                         row['level1'],
                         sig_text,
                         row['level2'],
                         row['p_corrected'])
        text += text_temp
    # interaction 
    text_temp = "\n\nThere is a {} interaction between window and attributes, F({:1},{:1}) = {:.4f}, p = {:.8f}"
    sig = res_anova.anova_table['Pr > F']['Window:Attributes']
    if sig < 0.05: sig_text = "a significant" 
    else: sig_text = "no"
    text_temp = text_temp.format(
                        sig_text,
                        res_anova.anova_table['Num DF']['Window:Attributes'],
                        res_anova.anova_table['Den DF']['Window:Attributes'],
                        res_anova.anova_table['F Value']['Window:Attributes'],
                        res_anova.anova_table['Pr > F']['Window:Attributes'],
                        )
    text += text_temp
    text_temp = "\n\nA post hoc multiple comparision reveal that:"
    text += text_temp
    interaction_temp = utils.posthoc_multiple_comparison_interaction(
            df_sub,
            depvar = 'Values',
            unique_levels = ['Window','Attributes'],
            n_ps = 100,
            n_permutation = int(1e4),
            selected = 0)
    interaction_temp = utils.strip_interaction_names(interaction_temp)
    interaction_temp = interaction_temp.sort_values(['p_corrected'])
    for ii, row in interaction_temp.iterrows():
        if row['p_corrected'] < 0.05:
            text_temp = "\n\n{} at {:1}-back is significantly different from {} at {:1}-back, p = {:.8f}"
            text_temp = text_temp.format(
                             row['attribute1'],
                             row['window'],
                             row['attribute2'],
                             row['window'],
                             row['p_corrected'])
            text += text_temp
        else:
            break
    text += "\n\nThe reset are not statitically significant, p > {:.4f}".format(
            row['p_corrected'])
    multicomparisonMessages[model_name] = text


with open('README.md','a') as f:
    f.write('\n\n')
    f.write(multicomparisonMessages['LogisticRegression'])
    f.close()

# results 2.2 - RF decoding results
result22="""
# Result - 2.2 - Exp 1.Random Forest
## ATT, correct, awareness and confidence as features, decoding scores of random forest
![att-3-rf](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%205.jpeg)
Decoding Decision of Engagement with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The RF decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(result22)
    f.close()

with open('README.md','a') as f:
    f.write('\n\n')
    f.write(pvalues2['RandomForestClassifier'])
    f.close()
    
features = """
### feature importance estimated by scitkit-learn random forest
![att-rf-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%203.jpeg)
"""
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(features)
    f.close()
with open('README.md','a') as f:
    f.write('\n\n')
    f.write(multicomparisonMessages['RandomForestClassifier'])
    f.close()

# cross experiment validation

subtitle = """
# Cross Experiment Validation
## Train Classifier in Exp.1 and test the trained classifier in Exp.2
"""

with open("README.md",'a') as f:
    f.write('\n\n')
    f.write(subtitle)
    f.close()

#########   cross experiment validation ######## direct copy from my other scripts
working_dir = '../results/cross_experiment_generalization'
df = pd.read_csv(os.path.join(working_dir,'cross experiment generalization (individual level).csv'))
df_plot = df.groupby(['window',
                      'model',
                      'participant',
                      'experiment_test',
                      'experiment_train']).mean().reset_index()
df_plot = df_plot[(df_plot['window'] > 0) & (df_plot['window'] < 5)]

cols = ['model', 'pval', 'score_mean', 'score_std', 'train', 'window',]
df_post = {name:[] for name in cols}
for (train,model_name,window),df_sub in df_plot.groupby(['experiment_train','model','window']):
    df_sub
    ps = utils.resample_ttest(df_sub['score'].values,
                              0.5,
                              n_ps = 100,
                              n_permutation = int(1e4),
                              one_tail = True)
    df_post['model'].append(model_name)
    df_post['window'].append(window)
    df_post['train'].append(train)
    df_post['score_mean'].append(df_sub['score'].values.mean())
    df_post['score_std'].append(df_sub['score'].values.std())
    df_post['pval'].append(ps.mean())
df_post = pd.DataFrame(df_post)
temp = []
for (train,model_name), df_sub in df_post.groupby(['train','model']):
    df_sub = df_sub.sort_values(['pval'])
    ps = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = ps)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
df_post = pd.concat(temp)

df_post['star'] = df_post['p_corrected'].apply(utils.stars)
df_post = df_post.sort_values(['window','train','model'])
df_post = df_post[df_post['window'] > 0]

text_dict = {}
experiments = ["POS","ATT"]
for model_name in model_order:
    df_post_sub = df_post[df_post['model'] == model_name]
    df_post_sub = df_post_sub.sort_values(['window'])
    text_dict['{}'.format(model_name)] = ""
    for ii,train in enumerate(experiments):
        df_sub = df_post_sub[df_post_sub['train'] == train]
        a,b,c,d = df_sub['p_corrected'].values
        text_temp = """
### p values of {} --> {} by {}
1-back = {:.4f}, 2-back = {:.4f}, 3-back = {:.4f}, 4-back = {:.4f}
""".format(train,experiments[ii - 1],
            model_name,
            a,b,c,d)
        text_dict['{}'.format(model_name)] += text_temp
         
# results 3.1 - Exp.1 <--> Exp.2, logistic regression
text = """
![ex12-lr](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%208.jpeg)
"""
with open("README.md",'a') as f:
    f.write('\n\n')
    f.write(text)
    f.close()

with open("README.md",'a') as f:
    f.write('\n\n')
    f.write(text_dict['{}'.format('LogisticRegression')])
# results 3.2 - Exp.1 <--> Exp.2, Random forest
text = """
![ex12-rf](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%208.jpeg)
"""
with open("README.md",'a') as f:
    f.write('\n\n')
    f.write(text)
    f.close()

with open("README.md",'a') as f:
    f.write('\n\n')
    f.write(text_dict['{}'.format('RandomForestClassifier')])

subtitle = """
# Linear Mixed Model
## As requested
"""

with open("README.md",'a') as f:
    f.write('\n\n')
    f.write(subtitle)
    f.close()

#########   GLM ######## direct copy from my other scripts
df_mixed = pd.read_csv('../results/mixed_linear_model.csv')
df_mixed_pair = pd.read_csv('../results/mixed_linear_model_pairwise.csv')

experiments = ["POS","ATT"]
for exp in experiments:
    df_mixed_sub = df_mixed[df_mixed['experiment'] == exp]
    df_mixed_pair_sub = df_mixed_pair[df_mixed_pair['experiment'] == exp]
    text_dict = f"""
### for {exp}:
![{exp.lower()}_mixed](https://github.com/nmningmei/metacognition/blob/master/figures/linear_mixed/{exp}.jpeg)
"""
    text_dict += '''
from the output of the R lmer package:
    '''
    for ii,row in df_mixed_sub.iterrows():
        row
        if row['star'] != 'n.s.':
            text_temp = f"""
coefficient of {row['Attributes']} at time {row['time']} = {row['Estimate']:.5f}, t({row['dof']:.2f}) = {row['t']:.2f},p = {row['ps_corrected']:1.3e}
"""
            text_dict += text_temp
    text_dict += """For pairwise comparison at each time:
    """
    for ii,row in df_mixed_pair_sub.iterrows():
        if row['star'] != 'n.s.':
            text_temp = f"""
There exists a significant difference between {row['level1']} and {row['level2']}, t = {row['t']:.3f}, p = {row['p_corrected']:1.3e}
"""
            text_dict += text_temp
    with open("README.md",'a') as f:
        f.write('\n\n')
        f.write(text_dict)
        f.close()








from shutil import copyfile
copyfile('README.md','../README.md')


























