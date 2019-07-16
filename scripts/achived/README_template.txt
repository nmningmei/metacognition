# [Metacognition Experiments](https://adowaconan.github.io/metacognition/)



## Exp1: {Porbability of Success}(POS, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)
## Exp2: {Attention of the coming trial}(ATT, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)

# Goals:
* - [x] predict POS/ATT with correct, awareness, and confidence ratings
* - [X] cross POS-ATT experiment generalization
* - [X] cross POS-ATT AUC ANOVA, with between subject factor (Exp) and within subject factor (trial window)
* - [x] use features from the previous trials to predict POS/ATT in the next N trials, where 0 < N <= 4
* - [x] interpret the results

# Decodings
1. RandomForest (n_estimators = 500) - to increase biases and avoid overfitting
2. Logistic Regression (C = 1e9) - to reduce regularization so that we can interpret the results

# Windows
1. -- use the features from the previous trial to the target
2. -- features from 2 trials prior to the target
3. -- features from 3 trials prior to the target
4. -- features from 4 trials prior to the target

# Result-1
## POS, correct, awareness and confidence as features, decoding scores of logistic regression and random forest
![pos-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Probability%20of%20Success_3_1_features%20(1%2C2%2C3%2C4).png)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. Both the random forest classifier and the logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. Error bars represent bootstrapped 95% confidence intervals\*, resampled from the average decoding scores of individual participants by each classifier with 1000 iterations.

Awareness, correctness, and confidence carried lots of information of how participants' POS for the next trial. And these features in the 2-back, 3-back, and 4-back trials carried enough information of how participants' POS for the successive trials for the classifiers to learn and make predictions.

\*Reference:

DiCiccio and Efron, 1996. Bootstrap confidence intervals, Statistical Science, 11(3), 189 - 228
## Significance of the decoding results
![sig-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Proabability%20of%20Success_3_1_features%20(1%2C2%2C3%2C4).png)
#### P values
1. RandomForest 1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0036, 4-back: 0.0076
2. LogisticRegression 1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0056, 4-back: 0.0084
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Probability%20of%20Success_3_1_features(1%2C2%2C3%2C4).png)
#### RandomForest
[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_3_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_3_RandomForestClassifier_sklearn.txt)
1. no main effect of window,    F(3,42) = 0.481,   p = 0.697,        eta = 1.58e-30
2. a main effect of attributes, F(2,28) = 8.244,   p = 0.00513**,  eta = 0.232
3. interaction,                 F(6,84) = 6.063,  p = 0.00000263***,  eta = 0.112

### main effect of feature importance
confidence (p = 0.000003017) and awareness (p = 0.000003059) have different feature importance to correctness
### interaction of widnow and feature importance
1. at 1-back, confidence (p = 0.0001199) and awareness (p = 0.001938) have different feature importance from correctness
2. at 2-back, confidence (p = 0.0052) and awareness (p = 0.000906) have different feature importance from correctness
#### LogisticRegression
[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_3_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_3_LogisticRegression_sklearn.txt)

1. a main effect of window,     F(3,42) = 19.15,  p = 5.58e-8***,   eta = 0.176
2. a main effect of attributes, F(2,28) = 14.43,  p = 0.00000492***,      eta = 0.179
3. an interaction,              F(6,84) = 10.66,  p = 8.83e-9***,   eta = 0.083

### main effect of N-back
1. 1-back is different from 2-back, p = 0.0003674
2. 1-back is different from 3-back, p = 0.0002018
3. 1-back is different from 4-back, p = 0.0001786
4. 2-back is different from 3-back, p = 0.0102729
5. 2-back is different from 4-back, p = 0.0001407

### main effect of odd ratios
both confidence (p = 0.000003449) and awareness (p = 0.0008536) has different odd ratios than the correctness.

### interaction of widnow and odd ratios
1. at 1-back, confidence (p = 0.0001199988) and awareness (p = 0.0265254947) have different odd ratio from correctness
2. at 2-back, confidence (p = 0.0008755112) and awareness (p = 0.0379712203) have different odd ratio from correctness

# Result-2
## ATT, correct, awareness, and confidence as features, decoding scores of logistic regression and random forest
![att-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Attention_3_1_features(1%2C2%2C3%2C4).png)
Decoding Attention with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. Both the random forest classifier and the logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. Error bars represent bootstrapped 95% confidence intervals, resampled from the average decoding scores of individual participants by each classifier.

Awareness, correctness, and confidence carried enough information for the classifier to learn and make predictions about how participants decided to pay attention in the next or the successive 2 trials. The information might not as clear as in the POS experiment.

There is a significant difference in 1-back  (RandomForest:p = 0.0018, logistic:p = 0.0025) and 2-back (RandomForest:p = 0.0113, logistic:p = 0.0081) trials for both models, but not for 3-back (RandomForest:p = 0.2692, logistic:p = 0.2684) or 4-back (RandomForest:p = 1., logistic:p = 1.).

## Significance of the decoding results
![sig-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Attention_3_1_features%20(1%2C2%2C3%2C4).png)
#### P values
1. RandomForest 1-back: 0.0004, 2-back: 0.00045, 3-back: 0.5258, 4-back: 0.0177
2. LogisticRegression 1-back: 0.0004, 2-back: 0.00046, 3-back: 0.3958, 4-back: 0.0138
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Attention_3_1_features(1%2C2%2C3%2C4).png)
#### RandomForest
[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_3_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_3_RandomForestClassifier_sklearn.txt)
1. no main effect of window,    F(3,45) = 0.165, p = 0.919,      eta = 1.62-30
2. a main effect of attributes, F(2,30) = 5.378, p = 0.0101*, eta = 0.103
3. an interaction,              F(6,90) = 3.667, p = 0.00265**, eta = 0.119
### main effect of feature importance
both awareness (p = 0.0016) and confidence (p = 0.0029) has different feature importance than correctness.

### interaction of feature importance
1. at 1-back, confidence (p = 0.0022504575) and awareness (p = 0.0037631624) have different feature importance than correctness.
#### LogisticRegression
[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_3_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_3_LogisticRegression_sklearn.txt)
1. a main effect of window,       F(3,45) = 3.224,  p = 0.0313*,  eta = 0.043
2. no main effect of attributes,  F(2,30) = 2.052,  p = 0.146,    eta = 0.023
3. no an interaction,             F(4,60) = 0.97, p = 0.45,    eta = 0.023
### main effect of windows
1. 1-back is different from 3-back, p = 0.03611
2. 1-back is different from 4-back, p = 0.02435

# Result-3
## Cross-experiment generalization, invidual subject level, correct, awareness, and confidence as features, decoding scores of logistic regression and random forest
![cross-123](https://github.com/adowaconan/metacognition/blob/master/figures/within%20cross%20experiment%20CV%20results_3_1_features.png)
To train the classifiers on one of the experiments, and test the classifiers on the each of the subjects of the other experiment.

There is no difference in decoding enhancement despite the direction of cross-experiment generalization (corrected p vales respect to the N-back trials showed below)

### p values
#### RandomForest

1-back:1., 2-back:1., 3-back:0.5327, 4-back:1.

#### LogisticRegression

1-back:1., 2-back:1., 3-back:0.6659, 4-back:1.

## Significance of the decoding results
![sig-cross-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20within%20cross%20experiments_3_1_features.png)
#### P values
- trained on POS:
1. RandomForest 1-back: 0.020292, 2-back: 1.0000, 3-back: 1.0000, 4-back: 1.0000
2. LogisticRegression 1-back: 0.0.0215, 2-back: 1.0000, 3-back: 1.0000, 4-back: 1.0000
- trained on ATT:
1. RandomForest 1-back: 0.00085, 2-back: 0.1264, 3-back: 0.1468, 4-back: 1.0000
2. LogisticRegression 1-back: 0.0009, 2-back: 0.0735, 3-back: 0.1156, 4-back: 1.0000

# Result-4 (supplementary) - logistic regression only
## variance explained by features (all included and single)
![variance_explained](https://github.com/adowaconan/metacognition/blob/master/figures/variance%20explained%20experiment%20comparison%20logistic%20regression.png)
## weights of features (all included and single)
![weights_logistic](https://github.com/adowaconan/metacognition/blob/master/figures/weights%20experiment%20comparison%20logistic%20regression.png)
Whether the logistic regression model includes all the regressor or only one of them provides a different amount of variance explained, but the differences are not statistically significant (Bonferroni corrected). However, we observed that the variance explained between the full regressors model and the single regressor model and the weights between these two models are consistent with each other. When a regressor (“correct”) explains little of the target in the model that contains only this regressor, it has a small weight in both the single regressor model and the full regressors model. By comparing variance explained by all predictors with variance explained by one of the predictors, we argue that the model that contains all predictors is better than any one of the models that contain one of the predictors for 2 reasons. First, including all the predictors, the model utilizes covariances among the predictors and thus provides a multivariate explanation of how these predictors predict the target. Second, including all the predictors makes the logistic regression model comparable to the random forest model in terms of how each predictor contribute to the prediction, except that the logistic regression provides the perspective of odd ratio (how likely a predictor affects the target numerically given the other predictors are constants), while the random forest model provides the perspective of feature importance (how much information gained if a predictor is included regardless the other predictors).
