# Metacognition Experiments

## Exp1: {Porbability of Success}(POS, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)
## Exp2: {Attention of the coming trial}(ATT, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)

# Goals:
* - [x] predict POS/ATT with correct, awareness, and confidence ratings
* - [x] predict POS/ATT with RT of correct, awareness, and confidence responses
* - [x] predict POS/ATT with all 6 features mentioned
* - [x] use features from the previous trials to predict POS/ATT in the next N trials, where 0 <= N <= 4
* - [x] interpret the results

# Decodings
1. RandomForest (n_estimators = 500) - to increase biases and avoid overfitting
2. Logistic Regression (C = 1e9) - to reduce regularization so that we can interpret the results

# Windows
1. 0 -- use the features from the same trial as the target
2. 1 -- use the features from the previous trial to the target
3. 2 -- features from 2 trials prior to the target
4. 3 -- features from 3 trials prior to the target
5. 4 -- features from 4 trials prior to the target

# Result-1
## POS, correct, awareness and confidence as features, decoding scores of logistic regression and random forest
![pos-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Probability%20of%20Success_3_1_features%20(1%2C2%2C3%2C4).png)
## Significance of the decoding results
![sig-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Proabability%20of%20Success_3_1_features%20(1%2C2%2C3%2C4).png)
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Probability%20of%20Success_3_1_features(1%2C2%2C3%2C4).png)

[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_3_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_3_RandomForestClassifier_sklearn.txt)
1. no main effect of window,    F(2,28) = 0.27,   p = 0.765,        eta = 2.75e-31
2. a main effect of attributes, F(2,28) = 9.49,   p = 0.000714***,  eta = 0.275
3. interaction,                 F(4,56) = 5.485,  p = 0.000848***,  eta = 0.089

[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_3_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_3_LogisticRegression_sklearn.txt)
1. a main effect of window,     F(2,28) = 17.39,  p = 1.23e-5***,   eta = 0.139
2. a main effect of attributes, F(2,28) = 16.32,  p = 2e-5***,      eta = 0.325
3. an interaction,              F(4, 56) = 12.71, p = 1.99e-7***,   eta = 0.079

# Result-2
## POS, RT as features, decoding scores of logistic regression and random forest
![pos-RT-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Probability%20of%20Success_RT_features%20(1%2C2%2C3%2C4).png)
## Significance of the decoding results
![sig-RT-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Proabability%20of%20Success_RT_features%20(1%2C2%2C3%2C4).png)
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-RT-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Probability%20of%20Success_RT_features(1%2C2%2C3%2C4).png)

[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_RT_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_RT_RandomForestClassifier_sklearn.txt)
1. no main effect of window,      F(2,28) = 0.425,  p = 0.658,  eta = 3.52e-30
2. no main effect of attributes,  F(2,28) = 1.305,  p = 0.287,  eta = 0.046
3. no interaction,                F(4,56) = 1.83,   p = 0.136,  eta = 0.052

[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_RT_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_RT_LogisticRegression_sklearn.txt)
1. a main effect of window,       F(2,28) = 9.274,  p = 0.000182***,  eta = 0.066
2. a main effect of attributes,   F(2,28) = 4.847,  p = 0.0156*,      eta = 0.061
3. an interaction,                F(4, 56) = 2.973, p = 0.0269*,      eta = 0.026

# Result-3
## POS, all 6 features, decoding scores of logistic regression and random forest
![pos-6-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Probability%20of%20Success_6_features%20(1%2C2%2C3%2C4).png)
## Significance of the decoding results
![sig-6-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Proabability%20of%20Success_6_features%20(1%2C2%2C3%2C4).png)
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Probability%20of%20Success_6_features(1%2C2%2C3%2C4).png)

[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_6_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_6_RandomForestClassifier_sklearn.txt)
1. no main effect of window,    F(2,28) = 1.667,    p = 0.207,          eta = 8.43e-31
2. main effect of attributes,   F(5,70) = 4.484,    p = 0.00133**,      eta = 0.1.512
3. an interaction,              F(10,140) = 9.078,  p = 0.1.92e-11***,  eta = 0.148

[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/pos_6_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20pos_6_LogisticRegression_sklearn.txt)
1. a main effect of window,     F(2,28) = 11.06,    p = 0.000289***,  eta = 0.033
2. a main effect of attributes, F(5,70) = 16.32,    p = 1.36e-13***,  eta = 0.378
3. an interaction,              F(10,140) = 13.08,  p = 5.91e-16***,  eta = 0.102

# Result-4
## ATT, correct, awareness, and confidence as features, decoding scores of logistic regression and random forest
![att-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Attention_3_1_features(1%2C2%2C3%2C4).png)
## Significance of the decoding results
![sig-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Attention_3_1_features%20(1%2C2%2C3%2C4).png)
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-3-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Attention_3_1_features(1%2C2%2C3%2C4).png)

[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_3_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_3_RandomForestClassifier_sklearn.txt)
1. no main effect of window,    F(2,30) = 1.884, p = 0.17,      eta = 8.16-30
2. a main effect of attributes, F(2,30) = 7.931, p = 0.00172**, eta = 0.156
3. an interaction,              F(4,60) = 3.992, p = 0.00616**, eta = 0.114

[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_3_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_3_LogisticRegression_sklearn.txt)
1. a main effect of window,       F(2,30) = 3.439,  p = 0.0452*,  eta = 0.027
2. no main effect of attributes,  F(2,30) = 1.659,  p = 0.207,    eta = 0.025
3. no an interaction,             F(4, 60) = 1.322, p = 0.272,    eta = 0.028

# Result-5
## ATT, RT as features, decoding scores of logistic regression and ranomd forest
![att-RT-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Attention_RT_features(1%2C2%2C3%2C4).png)
## Significance of the decoding results
![sig-RT-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Attention_RT_features%20(1%2C2%2C3%2C4).png)
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-RT-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Attention_RT_features(1%2C2%2C3%2C4).png)

[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_RT_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_RT_RandomForestClassifier_sklearn.txt)
1. a main effect of window,       F(2,30) = 3.411, p = 0.0463*, eta = 2.57e-29
2. a main effect of attributes,   F(2,30) = 3.808, p = 0.0336*, eta = 0.086
3. no interaction,                F(4,60) = 1.035, p = 0.397,   eta = 0.036

[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_RT_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_RT_LogisticRegression_sklearn.txt)
1. no a main effect of window,    F(2,30) = 2.44,     p = 0.104, eta = 0.0083
2. no main effect of attributes,  F(2,30) = 0.029,    p = 0.972, eta = 0.0004
3. no interaction,                F(4, 60) = 1.918,   p = 0.119, eta = 0.0296

# Result-6
## ATT, all 6 features, decoding scores of logistic regression and random forest
![att-6-123](https://github.com/adowaconan/metacognition/blob/master/figures/Model%20Comparison%20of%20Decoding%20Attention_6_features(1%2C2%2C3%2C4).png)
## Significance of the decoding results
![sig-6-123](https://github.com/adowaconan/metacognition/blob/master/figures/Significance%20test%20of%20Attention_6_features%20(1%2C2%2C3%2C4).png)
### feature importance estimated by random forest and odd ratios estimated by scitkit-learn logistic regression
![weight-6-123](https://github.com/adowaconan/metacognition/blob/master/figures/Weights%20plot%20of%20Attention_6_features(1%2C2%2C3%2C4).png)

[With a 2-way repeated measured ANOVA, feature importances factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_6_RandomForestClassifier_sklearn.txt),[eta squared RF](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_6_RandomForestClassifier_sklearn.txt)
1. no a main effect of window,    F(2,30) = 3.195,  p = 0.0552,       eta = 1.74e-29
2. a main effect of attributes,   F(5,75) = 22.89,  p = 7.09e-14***,  eta = 0.332
3. an interaction,                F(10,150) = 4.61, p = 1.02e-5,      eta = 0.105

[With a 2-way repeated measured ANOVA, odd ratios factored by window and attributes](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/att_6_LogisticRegression_sklearn.txt),[eta squared LG](https://github.com/adowaconan/metacognition/blob/master/results/anova%20results/Eta%20sq%20att_6_LogisticRegression_sklearn.txt)
1. no a main effect of window,    F(2,30) = 1.224,    p = 0.308,    eta = 0.003
2. no main effect of attributes,  F(5,75) = 1.036,    p = 0.403,    eta = 0.035
3. an interaction,                F(10,150) = 2.351,  p = 0.0131*,  eta = 0.049
