# Metacognition Experiments

## Exp1: {Probability of Success}(POS, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)
## Exp2: {Attention of the coming trial}(ATT, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)


# Goals:
* - [x] predict POS/ATT with correct, awareness, and confidence ratings
* - [X] cross POS-ATT experiment generalization
* - [x] cross POS-ATT AUC ANOVA, with between subject factor (Exp) and within subject factor (trial window)
* - [x] use features from the previous trials to predict POS/ATT in the next N trials, where 0 < N <= 4
* - [x] interpret the results and infer information processing


# Decodings
1. RandomForest (n_estimators = 500) - to increase biases and avoid overfitting
2. Logistic Regression (C = 1e9) - to reduce regularization so that we can interpret the results


# Windows
1. -- use the features from the previous trial to the target
2. -- features from 2 trials prior to the target
3. -- features from 3 trials prior to the target
4. -- features from 4 trials prior to the target



# Result - 1.1 - Exp 1.logistic regression
## POS, correct, awareness and confidence as features, decoding scores of logistic regression
![pos-3-lr](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%202.jpeg)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals\*, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



Awareness, correctness, and confidence carried lots of information of how participants' POS for the next trial. And these features in the 2-back, 3-back, and 4-back trials carried enough information of how participants' POS for the successive trials for the classifiers to learn and make predictions.

\*Reference:

DiCiccio and Efron, 1996. Bootstrap confidence intervals, Statistical Science, 11(3), 189 - 228



#### P values
1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0057, 4-back: 0.0085


### odd ratios estimated by scitkit-learn logistic regression
![pos-lr-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%203.jpeg)


There is a significant main effect of window, F(3.0,42.0) = 17.4610,p = 0.00000016

There is a significant main effect of attributes, F(2.0,28.0) = 7.5553,p = 0.00237703

A post hoc comparison reveal that:

confidence is significantly different from correct, p = 0.00029997

awareness is significantly different from correct, p = 0.00107689

awareness is not different from confidence, p = 1.00000000

There is a significant interaction between Window and Attributes,F(6.0,84.0) = 7.2377, p = 0.00000301

A post hoc multiple comparision reveal that:

confidence at 1-back is significantly different from correct at 1-back, p = 0.00119988

confidence at 2-back is significantly different from correct at 2-back, p = 0.00211179

awareness at 1-back is significantly different from correct at 1-back, p = 0.02651735

awareness at 2-back is significantly different from correct at 2-back, p = 0.03984802

The reset are not statitically significant, p > 0.0813


# Result - 1.2 - Exp 1.Random Forest
## POS, correct, awareness and confidence as features, decoding scores of random forest
![pos-3-rf](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%202.jpeg)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The RF decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



#### P values
1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0036, 4-back: 0.0076


### feature importance estimated by scitkit-learn random forest
![pos-rf-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%203.jpeg)


There is a significant main effect of window, F(3.0,42.0) = 28.0000,p = 0.00000000

There is a significant main effect of attributes, F(2.0,28.0) = 8.2440,p = 0.00153045

A post hoc comparison reveal that:

awareness is significantly different from correct, p = 0.00029997

confidence is significantly different from correct, p = 0.00029997

awareness is not different from confidence, p = 1.00000000

There is a significant interaction between Window and Attributes,F(6.0,84.0) = 6.0626, p = 0.00002631

A post hoc multiple comparision reveal that:

confidence at 1-back is significantly different from correct at 1-back, p = 0.00119988

awareness at 2-back is significantly different from correct at 2-back, p = 0.00187181

awareness at 1-back is significantly different from correct at 1-back, p = 0.00309569

confidence at 2-back is significantly different from correct at 2-back, p = 0.00652735

The reset are not statitically significant, p > 0.2743


# Result - 2.1 - Exp 2.logistic regression
## ATT, correct, awareness and confidence as features, decoding scores of logistic regression
![att-3-lr](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%205.jpeg)
Decoding decision of engagement with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



#### P values
1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0057, 4-back: 0.0085


### Odd ratio estimated by scitkit-learn logistic regression
![att-rf-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%206.jpeg)


There is a significant main effect of window, F(3.0,45.0) = 5.3268,p = 0.00315804

There is no main effect of attributes, F(2.0,30.0) = 1.9860,p = 0.15488542

A post hoc comparison reveal that:

confidence is not different from correct, p = 0.07132687

confidence is not different from awareness, p = 0.15188381

awareness is not different from correct, p = 1.00000000

There is a no interaction between window and attributes, F(6.0,90.0) = 1.5951, p = 0.15768893

A post hoc multiple comparision reveal that:

confidence at 1-back is significantly different from correct at 1-back, p = 0.03442456

The reset are not statitically significant, p > 1.0000


# Result - 2.2 - Exp 1.Random Forest
## ATT, correct, awareness and confidence as features, decoding scores of random forest
![att-3-rf](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%205.jpeg)
Decoding Decision of Engagement with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The RF decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



#### P values
1-back: 0.0004, 2-back: 0.0005, 3-back: 0.3959, 4-back: 0.0139


### feature importance estimated by scitkit-learn random forest
![att-rf-fw](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%203.jpeg)


There is no main effect of window, F(3.0,45.0) = -240.0000,p = 1.00000000

There is a significant main effect of attributes, F(2.0,30.0) = 5.3776,p = 0.01009451

A post hoc comparison reveal that:

awareness is significantly different from correct, p = 0.00192881

confidence is significantly different from correct, p = 0.00308369

awareness is not different from confidence, p = 0.73119788

There is a a significant interaction between window and attributes, F(6.0,90.0) = 3.6673, p = 0.00264570

A post hoc multiple comparision reveal that:

awareness at 1-back is significantly different from correct at 1-back, p = 0.00331167

confidence at 1-back is significantly different from correct at 1-back, p = 0.00496750

The reset are not statitically significant, p > 0.0729


# Cross Experiment Validation
## Train Classifier in Exp.1 and test the trained classifier in Exp.2



![ex12-lr](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/Figure%208.jpeg)



### p values of POS --> ATT by LogisticRegression
1-back = 0.0108, 2-back = 1.0000, 3-back = 1.0000, 4-back = 1.0000

### p values of ATT --> POS by LogisticRegression
1-back = 0.0004, 2-back = 0.0636, 3-back = 0.0743, 4-back = 1.0000



![ex12-rf](https://github.com/nmningmei/metacognition/blob/master/figures/final_figures/supplymentary/Supplementary%20Fig%208.jpeg)



### p values of POS --> ATT by RandomForestClassifier
1-back = 0.0100, 2-back = 1.0000, 3-back = 1.0000, 4-back = 1.0000

### p values of ATT --> POS by RandomForestClassifier
1-back = 0.0005, 2-back = 0.0580, 3-back = 0.0364, 4-back = 1.0000
