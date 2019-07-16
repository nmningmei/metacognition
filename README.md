# Metacognition Experiments

## Exp1: {Porbability of Success}(POS, low vs. high) --> {gabor patch} --> response (correct vs. incorrect) --> {awareness}(unseen vs. seen) --> {confidence}(low vs. high)
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
![pos-3-lr](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/Figure%202.png)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals\*, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



Awareness, correctness, and confidence carried lots of information of how participants' POS for the next trial. And these features in the 2-back, 3-back, and 4-back trials carried enough information of how participants' POS for the successive trials for the classifiers to learn and make predictions.

\*Reference:

DiCiccio and Efron, 1996. Bootstrap confidence intervals, Statistical Science, 11(3), 189 - 228



#### P values
1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0057, 4-back: 0.0085


### odd ratios estimated by scitkit-learn logistic regression
![pos-lr-fw](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/Figure%203.png)


There is a significant main effect of window, F(3.0,42.0) = 17.4610,p = 0.00000016

There is a significant main effect of attributes, F(2.0,28.0) = 7.5553,p = 0.00237703

A post hoc comparison reveal that:

confidence is significantly different from correct, p = 0.00030897

awareness is significantly different from correct, p = 0.00111589

awareness is not different from confidence, p = 1.00000000

There is a significant interaction between Window and Attributes,F(6.0,84.0) = 7.2377, p = 0.00000301

A post hoc multiple comparision reveal that:

confidence at 1-back is significantly different from correct at 1-back, p = 0.00119988

confidence at 2-back is significantly different from correct at 2-back, p = 0.00212379

awareness at 1-back is significantly different from correct at 1-back, p = 0.02766923

awareness at 2-back is significantly different from correct at 2-back, p = 0.03741226

The reset are not statitically significant, p > 0.0828


# Result - 1.2 - Exp 1.Random Forest
## POS, correct, awareness and confidence as features, decoding scores of random forest
![pos-3-rf](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/supplymentary/Supplementary%20Fig%202.png)
Decoding Probability of Success with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The RF decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



#### P values
1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0036, 4-back: 0.0076


### feature importance estimated by scitkit-learn random forest
![pos-rf-fw](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/supplymentary/Supplementary%20Fig%203.png)


There is a significant main effect of window, F(3.0,42.0) = 28.0000,p = 0.00000000

There is a significant main effect of attributes, F(2.0,28.0) = 8.2440,p = 0.00153045

A post hoc comparison reveal that:

awareness is significantly different from correct, p = 0.00029997

confidence is significantly different from correct, p = 0.00029997

awareness is not different from confidence, p = 1.00000000

There is a significant interaction between Window and Attributes,F(6.0,84.0) = 6.0626, p = 0.00002631

A post hoc multiple comparision reveal that:

confidence at 1-back is significantly different from correct at 1-back, p = 0.00119988

awareness at 2-back is significantly different from correct at 2-back, p = 0.00211179

awareness at 1-back is significantly different from correct at 1-back, p = 0.00303570

confidence at 2-back is significantly different from correct at 2-back, p = 0.00646735

The reset are not statitically significant, p > 0.2738


# Result - 2.1 - Exp 2.logistic regression
## ATT, correct, awareness and confidence as features, decoding scores of logistic regression
![att-3-lr](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/Figure%205.png)
Decoding decision of engagement with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The logistic regression decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



#### P values
1-back: 0.0004, 2-back: 0.0004, 3-back: 0.0057, 4-back: 0.0085


### Odd ratio estimated by scitkit-learn logistic regression
![att-rf-fw](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/Figure%206.png)


There is a significant main effect of window, F(3.0,45.0) = 5.3268,p = 0.00315804

There is no main effect of attributes, F(2.0,30.0) = 1.9860,p = 0.15488542

A post hoc comparison reveal that:

confidence is not different from correct, p = 0.07162084

confidence is not different from awareness, p = 0.15062094

awareness is not different from correct, p = 1.00000000

There is a no interaction between window and attributes, F(6.0,90.0) = 1.5951, p = 0.15768893

A post hoc multiple comparision reveal that:

confidence at 1-back is significantly different from correct at 1-back, p = 0.03536046

The reset are not statitically significant, p > 1.0000


# Result - 2.2 - Exp 1.Random Forest
## ATT, correct, awareness and confidence as features, decoding scores of random forest
![att-3-rf](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/supplymentary/Supplementary%20Fig%205.png)
Decoding Decision of Engagement with awareness, correctness, and confidence as features as a function of N-back trials, and factored by the classifiers. 
The RF decode the POS above chance at the group level (see p values below). Black dotted line is the theoretical chance level, 0.5. 
Error bars represent bootstrapped 95% confidence intervals, resampled from the distribution of decoding scores of individual participants by each classifier with 10000 iterations.



#### P values
1-back: 0.0004, 2-back: 0.0005, 3-back: 0.3959, 4-back: 0.0139


### feature importance estimated by scitkit-learn random forest
![att-rf-fw](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/supplymentary/Supplementary%20Fig%203.png)


There is no main effect of window, F(3.0,45.0) = -240.0000,p = 1.00000000

There is a significant main effect of attributes, F(2.0,30.0) = 5.3776,p = 0.01009451

A post hoc comparison reveal that:

awareness is significantly different from correct, p = 0.00202780

confidence is significantly different from correct, p = 0.00302670

awareness is not different from confidence, p = 0.72990801

There is a a significant interaction between window and attributes, F(6.0,90.0) = 3.6673, p = 0.00264570

A post hoc multiple comparision reveal that:

awareness at 1-back is significantly different from correct at 1-back, p = 0.00332367

confidence at 1-back is significantly different from correct at 1-back, p = 0.00477552

The reset are not statitically significant, p > 0.0735


# Cross Experiment Validation
## Train Classifier in Exp.1 and test the trained classifier in Exp.2



![ex12-lr](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/Figure%208.png)



### p values of POS --> ATT by LogisticRegression
1-back = 0.0110, 2-back = 1.0000, 3-back = 1.0000, 4-back = 1.0000

### p values of ATT --> POS by LogisticRegression
1-back = 0.0004, 2-back = 0.0637, 3-back = 0.0733, 4-back = 1.0000



![ex12-rf](https://github.com/adowaconan/metacognition/tree/master/figures/final_figures/supplymentary/Supplementary%20Fig%208.png)



### p values of POS --> ATT by RandomForestClassifier
1-back = 0.0103, 2-back = 1.0000, 3-back = 1.0000, 4-back = 1.0000

### p values of ATT --> POS by RandomForestClassifier
1-back = 0.0005, 2-back = 0.0577, 3-back = 0.0370, 4-back = 1.0000
