setwd("C:\\Users\\ning\\OneDrive\\python works\\metacognition\\results")
library('reshape2') # need for manipulate the data frame to the format that R could work
library('DescTools')
require(multcomp)


### model 1: 
# judgement features ,sklearn results
att = read.csv("ATT_3_1_features.csv")
features = c('correct','awareness','confidence')
id_vars = c('model','score','sub','window')
# manipulate the format of the data frame
att = melt(att,id.vars = id_vars,measure.vars = features)
# select the window (trials)
att_sub = subset(att,att$window>0 & att$window<5)
# say these variables are discrete
att_sub$sub = factor(att_sub$sub)
att_sub$window = factor(att_sub$window)
att_sub$model = factor(att_sub$model)
# given decision tree or logistic regression
for (model in levels(att_sub$model)){
  # select the data frame
  df_sub = att_sub[att_sub$model == model,]
  # the ANOVA
  result.aov = with(df_sub,
                    aov(value~window*variable+Error(sub/(window*variable))))
  print(model)
  print(summary(result.aov))
  capture.att.3.sklearn = summary(result.aov)
  capture.output(capture.att.3.sklearn,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'att','3',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'att','3',model,'sklearn'))
}

### model 2:
#RT features, sklearn results
att = read.csv("ATT_RT_features.csv")
features = c('RT_correct','RT_awareness','RT_confidence')
id_vars = c('model','score','sub','window')
att = melt(att,id.vars = id_vars,measure.vars = features)
att_sub = subset(att,att$window>0 & att$window<5)
att_sub$sub = factor(att_sub$sub)
att_sub$window = factor(att_sub$window)
att_sub$model = factor(att_sub$model)
for (model in levels(att_sub$model)){
  df_sub = att_sub[att_sub$model == model,]
  result.aov = with(df_sub,
                    aov(value~window*variable+Error(sub/(window*variable))))
  print(model)
  print(summary(result.aov))
  capture.att.RT.sklearn = summary(result.aov)
  capture.output(capture.att.RT.sklearn,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'att','RT',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'att','RT',model,'sklearn'))
  
}

#### model 3:
# 6 features ,sklearn results
att = read.csv("ATT_6_features.csv")
features = c('correct','awareness','confidence',
             'RT_correct','RT_awareness','RT_confidence')
id_vars = c('model','score','sub','window')
att = melt(att,id.vars = id_vars,measure.vars = features)
att_sub = subset(att,att$window>0 & att$window<5)
att_sub$sub = factor(att_sub$sub)
att_sub$window = factor(att_sub$window)
att_sub$model = factor(att_sub$model)
for (model in levels(att_sub$model)){
  df_sub = att_sub[att_sub$model == model,]
  result.aov = with(df_sub,
                    aov(value~window*variable+Error(sub/(window*variable))))
  print(model)
  print(summary(result.aov))
  capture.att.6.sklearn = summary(result.aov)
  capture.output(capture.att.6.sklearn,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'att','6',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'att','6',model,'sklearn'))
}

