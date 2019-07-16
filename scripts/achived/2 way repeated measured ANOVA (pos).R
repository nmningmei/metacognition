setwd("C:\\Users\\ning\\OneDrive\\python works\\metacognition\\results")
library('reshape2') # need for manipulate the data frame to the format that R could work
library('DescTools')
### model 1: 
# judgement features ,sklearn results
pos = read.csv("Pos_3_1_features.csv")
features = c('correct','awareness','confidence')
id_vars = c('model','score','sub','window')
# manipulate the format of the data frame
pos = melt(pos,id.vars = id_vars,measure.vars = features)
# select the window (trials)
pos_sub = subset(pos,pos$window>0 & pos$window<5)
# say these variables are discrete
pos_sub$sub = factor(pos_sub$sub)
pos_sub$window = factor(pos_sub$window)
pos_sub$model = factor(pos_sub$model)
pos_sub$variable = factor(pos_sub$variable)
# given decision tree or logistic regression
for (model in levels(pos_sub$model)){
  # select the data frame
  df_sub = pos_sub[pos_sub$model == model,]
  # the ANOVA
  result.aov = with(df_sub,
                    aov(value~window*variable+Error(sub/(window*variable))))
  print(model)
  print(summary(result.aov))
  capture.pos.3.sklearn = summary(result.aov)
  capture.output(capture.pos.3.sklearn,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'pos','3',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'pos','3',model,'sklearn'))
}

### model 2:
#RT features, sklearn results
pos = read.csv("Pos_RT_features.csv")
features = c('RT_correct','RT_awareness','RT_confidence')
id_vars = c('model','score','sub','window')
pos = melt(pos,id.vars = id_vars,measure.vars = features)
pos_sub = subset(pos,pos$window>0 & pos$window<5)
pos_sub$sub = factor(pos_sub$sub)
pos_sub$window = factor(pos_sub$window)
pos_sub$model = factor(pos_sub$model)
for (model in levels(pos_sub$model)){
  df_sub = pos_sub[pos_sub$model == model,]
  result.aov = with(df_sub,
                    aov(value~window*variable+Error(sub/(window*variable))))
  print(model)
  print(summary(result.aov))
  capture.pos.RT.sklearn = summary(result.aov)
  capture.output(capture.pos.RT.sklearn,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'pos','RT',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'pos','RT',model,'sklearn'))
  
}

#### model 3:
# 6 features ,sklearn results
pos = read.csv("Pos_6_features.csv")
features = c('correct','awareness','confidence',
             'RT_correct','RT_awareness','RT_confidence')
id_vars = c('model','score','sub','window')
pos = melt(pos,id.vars = id_vars,measure.vars = features)
pos_sub = subset(pos,pos$window>0 & pos$window<5)
pos_sub$sub = factor(pos_sub$sub)
pos_sub$window = factor(pos_sub$window)
pos_sub$model = factor(pos_sub$model)
for (model in levels(pos_sub$model)){
  df_sub = pos_sub[pos_sub$model == model,]
  result.aov = with(df_sub,
                    aov(value~window*variable+Error(sub/(window*variable))))
  print(model)
  print(summary(result.aov))
  capture.pos.6.sklearn = summary(result.aov)
  capture.output(capture.pos.6.sklearn,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'pos','6',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'pos','6',model,'sklearn'))
}

