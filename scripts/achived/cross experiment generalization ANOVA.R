setwd("C:\\Users\\ning\\OneDrive\\python works\\metacognition\\results")
library('reshape2') # need for manipulate the data frame to the format that R could work

df = read.csv('for_spss/auc as by window and model.csv')
# say these variables are discrete
df$sub = factor(df$participant)
df$window = factor(df$window)
df$model = factor(df$model)
df$experiment_test = factor(df$experiment_test)
df$experiment_train = factor(df$experiment_train)
#
for (model in levels(df$model)){
  # select the data frame
  df_sub = df[df$model == model,]
  # anova
  # N-back as within-subject variable
  # experiment as between-subject variable
  result.aov = with(df_sub,
                    aov(score~window*experiment_train+Error(sub/experiment_train)))
  print(model)
  print(summary(result.aov))
  capture.cross = summary(result.aov)
  capture.output(capture.cross,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'cross','3',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'cross','3',model,'sklearn'))
}
require(tidyr)
# all 4 - within and cross experiments
df = read.csv('for_spss/auc as by window and model (4way).csv')
# say these variables are discrete
df$sub = factor(df$participant)
df$window = factor(df$window)
df$model = factor(df$model)
df$experiment_test = factor(df$experiment_test)
df$experiment_train = factor(df$experiment_train)
df$experiment = paste(df$experiment_train,df$experiment_test,sep="_")
df$experiment = factor(df$experiment)

#
for (model in levels(df$model)){
  # select the data frame
  df_sub = df[df$model == model,]
  # anova
  # N-back as within-subject variable
  # experiment as between-subject variable
  result.aov = with(df_sub,
                    aov(score~window*experiment+Error(sub/experiment)))
  print(model)
  print(summary(result.aov))
  capture.cross = summary(result.aov)
  capture.output(capture.cross,
                 file = sprintf("anova results/%s_%s_%s_%s.txt",
                                'cross all','3',model,'sklearn'))
  eta = EtaSq(result.aov,type = 1)
  capture.output(eta,
                 file=sprintf("anova results/Eta sq %s_%s_%s_%s.txt",
                              'cross all','3',model,'sklearn'))
}
