
#--------------------------------------------------------------------------
# Analysis 2 from "The correct Database" paper by Rahnev, Desender, Lee, et al.
# 
# This analysis explores serial dependence in correct RTs, up to lag 7.
# This is done on all datasets including this variable.
#
# To run this analysis, all the files of the correct Database should be placed in a folder called 'correct Database' located in your current WD
#
# Written by Kobe Desender. Last update: Sep 16, 2019.
#--------------------------------------------------------------------------
rm(list=ls());library(here);library(pwr);library(emmeans)
setwd('C:/Users/ning/Documents/python works/metacognition/scripts')
DataSelect = read.csv('../results/linear_mixed/POS.csv') # <-- change the name of the csv file
vif.mer <- function (fit) {
  ## adapted from rms::vif
  v <- vcov(fit)
  nam <- names(fixef(fit))
  ## exclude intercepts
  ns <- sum(1 * (nam == "Intercept" | nam == "(Intercept)"))
  if (ns > 0) {
    v <- v[-(1:ns), -(1:ns), drop = FALSE]
    nam <- nam[-(1:ns)]
  }
  d <- diag(v)^0.5
  v <- diag(solve(v/(d %o% d)))
  names(v) <- nam
  v
}
#Run mixed model
library(lmerTest);library(multcomp);library("ggplot2")
# change the name of the depedent variable below: attention or success
fit <- lmer(success ~ awareness_1 + awareness_2 + awareness_3 + awareness_4 + confidence_1 + confidence_2 + confidence_3 + confidence_4 + correct_1 + correct_2 + correct_3 + correct_4 + (1|sub_name),data=DataSelect)
coef(summary(fit))
glht(fit)
vif.mer(fit) #check VIFs for the predictors
k <-summary(fit) #model output
em <- emmeans(fit,c("awareness_1",'awareness_2',"awareness_3","awareness_4","confidence_1","confidence_2","confidence_3","confidence_4","correct_1","correct_2","correct_3","correct_4"))
contrast(em, adjust = "bonferroni")
# Extract the fixed effect estimates & confints, and plot these
tmp <- as.data.frame(k$coefficients)
tmp$sign <- k$coefficients[,5]
tmp$se <-k$coefficients[,2]
tmp$dof <- k$coefficients[,3]
tmp$t <- k$coefficients[,4]
# don't forget to change the saving name here!!!!!!!!!!!!!!!!!!!
write.csv(tmp,'../results/linear_mixed/POS_fit.csv')







