rm(list=ls());library(here);library(pwr)
setwd('C:/Users/ning/Documents/python works/metacognition/scripts')
DataSelect <- read.csv('../results/linear_mixed/for_ezANOVA_pos.csv') # <-- change the name of the csv file

library(nlme)
res <- lme(success ~ awareness*confidence*correct*time,
           random = ~1|sub_name*time,
           data = DataSelect,
           method = 'REML')
library(car)
Anova(res)
