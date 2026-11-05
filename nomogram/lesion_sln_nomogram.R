source("nomogram/common.R")
d <- load_data(c("DL_lesion", "DL_SLN"))
train <- d$train; test <- d$test
train$DL_lesion_logit <- prob_to_logit(train$DL_lesion)
test$DL_lesion_logit <- prob_to_logit(test$DL_lesion)
train$DL_SLN_logit <- prob_to_logit(train$DL_SLN)
test$DL_SLN_logit <- prob_to_logit(test$DL_SLN)
run_nomogram(train,test,label ~ DL_lesion_logit + DL_SLN_logit,"lesion_sln",extra_cols=c("DL_lesion_logit", "DL_SLN_logit"))
