source("nomogram/common.R")
d <- load_data(c("Ki_67", "SLN_diameter", "HER_2", "cortex_thickness", "N_grade_pred", "DL_SLN"))
train <- d$train; test <- d$test
train$DL_SLN_logit <- prob_to_logit(train$DL_SLN)
test$DL_SLN_logit <- prob_to_logit(test$DL_SLN)
run_nomogram(train,test,label ~ Ki_67 + SLN_diameter + HER_2 + cortex_thickness + N_grade_pred + DL_SLN_logit,"clinical_sln",extra_cols=c("DL_SLN_logit"))
