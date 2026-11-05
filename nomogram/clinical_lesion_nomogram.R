source("nomogram/common.R")
d <- load_data(c("Ki_67", "SLN_diameter", "HER_2", "cortex_thickness", "N_grade_pred", "DL_lesion"))
train <- d$train; test <- d$test
train$DL_lesion_logit <- prob_to_logit(train$DL_lesion)
test$DL_lesion_logit <- prob_to_logit(test$DL_lesion)
run_nomogram(train,test,label ~ Ki_67 + SLN_diameter + HER_2 + cortex_thickness + N_grade_pred + DL_lesion_logit,"clinical_lesion",extra_cols=c("DL_lesion_logit"))
