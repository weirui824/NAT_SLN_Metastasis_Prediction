source("nomogram/common.R")
d <- load_data(c("Ki_67", "SLN_diameter", "HER_2", "cortex_thickness", "N_grade_pred"))
train <- d$train; test <- d$test
run_nomogram(train,test,label ~ Ki_67 + SLN_diameter + HER_2 + cortex_thickness + N_grade_pred,"clinical",extra_cols=c())
