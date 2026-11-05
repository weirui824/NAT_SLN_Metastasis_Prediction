# Step 1: glm(label ~ lesion_logit + SLN_logit) on training data.
# Step 2: use its linear predictor DL_signature with five clinical factors in lrm.
source("nomogram/common.R")
d <- load_data(c(clinical_vars, "DL_lesion", "DL_SLN"))
train <- d$train; test <- d$test
train$DL_lesion_logit <- prob_to_logit(train$DL_lesion)
train$DL_SLN_logit <- prob_to_logit(train$DL_SLN)
test$DL_lesion_logit <- prob_to_logit(test$DL_lesion)
test$DL_SLN_logit <- prob_to_logit(test$DL_SLN)
fit_dl <- glm(label ~ DL_lesion_logit + DL_SLN_logit, data=train, family=binomial())
train$DL_signature <- as.numeric(predict(fit_dl,newdata=train,type="link"))
test$DL_signature <- as.numeric(predict(fit_dl,newdata=test,type="link"))
if (!all(is.finite(train$DL_signature)) || !all(is.finite(test$DL_signature))) stop("Invalid DL_signature")
dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
saveRDS(fit_dl,file.path(output_dir,"comprehensive_imaging_signature_glm.rds"))
run_nomogram(train,test,label ~ Ki_67 + SLN_diameter + HER_2 + cortex_thickness + N_grade_pred + DL_signature,
             "comprehensive",extra_cols="DL_signature")
