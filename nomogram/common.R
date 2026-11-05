suppressPackageStartupMessages({ library(rms); library(pROC); library(readr) })
input_dir <- "data/processed"
output_dir <- "results/nomogram"
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
clinical_vars <- c("Ki_67", "SLN_diameter", "HER_2", "cortex_thickness", "N_grade_pred")
load_data <- function(required) {
  files <- file.path(input_dir, c("train_all.csv", "test_all.csv"))
  if (!all(file.exists(files))) stop("Missing input CSV(s) under data/processed/.")
  train <- as.data.frame(readr::read_csv(files[1], show_col_types=FALSE))
  test <- as.data.frame(readr::read_csv(files[2], show_col_types=FALSE))
  for (nm in c("patient_id", "sln_id", "label", required)) {
    if (!(nm %in% names(train)) || !(nm %in% names(test))) stop("Missing column: ", nm)
  }
  numeric_vars <- unique(c("label", required))
  for (nm in numeric_vars) {
    train[[nm]] <- suppressWarnings(as.numeric(as.character(train[[nm]])))
    test[[nm]] <- suppressWarnings(as.numeric(as.character(test[[nm]])))
  }
  if (anyNA(train[numeric_vars]) || anyNA(test[numeric_vars])) stop("Missing/non-numeric model inputs; prepare data before running.")
  if (!all(train$label %in% c(0,1)) || !all(test$label %in% c(0,1))) stop("label must be 0/1.")
  if (length(unique(train$label)) != 2) stop("Training data must include both outcomes.")
  if (anyDuplicated(train$sln_id) || anyDuplicated(test$sln_id)) stop("sln_id must be unique within each split.")
  if (length(intersect(unique(train$patient_id),unique(test$patient_id)))) stop("Patient overlap between train and test.")
  list(train=train, test=test)
}
# Match the archived scripts: log((p + 1e-6)/(1 - p + 1e-6)).
prob_to_logit <- function(p) {
  if (any(!is.finite(p)) || any(p < 0 | p > 1)) stop("Image probabilities must lie in [0,1].")
  eps <- 1e-6
  log((p + eps) / (1 - p + eps))
}
run_nomogram <- function(train, test, formula, model_name, extra_cols=character()) {
  dd <- rms::datadist(train)
  # rms looks up the object by name in the environment used to fit the model.
  assign("dd", dd, envir=.GlobalEnv)
  old <- options(datadist="dd")
  on.exit(options(old), add=TRUE)
  fit <- rms::lrm(formula, data=train, x=TRUE, y=TRUE)
  print(fit)
  nom <- rms::nomogram(fit, fun=plogis, funlabel="Probability of SLN Metastasis", lp=FALSE)
  grDevices::pdf(file.path(output_dir,paste0(model_name,"_nomogram.pdf")),width=15,height=8)
  plot(nom); grDevices::dev.off()
  train$nomogram_pred <- as.numeric(predict(fit,newdata=train,type="fitted"))
  test$nomogram_pred <- as.numeric(predict(fit,newdata=test,type="fitted"))
  auc_value <- function(dat) if (length(unique(dat$label)) < 2) NA_real_ else as.numeric(pROC::auc(pROC::roc(dat$label,dat$nomogram_pred,levels=c(0,1),direction="<",quiet=TRUE)))
  scores <- data.frame(dataset=c("train","test"), AUC=c(auc_value(train),auc_value(test)))
  write.csv(scores,file.path(output_dir,paste0(model_name,"_auc.csv")),row.names=FALSE)
  columns <- unique(c("patient_id","sln_id","label",extra_cols,"nomogram_pred"))
  out <- rbind(transform(train[,columns,drop=FALSE],dataset="train"),transform(test[,columns,drop=FALSE],dataset="test"))
  write.csv(out,file.path(output_dir,paste0(model_name,"_predictions.csv")),row.names=FALSE)
  saveRDS(fit,file.path(output_dir,paste0(model_name,"_model.rds")))
  # Internal bootstrap calibration of training model, NOT external-test calibration.
  set.seed(27)
  cal <- rms::calibrate(fit,method="boot",B=1000)
  grDevices::pdf(file.path(output_dir,paste0(model_name,"_bootstrap_calibration.pdf")),width=6,height=6)
  plot(cal,xlab="Predicted Probability",ylab="Observed Probability",subtitles=FALSE)
  grDevices::dev.off()
  invisible(fit)
}
