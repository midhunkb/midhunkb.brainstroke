# midhunkb.brainstroke

# Introduction 
As per the World Stroke Organization, approximately 25% of adults aged 25 or above will suffer from a stroke
 at some point in their lives. It is projected that this year alone, about 12.2 million individuals globally will
 encounter their first stroke, resulting in 6.5 million fatalities. To date, over 110 million people worldwide
 have experienced a stroke. The prevalence of stroke has reached epidemic proportions. The Heart and Stroke
 Foundation of Canada published a study in the Canadian Journal of Neurological Sciences on December 20.
 2022, revealing that the yearly occurrence of strokes in Canada has increased to 108,707 cases, translating
 to roughly one stroke transpiring every five minutes.
 For early detection of the risk factors and potentially minimizing the risk factors, information technologies,
 especially artificial intelligence, and machine learning can play a pivotal role. Here, for our project work,
 we have selected a dataset containing important risk factors for stroke to happen. This data is downloaded
 from the Kaggle library and constitutes of 12 features (both biochemical and psychological): age, sex,
 hypertension, underlying heart disease, marital status, smoking status, type of work, average glucose level,
 residence type, body mass index, the prevalence of past stroke.
 All of these factors contribute to having a stroke. Hence, we are trying to predict the correlation and variance
 in between the features of having a stroke using Principal Component Analysis. Given the necessary features,
 we are also trying to classify a particular patient having a stroke using the Logistic Regression model.
 

# Loading Necccsary Libraries
library(tidyverse)
library(caret)
library(class)
library(pROC)
library(glmnet)

# Load Data
data <- read.csv("stroke.csv")

data <- data %>%
  select(-id) %>% # Remove ID column
  drop_na() %>%  # Remove missing values
  mutate(
    gender = as.factor(gender),
    ever_married = as.factor(ever_married),
    work_type = as.factor(work_type),
    Residence_type = as.factor(Residence_type),
    smoking_status = as.factor(smoking_status),
    stroke = as.factor(stroke)
  )

# Train-Test Split
set.seed(123)
train_index <- createDataPartition(data$stroke, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Standardization
preproc <- preProcess(train_data %>% select(-stroke), method = c("center", "scale"))
train_scaled <- predict(preproc, train_data %>% select(-stroke))
test_scaled <- predict(preproc, test_data %>% select(-stroke))

train_final <- data.frame(train_scaled, stroke = train_data$stroke)
test_final <- data.frame(test_scaled, stroke = test_data$stroke)

# Logistic Regression Model
stroketib<-as_tibble(datacleaned1)
stroketibknn<-as_tibble(datacleaned1)
stroketask<-makeClassifTask(data = stroketib,target = "Stroke")
log_reg<-makeLearner("classif.logreg",predict.type = "prob")
logregmodel<-train(log_reg,stroketask)

# cross validating our model
kfold<-makeResampleDesc(method = "RepCV",folds = 10,reps = 50,stratify = TRUE)
logregcv<-resample(log_reg,stroketask,resampling = kfold,measures = list(acc,fpr,fnr))
# extracting model parameters

logregmodeldata<-getLearnerModel(logregmodel)
coef(logregmodeldata)

# converting model parameters into odds ratio
exp(cbind(odds_ratio = coef(logregmodeldata),confint(logregmodeldata)))

calculateConfusionMatrix(logregcv$pred,relative = TRUE)


# 2 nd time using the significant that is 97.5 greater than 1 


stroke_select_tib<-stroketib[,c(2,3,7,11)]
view(stroke_select_tib)

stroketask1<-makeClassifTask(data = stroke_select_tib,target = "Stroke")

log_reg1<-makeLearner("classif.logreg",predict.type = "prob")

logregmodel1<-train(log_reg1,stroketask1)

# cross validating our model

kfold1<-makeResampleDesc(method = "RepCV",folds = 10,reps = 50,stratify = TRUE)

logregcv1<-resample(log_reg1,stroketask1,resampling = kfold1,measures = list(acc,fpr,fnr))


# extracting model parameters

logregmodeldata1<-getLearnerModel(logregmodel1)

coef(logregmodeldata1)

# converting model parameters into odds ratio
exp(cbind(odds_ratio = coef(logregmodeldata1),confint(logregmodeldata1)))

calculateConfusionMatrix(logregcv1$pred,relative = TRUE)

logistic_model <- glm(stroke ~ ., data = train_final, family = binomial)
pred_probs <- predict(logistic_model, test_final, type = "response")
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)

# KNN Model
knn_preds <- knn(
  train = train_scaled,
  test = test_scaled,
  cl = train_data$stroke,
  k = 5
)

# Model Evaluation
evaluate_model <- function(actual, predicted) {
  confusion <- confusionMatrix(as.factor(predicted), as.factor(actual), positive = "1")
  roc_curve <- roc(actual, as.numeric(predicted))
  list(
    Confusion_Matrix = confusion,
    AUC = auc(roc_curve)
  )
}

logistic_results <- evaluate_model(test_data$stroke, pred_classes)
knn_results <- evaluate_model(test_data$stroke, knn_preds)


# Print Results
print(logistic_results)
print(knn_results)


# Conclusion
 The dataset that we used is the “Stroke Prediction Dataset” which contains information about patients who
 have suffered brain strokes. The dataset includes 12 features, including age, sex, hypertension, heart disease,
 smoking status, and average glucose level etc. The target variable is a binary variable indicating whether
 the patient has suffered a brain stroke (1) or not (0). The dataset contains 4981 records.
 Wewere able to understand more about data through exploratory analysis of the data. Relationship between
 Stroke and other variables were established through detailed analysis of the data. We then cleaned the data
 to do the Principal component analysis to reduce the dimension to find similarity between observation and
 Then created a logistic regression model to predict the probability of having stroke.
