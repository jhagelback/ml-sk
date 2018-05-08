#Load the ML library
library(caret)
library(randomForest)

#You might need to set working directory
setwd("/Users/jhg/Box Sync/DevProjects/ml-sk")

#Read the dataset
dataset <- read.csv("data/diabetes.csv")

#Split into 20% validation and 80% training
val_index <- createDataPartition(dataset$species, p=0.80, list=FALSE)
validation <- dataset[-val_index,]
training <- dataset[val_index,]

#Train ML models and estimate accuracy on test data
#setup 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Evaluate three different algorithms
#Support Vector Machines
set.seed(7)
fit.svm <- train(Diabetes~., data=training, method="svmRadial", metric=metric, trControl=control)
#k-Nearest Neighbors (kNN)
set.seed(7)
fit.nn <- train(Diabetes~., data=training, method="nnet", metric=metric, trControl=control)
#Random Forest
set.seed(7)
fit.rf <- train(Diabetes~., data=training, method="rf", metric=metric, trControl=control)

#Evaluate the models and check which one is best
#summarize accuracy of the models
results <- resamples(list(svm=fit.svm, nn=fit.nn, rf=fit.rf))
summary(results)
#plot spread and mean accuracy of each model (since 10-fold CV give 10 different accuracy measures)
dotplot(results)

#Use the best model and evaluate on the validation dataset
predictions <- predict(fit.rf, validation)
confusionMatrix(predictions, validation$Diabetes)