#Load the ML library
library(caret)
library(randomForest)

#You might need to set working directory
#setwd("/Users/jhg/Box Sync/Lectures/Applied ML (4DV117)/examples-R")

#Read the dataset
dataset <- read.csv("iris.csv")

#Split into 20% validation and 80% training
val_index <- createDataPartition(dataset$species, p=0.80, list=FALSE)
validation <- dataset[-val_index,]
training <- dataset[val_index,]

#Train ML models and estimate accuracy on test data
#setup 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Evaluate two different models
#Classification and Regression Trees (CART)
set.seed(7)
fit.cart <- train(species~., data=training, method="rpart", metric=metric, trControl=control)
#k-Nearest Neighbors (kNN)
set.seed(7)
fit.knn <- train(species~., data=training, method="knn", metric=metric, trControl=control)
#Random Forest
set.seed(7)
fit.rf <- train(species~., data=training, method="rf", metric=metric, trControl=control)

#Evaluate the models and check which one is best
#summarize accuracy of the models
results <- resamples(list(cart=fit.cart, knn=fit.knn, rf=fit.rf))
summary(results)
#plot spread and mean accuracy of each model (since 10-fold CV give 10 different accuracy measures)
dotplot(results)

#Use the best model and evaluate on the validation dataset
predictions <- predict(fit.knn, validation)
confusionMatrix(predictions, validation$species)