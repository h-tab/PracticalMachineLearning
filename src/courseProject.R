library(caret)
library(randomForest)
library(foreach)
library(doParallel)

set.seed(12345) #to provide reproducibility

trainingUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
evaluationUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#load data
trainingData <- read.csv(url(trainingUrl), na.strings = c("NA", "#DIV/0!", ""))
validationData <- read.csv(url(evaluationUrl), na.strings = c("NA", "#DIV/0!", ""))

#use data from accelerometers on the belt, forearm, arm, and dumbell
features <- colnames(trainingData[colSums(is.na(trainingData)) == 0])[-(1:7)] 
modelData <- trainingData[features]
print("Features, that are used: ")
features

#training data has to be partinioned: 75% for training and 25% for testing, 
#this will provide cross validation
initialTraining <- createDataPartition(y = modelData$classe, p = 0.75, list = FALSE )
trainingModel <- modelData[initialTraining,]
testingModel <- modelData[-initialTraining,]

registerDoParallel() #thanks to Jeff Heaton with his idea to use parallel processing, this is interesting
x <- trainingModel[-ncol(trainingModel)]
y <- trainingModel$classe

#predict with random forests algorithm
randomForestValues <- 
    foreach(ntree = rep(300, 5), .combine = randomForest::combine, .packages = 'randomForest') %dopar% 
{
    randomForest(x, y, ntree = ntree) 
}

predictions1 <- predict(randomForestValues, newdata = trainingModel)
confusionMatrix(predictions1, trainingModel$classe)


predictions2 <- predict(randomForestValues, newdata = testingModel)
confusionMatrix(predictions2, testingModel$classe)

#Prediction Assignment Submission
pmlWriteFiles = function(x)
{
    n = length(x)
    for(i in 1:n)
    {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i],file = filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}

x <- validationData
x <- x[features[features != 'classe']]
answers <- predict(randomForestValues, newdata=x)

print("Predicted values to sumbit: ")
answers

pmlWriteFiles(answers)
