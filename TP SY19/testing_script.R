# Loading packages
require(mxnet) # Neural networks
require(randomForest) 
require(e1071)

# Loading functions
load("functions.R")

# Loading datasets
expressions<-read.table("expressions_train.txt")
characters<-read.table("characters_train.txt")
parole<-read.table("parole_train.txt")

# Test classifiers

# # Deleting black pixels from the dataset expressions
# expressions <- expressions[,-which(expressions[1,] == 0)]
# 
# knn.test<-knn.cv(2,6,expressions, 10, type="fda")
# 
# # Display accuracy 
# print(knn.test)
