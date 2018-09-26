#-------------------------------------------------------------------------------
# Multi-Layer Perceptron
#-------------------------------------------------------------------------------

mlp.cv <- function(r, k, data,hn,on,round,bsize,lr, type)
  
{
  err_rate=vector(length=r)
  
  # Cross-Validation
  for(i in 1:r){
    n = nrow(data)
    m = ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    # Cross-Validation
    for(j in (1:k)){
      e.train <-data[folds!=j,]
      e.test <-data[folds==j,]
      
      # FDA during cross-validation
      if(type=="fda"){
        e.train.lda <-lda(y~.,e.train)
        e.train.fda <-as.matrix(e.train[,1:m-1])%*%e.train.lda$scaling
        e.train.fda <-cbind(as.data.frame(e.train.fda), e.train$y)
        colnames(e.train.fda)[dim(e.train.fda)[2]] = "y"
        e.test.fda <- as.matrix(e.test[,1:(m-1)])%*%e.train.lda$scaling
        e.test.fda<-cbind(as.data.frame(e.test.fda), e.test$y)
        colnames(e.test.fda)[dim(e.test.fda)[2]] = "y"
        e.train<-e.train.fda
        e.test<-e.test.fda
      }
      # MLP model generation
      model <- mx.mlp(as.matrix(e.train[,1:ncol(e.train)-1]), e.train$y, hidden_node=hn, out_node=on, out_activation="softmax",
                      num.round=round, array.batch.size=bsize, learning.rate=lr, momentum=0.9,
                      eval.metric=mx.metric.accuracy, array.layout = "rowmajor")
      pred.test<-predict(model, as.matrix(e.test[,1:ncol(e.test)-1]))
      ntst<-nrow(e.test)
      pred.test.clean<-vector(length=ntst)
      
      # Finding the best class
      for (j in 1:ntst){
        pred.test.clean[j]<-which(pred.test[,j]==max(pred.test[,j]), arr.ind = T)-1
        if(as.numeric(pred.test.clean[j])==as.numeric(e.test$y[j])){
          CV=CV+1
        }
      }
    }
    err_rate[i] <- CV/n
  }
  
  # Displays
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate)) # accuracy
}

#-------------------------------------------------------------------------------
# Convolutional Neural Network
#-------------------------------------------------------------------------------

cnn <-function (data)
  
{ 
  # Expressions dataset only
  
  # Dataset preparation
  #-------------------------------------------------------------------------------
  expressions <- data
  napp <- 80
  nl <- nrow(expressions)
  ntst <- nl - napp
  train <- sample(1:nl, napp)
  e.test <- expressions[-train,]
  e.app <- expressions[train,]
  train_x <- t(e.app[, -4201])
  train_y <- e.app[, 4201]
  test_x <- t(e.test[, -4201])
  test_y <- e.test[, 4201]
  train_array <- train_x
  test_array <- test_x
  dim(train_array) <- c(60, 70, 1, ncol(train_x))
  dim(test_array) <- c(60, 70, 1, ncol(test_x))
  
  # Test half-matrix
  #-------------------------------------------------------------------------------
  # train_array<-train_array[1:30,1:70,1,]
  # dim(train_array)<-c(30,70,1,ncol(train_x))
  # test_array<-test_array[1:30,1:70,1,]
  # dim(test_array)<-c(30,70,1,ncol(test_x))
  
  
  # Set up the symbolic model
  #-------------------------------------------------------------------------------
  data <- mx.symbol.Variable('data')
  # 1st convolutional layer
  conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
  tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
  pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 2nd convolutional layer
  conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
  tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
  pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 1st fully connected layer
  flatten <- mx.symbol.Flatten(data = pool_2)
  fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
  tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
  # 2nd fully connected layer
  fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)
  
  # Pre-training set up
  #-------------------------------------------------------------------------------
  # Set seed for reproducibility
  mx.set.seed(100)
  # Device used. CPU in my case.
  devices <- mx.cpu()
  
  # Training
  #-------------------------------------------------------------------------------
  
  # Train the model
  model <- mx.model.FeedForward.create(NN_model,
                                       X = train_array,
                                       y = train_y,
                                       ctx = devices,
                                       num.round = 150,
                                       array.batch.size = 20,
                                       learning.rate = 0.01,
                                       momentum = 0.9,
                                       eval.metric = mx.metric.accuracy,
                                       epoch.end.callback = mx.callback.log.train.metric(100))
  
  # Testing
  #-------------------------------------------------------------------------------
  
  # Predict labels
  predicted <- predict(model, test_array)
  # Assign labels
  predicted_labels <- max.col(t(predicted)) - 1
  # Get accuracy
  return(sum(diag(table(test_y, predicted_labels)))/ntst)
}

#-------------------------------------------------------------------------------
# SVM
#-------------------------------------------------------------------------------

svm.cv <- function(r, k, data, cos, kern, gam)
  
{
  err_rate=vector(length=r)
  
  for(i in 1:r){
    n = nrow(data)
    m= ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    for(j in (1:k)){
      e.train=data[folds!=j,]
      e.test=data[folds==j,]
      if(type=="fda"){
        e.train.lda<-lda(y~.,e.train)
        e.train.fda<- as.matrix(e.train[,1:m-1])%*%e.train.lda$scaling
        e.train.fda<-cbind(as.data.frame(e.train.fda), e.train$y)
        colnames(e.train.fda)[dim(e.train.fda)[2]] = "y"
        e.test.fda<-as.matrix(e.test[,1:(m-1)])%*%e.train.lda$scaling
        e.test.fda<-cbind(as.data.frame(e.test.fda), e.test$y)
        colnames(e.test.fda)[dim(e.test.fda)[2]] = "y"
        e.train<-e.train.fda
        e.test<-e.test.fda
      }
      
      svm.app <- svm(y~., data=e.train , kernel = kern, cost=cos, scale=FALSE, gamma=gam)
      pred.test<-predict(svm.app, newdata=e.test)
      CV = CV + table(e.test$y, pred.test)
    }
    
    # print(CV)
    err_rate[i] <- sum(diag(CV))/n
  }
  
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate))
}

#-------------------------------------------------------------------------------
# Random Forest
#-------------------------------------------------------------------------------


rf.cv <- function(r, k, data, try, tree, type) 
  
{
  err_rate=vector(length=r)
  
  for(i in 1:r){
    n = nrow(data)
    m = ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    for(j in (1:k)){
      e.train=data[folds!=j,]
      e.test=data[folds==j,]
      if(type=="fda"){
        e.train.lda<-lda(y~.,e.train)
        e.train.fda<- as.matrix(e.train[,1:m-1])%*%e.train.lda$scaling
        e.train.fda<-cbind(as.data.frame(e.train.fda), e.train$y)
        colnames(e.train.fda)[dim(e.train.fda)[2]] = "y"
        e.test.fda<-as.matrix(e.test[,1:(m-1)])%*%e.train.lda$scaling
        e.test.fda<-cbind(as.data.frame(e.test.fda), e.test$y)
        colnames(e.test.fda)[dim(e.test.fda)[2]] = "y"
        e.train<-e.train.fda
        e.test<-e.test.fda
      }
      
      bag.e=randomForest(y~.,data=e.train,mtry=try, ntree=tree)
      pred.test=predict(bag.e,newdata=e.test,type="response")
      CV = CV + table(e.test$y, pred.test)
    }
    
    # print(CV)
    err_rate[i] <- sum(diag(CV))/n
  }
  
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate))
}

#-------------------------------------------------------------------------------
# Linear Discriminant Analysis
#-------------------------------------------------------------------------------

lda.cv <- function(r, k, data)   
{
  err_rate=vector(length=r)
  
  for(i in 1:r){
    n = nrow(data)
    m = ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    for(j in (1:k)){
      e.train=data[folds!=j,]
      e.test=data[folds==j,]
      lda.e=lda(y~.,data=e.train)
      pred.test=predict(lda.e,newdata=e.test)
      CV = CV + table(e.test$y, pred.test$class)
    }
    
    # print(CV)
    err_rate[i] <- sum(diag(CV))/n
  }
  
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate))
}

#-------------------------------------------------------------------------------
# K-Nearest-Neighbors
#-------------------------------------------------------------------------------

knn.cv <- function(r, k, data, nk, type)   
{
  err_rate=vector(length=r)
  
  for(i in 1:r){
    n = nrow(data)
    m = ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    for(j in (1:k)){
      e.train=data[folds!=j,]
      e.test=data[folds==j,]
      if(type=="fda"){
        e.train.lda<-lda(y~.,e.train)
        e.train.fda<- as.matrix(e.train[,1:m-1])%*%e.train.lda$scaling
        e.train.fda<-cbind(as.data.frame(e.train.fda), e.train$y)
        colnames(e.train.fda)[dim(e.train.fda)[2]] = "y"
        e.test.fda<-as.matrix(e.test[,1:(m-1)])%*%e.train.lda$scaling
        e.test.fda<-cbind(as.data.frame(e.test.fda), e.test$y)
        colnames(e.test.fda)[dim(e.test.fda)[2]] = "y"
        e.train<-e.train.fda
        e.test<-e.test.fda
      }
      
      model = knn(e.train[,1:ncol(e.train-1)], e.test[,1:col(e.train-1)], e.train$y, k = nk, prob = FALSE)
      CV = CV + table(e.test$y, model)
      
    }
    # print(CV)
    err_rate[i] <- sum(diag(CV))/n
  }
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate))
}

#-------------------------------------------------------------------------------
# Quadratic Discriminant Analysis
#-------------------------------------------------------------------------------

qda.cv <- function(r, k, data,type)   
{
  err_rate=vector(length=r)
  
  for(i in 1:r){
    n = nrow(data)
    m = ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    for(j in (1:k)){
      e.train=data[folds!=j,]
      e.test=data[folds==j,]
      if(type=="fda"){
        e.train.lda<-lda(y~.,e.train)
        e.train.fda<- as.matrix(e.train[,1:m-1])%*%e.train.lda$scaling
        e.train.fda<-cbind(as.data.frame(e.train.fda), e.train$y)
        colnames(e.train.fda)[dim(e.train.fda)[2]] = "y"
        e.test.fda<-as.matrix(e.test[,1:(m-1)])%*%e.train.lda$scaling
        e.test.fda<-cbind(as.data.frame(e.test.fda), e.test$y)
        colnames(e.test.fda)[dim(e.test.fda)[2]] = "y"
        e.train<-e.train.fda
        e.test<-e.test.fda
      }
      
      qda.e=qda(y~.,data=e.train)
      pred.test=predict(qda.e,newdata=e.test)
      CV = CV + table(e.test$y, pred.test$class)
    }
    
    #print(CV)
    err_rate[i] <- sum(diag(CV))/n
  }
  
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate))
}

#-------------------------------------------------------------------------------
# Naive Bayes Classifier
#-------------------------------------------------------------------------------


nb.cv <- function(r, k, data, type)
  
{
  err_rate=vector(length=r)
  
  for(i in 1:r){
    n = nrow(data)
    m = ncol(data)
    folds = sample(1:k,n,replace=TRUE)
    CV = 0
    
    for(j in (1:k)){
      e.train=data[folds!=j,]
      e.test=data[folds==j,]
      
      if(type=="fda"){
        e.train.lda<-lda(y~.,e.train)
        e.train.fda<- as.matrix(e.train[,1:m-1])%*%e.train.lda$scaling
        e.train.fda<-cbind(as.data.frame(e.train.fda), e.train$y)
        colnames(e.train.fda)[dim(e.train.fda)[2]] = "y"
        e.test.fda<-as.matrix(e.test[,1:(m-1)])%*%e.train.lda$scaling
        e.test.fda<-cbind(as.data.frame(e.test.fda), e.test$y)
        colnames(e.test.fda)[dim(e.test.fda)[2]] = "y"
        e.train<-e.train.fda
        e.test<-e.test.fda
      }
      
      model <- naiveBayes(y~.,e.train)
      pred.test<-predict(model, newdata=e.test)
      ntst<-nrow(e.test)
      pred.test.clean<-vector(length=ntst)
      #print(pred.test)
      
      for (j in 1:ntst){
        pred.test.clean[j]<-which(pred.test[j,1]==max(pred.test[j,1]), arr.ind = T)
        if(as.numeric(pred.test.clean[j])==as.numeric(e.test$y[j])){
          CV=CV+1
        }
      }
    }
    
    print(CV)
    err_rate[i] <- CV/n
  }
  
  # message("var : ",var(err_rate)) # variance of error rates
  # message("mean : ",mean(err_rate)) # mean error rate
  return(1-mean(err_rate))
}
