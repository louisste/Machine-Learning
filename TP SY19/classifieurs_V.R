classifieur_expressions <- function(dataset) {
  # Chargement des données construites lors de l'apprentissage (si besoin)
  load("env_V.Rdata")
  require(MASS)
  #suppression des pixels noirs
  data<-dataset[,-which(dataset[1,] == 0)]
  predictions <- predict(expressions.lda, newdata=dataset)
  return(predictions)
}

classifieur_characters <- function(dataset) {
  # Chargement de l'environnement
  load("env_V.Rdata")
  require(randomForest)
  # Mon algorithme qui renvoie les prédictions sur le jeu de données
  # 'dataset' fourni en argument.
  #scaling des données
  data<-dataset[,-1]
  data.scaled<-scale(data)
  data.scaled<-as.data.frame(data.scaled)
  data.scaled$Y<-dataset$Y
  predictions=predict(bag,newdata=data.scaled,type="response")
  return(predictions)
}
classifieur_parole <- function(dataset) {
  # Chargement de l'environnement
  load("env_V.Rdata")
  # Mon algorithme qui renvoie les prédictions sur le jeu de données
  # 'dataset' fourni en argument.
  require(e1071)
  data.scaled<-scale(dataset[,1:256])
  data.scaled<-as.data.frame(data.scaled)
  data.scaled$y<-dataset$y
  predictions<-predict(svm.p, newdata=data.scaled)
  return(predictions)
}