AR <- function(X, Y){
  library(matrixStats)
  library(geigen)
  library(far)
  library(pracma)
  
  RXX<- (t(X)%*%X)

  RXY<- (t(X)%*%Y)

  RYX<- (t(Y)%*%X)

  RYY<- (t(Y)%*%Y)

  print("Tipo Kroonenberg")
  K<-eigen(RXY%*%RYX)
  print(K)

  W<-K$vectors

  IndiceRedundancia=sum(diag(t(W)%*%t(X)%*%Y%*%t(Y)%*%X%*%W))
  print("Indice Redundancia")
  print(IndiceRedundancia)
  return(K)
}