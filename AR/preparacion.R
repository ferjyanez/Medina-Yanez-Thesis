Preparacion <- function(X, Y){
  
  X <- t((t(X) - colMeans(X)) / colSds(X))

 if((dim(Y)[1])>1){
  Y <- t((t(Y) - colMeans(Y)) / colSds(Y))}
  else{
    Y <- t((t(Y) - mean(Y)) / sd(Y))
  }
  X <-gramSchmidt(X)$Q
  
  print("X ortonormalizada: ")
 
  M=cbind(X,Y)
  return(M)
}