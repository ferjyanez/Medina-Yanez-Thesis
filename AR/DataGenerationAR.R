#Generación de Datos
DataGenerationAR <- function(n,p,q,nombre){
  

  
  library(matrixStats)
  library(geigen)
  library(far)
  library(pracma)
  library(vegan)
  library(xlsx)
  
media<-0
des<-1

X<-ones(max(n,p,q))[,1:p]
Y<-ones(max(n,p,q))[,1:q]
for (i in 1:p){
  X[,i]<-rnorm(n,mean=media,sd=des)
}

for (i in 1:q){
  Y[,i]<-rnorm(n,mean=media,sd=des)
}

M<-Preparacion(X,Y)
MyData <-as.data.frame(M)
write.xlsx(MyData, file = paste(nombre,".xlsx", sep=""))


return (M)
}