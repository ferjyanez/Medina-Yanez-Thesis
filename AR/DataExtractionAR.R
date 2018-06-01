#Generación de Datos
DataExtractionAR<-function(file,p,q){

  
  library(matrixStats)
  library(geigen)
  library(far)
  library(pracma)
  library(vegan)
  library(xlsx)
  
  MyData <- read.csv(file=(paste(file,".csv", sep="")), header=TRUE, sep=",")
  M<-data.matrix(MyData)
  
  X<-M[,2:(p+1)]
  Y<-M[,(p+2):(p+q+1)]
  
  M<-Preparacion(X,Y)
  MyData <-as.data.frame(M)
  write.xlsx(MyData, file = paste(file,".xlsx"))
  
  
  return (M)
}