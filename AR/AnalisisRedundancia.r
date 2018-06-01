##Análisis de Redundancia.
## 1:20
#install.packages("far")
#install.packages("geigen")
#install.packages("matrixStats")
#install.packages("pracma")
#install.packages("vegan")
#install.packages("xlsx")
#install.packages("rstudioapi")

setwd("C:/Users/aleja/Documents/R/AR")

library(matrixStats)
library(geigen)
library(far)
library(pracma)
library(vegan)
library(xlsx)
library(rstudioapi)
source("AR.r")
source("preparacion.R")
source("DataGenerationAR.R")
source("DataExtractionAR.R")


p<-5
q<-4
n<-10
nombre="bad-drivers"
file="precipitation_"

M<-DataGenerationAR(n,p,q,nombre)
M<-DataExtractionAR(file,p,q)

X<-M[,1:p]
Y<-M[,(p+1):(p+q)]


W<-AR(X,Y)$vectors



