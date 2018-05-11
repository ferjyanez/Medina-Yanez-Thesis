# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:48:41 2018

@author: Fernando J. Yanez

Name: RNA-TesisMedinaYanez
Last modification: 11May2018

"""

import numpy as np
import pandas as pd
import math


#==============================================================================
#Variables globales
#==============================================================================
#Coeficiente de aprendizaje
etha = 0.2

#Numero de capas ocultas
numberOfHiddenLayers = 1

#Numero de corridas maximo para el entrenamiento
numberOfRuns = 1

#Numero de neuronas en la capa oculta
L1 = 0
#==============================================================================
    


#==============================================================================
#Toma de la data
#==============================================================================
##Importa la data de un archivo csv 
#RawData = pd.read_csv('path\file.csv')
##Transforma la data en una matriz n*(p+q)
#RawData = pd.DataFrame.as_matrix(RawData)
##Importa los tamaños de ambos conjuntos de variables de un archivo csv (ver como se puede introducir como argumento al codigo)
#sizes = pd.read_csv('path\file2.csv')
##Transforma en un vector 1x2
#sizes = pd.DataFrame.as_matrix(sizes)


#p = sizes[0]
#q = sizes[1]

p = 5
q = 2
#Define el numero de neuronas ocultas
L1 = min(p, q)
#==============================================================================
    


##==============================================================================
##Separacion de la data
##==============================================================================
#X = np.zeros(len(RawData)*p).reshape(len(RawData), p)
#Y = np.zeros(len(RawData)*q).reshape(len(RawData), q)
#
##Llena las matrices creadas con la data
#for i in range (0, len(RawData)):
#    for j in range (0, len(X[0])):
#        X[i][j] = RawData[i][j]
#    for j in range (len(X[0]), len(X[0])+len(Y[0])):    
#        Y[i][j-len(X[0])] = RawData[i][j]
##============================================================================== 
X = np.arange(20*p).reshape(20,p)
X = np.asmatrix(X)
Y = np.arange(20*q).reshape(20,q)
Y = np.asmatrix(Y)
Y[0][0] = 1
#==============================================================================    
#Red Neuronal Artificial
#==============================================================================
#Creacion de la matriz W de pesos entre la capa de entrada y la capa oculta

#Should be normal or uniform????? It wil be trained so it doent matter(?)###########################
W = np.random.normal(0, 0.2, p*L1).reshape(p, L1)
W = np.asmatrix(W)    

#Creacion de la matriz P de pesos entre la capa oculta y la capa de salida  
P = np.random.normal(0, 0.2, L1*q).reshape(L1, q)
P = np.asmatrix(P)  

contError = 1
#Starts the process
for i in range (0,numberOfRuns):#numero de corridas
    if(contError==0):#condicion de parada         
        break
    else:
        contError = 0        
        for j in range(0,len(X)):#number of entrances for training
            #Calculates de matrix a1 of entrances for the activation function in the hidden layer 1
            aux = X[j].dot(W)                
            
            Ycalc = aux.dot(P)
            print("Ycalc")            
            print(Ycalc)
    #==============================================================================
            
  
    #==============================================================================       
            #Learning process
    #==============================================================================
            delta1 = np.zeros(q)
            delta1 = np.asmatrix(delta1)
            delta2 = np.zeros(q)   
            delta2 = np.asmatrix(delta2)           
            for k in range(0,Y.shape[1]):         
                #Tolerance for the error of 10% over the non-normalized data                
                if(Ycalc[0,k]/Y[j,k]>0.95 and Ycalc[0,k]/Y[j,k]<1.05):
                    #If the predicted value is within the 10% error, don't learn                        
                    #delta1 = np.zeros(q).reshape(q)
                    #delta2 = np.zeros(q).reshape(q)
                    print("")
                else:
                    contError = contError + 1
                #Difference between the output and the spected value for the output layer
                    print(Y[j,k])
                    print(Ycalc[0,k])
                    print("")
                    delta1[0,k] = (Y[j,k]-Ycalc[0,k]) #A number
                    print("delta1")            
                    print(delta1)
                    print("")
            #Difference between the output and the spected value for the hidden layer
            delta2 = delta1.dot(P) #A vector 1xL1
            print("delta2")            
            print(delta2)                    
            #Updates the weight for the output neuron
            print("..")
            print(delta1.transpose().dot(Ycalc))
            P = P + (etha*aux.transpose().dot(delta1))        
            #Updates the weight for the hidden layer
            W = W + etha*X[j].transpose().dot(delta2)

