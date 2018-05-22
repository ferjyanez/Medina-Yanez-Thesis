# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:48:41 2018

@author: Fernando J. Yanez

Name: RNA-TesisMedinaYanez
Last modification: 11May2018



installation for linux: https://www.tensorflow.org/install/install_sources 

data: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency#


import tensorflow as tf

#initial variables 

l_rate = tf.constant(0.2)
epochs = tf.constant(100)


W_ = tf.constant() #initialize a constant tensor
W = tf.Variable(W_, dtype=tf.float64) #gets a tensor with the values


#Data matrices
X_ = tf.constant() #initialize with the X data matrix
X = tf.placeholder(tf.float64)

Y_ = tf.constant() #initialize with the Y data matrix
Y = tf.placeholder(tf.float64)


#RNA model equivalent to RA
model_rna_redundancy = X*W*W.tanspose()*X.transpose()*Y

#Cost function
diff_sqr = tf.square(model_rna_redundancy - Y)
loss = tf.reduce_sum(diff_sqr)

#Optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate = l_rate)
train = opt.minimize(loss)

#training
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(epochs):
  session.run(train, {X:X_, Y:Y_})

#Get results
final_W, final_loss = session.run([W, loss], {X:X_, Y:Y_})
print("W: %s", final_W)
print("loss: %s", final_loss)

"""