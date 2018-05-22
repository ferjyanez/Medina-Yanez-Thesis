#https://www.youtube.com/watch?v=90gpNF3KzK8

import tensorflow as tf
import numpy as np

#initial variables 

l_rate = 0.01
epochs = 20000


W = tf.Variable([.3], dtype=tf.float64) #gets a tensor with the values
#W_ = tf.assign(W, [-1.0])
b = tf.Variable([-.3], dtype=tf.float64)
#b_ = tf.assign(b, [1.0])

#Data matrices
X_ = [1,2,3,4] #initialize with the X data matrix
X = tf.placeholder(tf.float64) #Inserts a placeholder for a tensor that will be always fed.

Y_ = [0,-1,-2,-3] #initialize with the Y data matrix
Y = tf.placeholder(tf.float64) #Inserts a placeholder for a tensor that will be always fed.


#RNA model equivalent to RA
#model_rna_redundancy = X*W*W.tanspose()*X.transpose()*Y
lineal_model = X*W+b

#Cost function
loss = tf.reduce_sum(tf.square(lineal_model - Y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(l_rate)
train = optimizer.minimize(loss)

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(epochs):
  sess.run(train, {X: X_, Y: Y_})

#Get results
final_W, final_b, final_loss = sess.run([W, b, loss], {X:X_, Y:Y_})
print("W: %s"%final_W)
print("b: %s"%final_b)
print("loss: %s"%final_loss)

sess.close()