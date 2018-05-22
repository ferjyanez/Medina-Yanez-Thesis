#https://www.youtube.com/watch?v=90gpNF3KzK8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#initial variables 

l_rate = 0.01
epochs = 3000


W = tf.Variable([[.3], [.3]], dtype=tf.float32) #gets a tensor with the values
#W_ = tf.assign(W, [-1.0])
b = tf.Variable([-.3], dtype=tf.float32)
#b_ = tf.assign(b, [1.0])

#Data matrices
#X_ = [[1,1],[1,4],[2,0],[5,7],[3,9],[5,3],[9,5],[2,7],[8,8],[2,9]] #initialize with the X data matrix
X = tf.placeholder(tf.float32) #Inserts a placeholder for a tensor that will be always fed.

#Y_ = [5,7,4,14,14,10,16,11,18,13] #initialize with the Y data matrix
Y = tf.placeholder(tf.float32) #Inserts a placeholder for a tensor that will be always fed.


#RNA model equivalent to RA
#model_rna_redundancy = X*W*W.tanspose()*X.transpose()*Y
lineal_model = tf.add(tf.matmul(X,W), b)

#Cost function
cost = tf.square(Y - lineal_model)
loss = tf.reduce_sum(cost)

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(l_rate)
train = optimizer.minimize(loss)

errors = []

#training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(epochs):
  X_val = tf.random_normal([1,2], mean=5, stddev=2.0)
  W_ = tf.constant((1.0), shape = [2,1])
  Y_val = tf.add(tf.matmul(X_val, W_), 2)
  X_, Y_ = sess.run([X_val, Y_val])
  #print(X_)
  #print(Y_)
  _, error_value = sess.run([train, cost], {X: X_, Y: Y_})
  errors.append(error_value)

#Get results
final_W, final_b, final_loss = sess.run([W, b, loss], {X:X_, Y:Y_})
print("W: %s"%final_W)
print("b: %s"%final_b)
print("loss: %s"%final_loss)

sess.close()

#print (errors)

plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()
#plt.savefig("errors.png")