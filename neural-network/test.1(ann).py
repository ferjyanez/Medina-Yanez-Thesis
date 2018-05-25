import tensorflow as tf
import openpyxl as oxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

import os
duration = 1  # second
freq = 440  # Hz

#initial variables 
l_rate_0 = 0.001
epochs = 100

wb = oxl.Workbook()
wb = oxl.load_workbook('/home/ferjyanez/Documents/Tesis/Medina-Yanez-Thesis/data/MyData.xlsx')
ws = wb.active
df = pd.DataFrame(ws.values)

df = df.drop(df.columns[[0]], axis=1)
df = df[:201]
df = df.drop(df.index[0])

df_X = df.drop(df.columns[[4, 5]], axis=1)
df_Y = df.drop(df.columns[[0, 1, 2, 3]], axis=1)

#X = np.asmatrix(df_X, dtype = np.float32)
#Y = np.asmatrix(df_Y, dtype = np.float32)

X_matrix = np.asmatrix(df_X)
Y_matrix = np.asmatrix(df_Y)

#Sizes
x_var = X_matrix.shape[1] #x_var
y_var = Y_matrix.shape[1] #y_var
n_ind = X_matrix.shape[0] #n_ind

X = tf.placeholder(tf.float32, [None,x_var]) #Inserts a placeholder for a tensor that will be always fed.
Y = tf.placeholder(tf.float32, [None,y_var]) #Inserts a placeholder for a tensor that will be always fed.

W = tf.Variable(tf.truncated_normal([x_var,y_var], stddev=0.01)) #gets a tensor with the values

model = tf.matmul(tf.matmul(tf.matmul(tf.matmul(X,W),tf.transpose(W)),tf.transpose(X)),Y) #Y estimada

#shooting
#pkeep = tf.placeholder(tf.float32)
#Y = tf.nn.droput(model, pkeep) #se envia una probabilidad (0,75) cuando se corre la sesion y deshabilita
                                    #el 25% de las salidas de cada capa, cuando se testea se manda un 1

#Cost function
cost0 = tf.square(Y - model) #subtracts element-wise the vector and the elevates them to the 2nd power
cost1 = tf.reduce_sum(cost0)
cost2 = 0
#cost2 = tf.trace(tf.matmul(tf.transpose(tf.subtract(tf.matmul(tf.transpose(W),W),tf.eye(y_var))),tf.subtract(tf.matmul(tf.transpose(W),W),tf.eye(y_var))))
loss = tf.add(cost1,cost2)  #adds all of the elements plus the error for normalization

loss2 = tf.trace(tf.matmul(tf.transpose(W),tf.matmul(tf.transpose(X),tf.matmul(Y, tf.matmul(tf.transpose(Y),tf.matmul(X,W))))))
#print(np.trace(final_W.transpose().dot(X_matrix.transpose()).dot(Y_matrix).dot(Y_matrix.transpose()).dot(X_matrix).dot(final_W)))

# % of correct answers found
#differs_Y = tf.square(tf.subtract(Y, model))
#accuracy_Y = tf.reduce_mean(tf.cast(differs_Y, tf.float32)) #suma la media de los elementos de differs_Y mientras que tf.cast redefine el tipo de variable
#differs_W = tf.subtract(tf.matmul(W,tf.transpose(W)),tf.eye(x_var))
#accuracy_W = tf.reduce_sum(tf.square(tf.cast(differs_Y, tf.float32))) #suma la media de los elementos de differs_Y mientras que tf.cast redefine el tipo de variable

#Optimizer
global_step = tf.Variable(0, trainable=False)
#l_rate = tf.train.exponential_decay(l_rate_0, global_step, 100000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(l_rate_0)
train = optimizer.minimize(loss)

errors_e = []
errors_n = []

#training
#initialise the global variables in order to run the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(epochs):
  _, ir = sess.run([train, loss], {X: X_matrix, Y: Y_matrix})
  errors_e.append(ir)
  #errors_e.append(error_n/n_ind)

  #if(epochs%(epochs/20)==0):
  #  acc_Y, acc_W = sess.run([accuracy_Y, accuracy_W], {X: X_matrix, Y: Y_matrix})

    



#Get results

final_W, ir = sess.run([W, loss], {X:X_matrix, Y:Y_matrix})
#EE = tf.self_adjoint_eigvals(final_W)
print("W: %s"%final_W)
#print("autovalores: ", np.linalg.eig(final_W))
print("loss: %s"%(ir/n_ind))
#print("loss n: %s"%final_n/n_ind)
print(np.trace(final_W.transpose().dot(X_matrix.transpose()).dot(Y_matrix).dot(Y_matrix.transpose()).dot(X_matrix).dot(final_W)))
print(final_W.transpose().dot(final_W))

sess.close()

os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

#x_plot = list(range(len(errors_e)))
#ax1 = plt.subplot(2, 1, 1)
plt.plot(np.asarray(errors_e))
plt.ylabel("MSE")
plt.show()
#ax2 = plt.subplot(2, 1, 2)
#plt.plot(np.asarray(errors_n))
#plt.ylabel('Norm')
#plt.savefig("errors.png")

#learning rate decay
#"""