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
l_rate = 0.0000000001
epochs = 100

""" x_var = 3 #img-size
y_var = 2 #NUM_CLASSES
n_ind = 50  """

""" wb = oxl.Workbook()
wb = oxl.load_workbook('/home/ferjyanez/Documents/Tesis/Medina-Yanez-Thesis/data/datosfer.xlsx')
ws = wb.active
df = pd.DataFrame(ws.values)
df = df.drop(df.columns[[0]], axis=1)
df = df[:51]
df = df.drop(df.index[0])

df_X = df.drop(df.columns[[3, 4]], axis=1)
df_Y = df.drop(df.columns[[0, 1, 2]], axis=1) """

wb = oxl.Workbook()
wb = oxl.load_workbook('/home/ferjyanez/Documents/Tesis/Medina-Yanez-Thesis/data/DatosYahoo.xlsx')
ws = wb.active
df = pd.DataFrame(ws.values)

df = df.drop(df.columns[[0]], axis=1)
df = df[:201]
df = df.drop(df.index[0])

df_X = df.drop(df.columns[[4, 5]], axis=1)
df_Y = df.drop(df.columns[[0, 1, 2, 3]], axis=1)

X = np.asmatrix(df_X, dtype = np.float32)
Y = np.asmatrix(df_Y, dtype = np.float32)

X = ss.zscore(X, axis=0)                      ####revisar si es 0 o 1
Y = ss.zscore(Y, axis=0)

X_matrix = np.asmatrix(df_X)
Y_matrix = np.asmatrix(df_Y)

#Sizes
x_var = X.shape[1] #x_var
y_var = Y.shape[1] #y_var
n_ind = X.shape[0] #n_ind

X = tf.placeholder(tf.float32, [None,x_var]) #Inserts a placeholder for a tensor that will be always fed.
Y = tf.placeholder(tf.float32, [None,y_var]) #Inserts a placeholder for a tensor that will be always fed.

W = tf.Variable(tf.truncated_normal([x_var,y_var], stddev=0.01)) #gets a tensor with the values



model = tf.matmul(tf.matmul(tf.matmul(tf.matmul(X,W),tf.transpose(W)),tf.transpose(X)),Y) #Y estimada

#shooting
#pkeep = tf.placeholder(tf.float32)
#Y = tf.nn.droput(model, pkeep) #se envia una probabilidad (0,75) cuando se corre la sesion y deshabilita
                                    #el 25% de las salidas de cada capa, cuando se testea se manda un 1

#Cost function
cost = tf.square(Y - model) #subtracts element-wise the vector and the elevates them to the 2nd power
loss = tf.reduce_sum(cost)  #adds all of the elements

# % of correct answers found
differs_Y = tf.square(tf.subtract(Y, model))
accuracy_Y = tf.reduce_mean(tf.cast(differs_Y, tf.float32)) #suma la media de los elementos de differs_Y mientras que tf.cast redefine el tipo de variable
differs_W = tf.subtract(tf.matmul(W,tf.transpose(W)),tf.eye(x_var))
accuracy_W = tf.reduce_sum(tf.square(tf.cast(differs_Y, tf.float32))) #suma la media de los elementos de differs_Y mientras que tf.cast redefine el tipo de variable

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(l_rate)
train = optimizer.minimize(loss)

errors = []

#training
#initialise the global variables in order to run the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(epochs):
  _, error_value = sess.run([train, loss], {X: X_matrix, Y: Y_matrix})
  print(i)
  print(error_value)
  errors.append(error_value/n_ind)

  if(epochs%(epochs/20)==0):
    acc_Y, acc_W = sess.run([accuracy_Y, accuracy_W], {X: X_matrix, Y: Y_matrix})

    



#Get results
final_W, final_loss = sess.run([W, loss], {X:X_matrix, Y:Y_matrix})
print("W: %s"%final_W)
print("loss: %s"%final_loss)

sess.close()

os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

plt.plot([np.mean(errors[i-1:i]) for i in range(len(errors))])
plt.show()
#plt.savefig("errors.png")

#learning rate decay
#"""