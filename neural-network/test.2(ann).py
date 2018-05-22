import tensorflow as tf
import openpyxl as oxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#tf.eye(5 is a identity matrix 5x5)

#initial variables 
l_rate = 0.00001
epochs = 500
mean_soft = 50

x_var = 5 #img-size
y_var = 1 #NUM_CLASSES
n_ind = 1503

X = tf.placeholder(tf.float32, [None,x_var]) #Inserts a placeholder for a tensor that will be always fed.
Y = tf.placeholder(tf.float32, [None,y_var]) #Inserts a placeholder for a tensor that will be always fed.

W = tf.Variable(tf.zeros([x_var,y_var])) #gets a tensor with the values
B = tf.Variable(tf.zeros([y_var,y_var]))
"""
wb = oxl.Workbook()
wb = oxl.load_workbook('/home/ferjyanez/Documents/Tesis/Medina-Yanez-Thesis/data/ENB2012_data.xlsx')
ws = wb.active
df = pd.DataFrame(ws.values)
df = df.drop(df.columns[[10, 11]], axis=1)
df = df[:769]
df = df.drop(df.index[0])

df_X = df.drop(df.columns[[8, 9]], axis=1)
df_Y = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7]], axis=1)
"""
xy = np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat')

X_matrix = np.delete(xy, 5, axis=1)
Y_matrix = np.delete(xy, np.s_[0:5], axis=1)

"""
print(df_X[0:2,])
print(50*'=')
print(df_Y[0:2,])


"""

#RNA model equivalent to RA
#model_rna_redundancy = X*W*W.tanspose()*X.transpose()*Y
#mmr_model = tf.add(tf.matmul(X,W), b)
aux = tf.matmul(X, W)
mmr_model = tf.matmul(aux, B)

#Cost function
cost = tf.square(Y - mmr_model)
loss = tf.reduce_sum(cost)

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
  errors.append(error_value)

#Get results
final_W, final_B, final_loss = sess.run([W, B, loss], {X:X_matrix, Y:Y_matrix})
print("W: %s"%final_W)
print("B: %s"%final_B)
print("loss: %s"%final_loss)

sess.close()

plt.plot([np.mean(errors[i-mean_soft:i]) for i in range(len(errors))])
plt.show()
#plt.savefig("errors.png")