import tensorflow as tf
import openpyxl as oxl
import pandas as pd
import numpy as np

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

#X_matrix = np.asmatrix(df_X)

#Y_matrix = np.asmatrix(df_Y)


#X_ = tf.constant(X_matrix, dtype = tf.float64, shape = [768,8])

#Y_ = tf.constant(Y_matrix, dtype = tf.float64, shape = [768,2])

#P = tf.constant([[2],[2]], dtype = tf.float64, shape = [2,1])

#Z = tf.matmul(tf.transpose(X_),Y_)

#sess = tf.Session()

#result = sess.run(Z)

print(xy)