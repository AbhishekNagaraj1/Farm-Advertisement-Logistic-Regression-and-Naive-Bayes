
# coding: utf-8

# In[ ]:

import pandas as pd
import collections, re
import numpy as np
import matplotlib.pyplot as plt
import math
import random

df = pd.read_csv("datamatrix.csv")
df1 = pd.read_csv("farm-ads-label.txt", sep = " ", header = None)

len1 = [str(i) for i in range(4143)]
features = df.as_matrix(columns = len1).T.astype(float)
labels = df1[1].values.astype(float)

def log_reg(value):
    log_reg_val = (1 / (1 + np.exp(-value)))
    return log_reg_val

def weights(x,y,fraction):
    weight = np.zeros(x.shape[1])
    #print weight
    for i in range(200000):
        a = random.sample(range(fraction), 2)
        #print a
        v = np.dot(x[a,:], weight)
        #print v
        fraction_out = np.take(y,a)
        out = log_reg(v)
        error = fraction_out - out
        new_w = np.dot(x[a,:].T, error)
        weight = weight + 5e-5*new_w
    return weight

w0 = np.ones((features.shape[0],1))
features1 = np.hstack((w0, features))
fraction = [0.1, 0.3 , 0.5, 0.7, 0.8, 0.9]
#fraction = [0.1]
Acc = []
for x,b in enumerate(fraction):
    length = [i for i in range(int(b*4143))]
    ip_train = features1[length,:]
    op_labels = np.take(labels, length)
    weight = weights(ip_train, op_labels, int(b*4143)) 
    length = [i for i in range(int(b*4143), 4143)]
    ip_test = features1[length,:]
    op_labels_test = np.take(labels, length)
    w_f = np.dot(ip_test, weight)
    predict = np.round(log_reg(w_f))
    predict_sum = (predict == op_labels_test).sum().astype(float)
    Acc.append(predict_sum / len(predict))
    print"Accuracy for:",b,":", Acc[x]
    
plt.plot(fraction, Acc)
plt.xlabel("Size of training Data")
plt.ylabel("Classification Accuracy")
plt.show()

