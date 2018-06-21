
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

#dfx=df.T
#df11 = df.iloc[:,0:]
#print df11
#print "dfx:",dfx[0].values

#newdf = df.columns[0:4143]
#print newdf.values

#arr1 = newdf.values


def accepted(sum_accepted_ads,training_data,indices_words):
    prob = np.log10(float(sum_accepted_ads)/len(training_data))
    if any(accepted_ads.ix[i].sum()==0 for i in indices_words):
        for i in indices_words:
            prob = prob + np.log10(float(accepted_ads.ix[i].sum() + 1)/(sum_accepted_ads + v))
    else:
        for i in indices_words:
            prob = prob + np.log10(float(accepted_ads.ix[i].sum())/(sum_accepted_ads))
    return prob

def rejected(sum_rejected_ads,training_data,indices_words):
    prob = np.log10(float(sum_rejected_ads)/len(training_data))
    
    if any(rejected_ads.ix[i].sum()==0 for i in indices_words):
        for i in indices_words:
            prob = prob + np.log10(float(rejected_ads.ix[i].sum() + 1)/(sum_rejected_ads + v))
    else:
        for i in indices_words:
            prob = prob + np.log10(float(rejected_ads.ix[i].sum())/(sum_rejected_ads))
    return prob


training_size = [0.1,0.3,0.5,0.7,0.8,0.9]

v = len(df)
numberofads = len(df1)
count = 0
acc = []
for x,i in enumerate(training_size):    
    size = int(i*numberofads)
    test_list = [str(n) for n in range(size,numberofads)]
    for j in test_list:
        indices_words = df.index[df[j]>0].tolist()
        training_df = df1.ix[:size]
        accepted_index = []
        rejected_index = []
        arr = df1[1].values
        #print len(arr)
        for i in range (len(arr)):
            if arr[i] == 0:
                accepted_index.append(i)
            else:
                rejected_index.append(i)


            #dfx = df1[0].values
        #print accepted_index
        #print "----"
        #print rejected_index

        accepted_index = [str(x) for x in accepted_index]
        accepted_ads = df[accepted_index]
        sum_accepted_ads = accepted_ads.sum().sum()
        #print sum_accepted_ads
        prob_accepted = accepted(sum_accepted_ads,training_df, indices_words)


        rejected_index = [str(x) for x in rejected_index]
        rejected_ads = df[rejected_index]
        sum_rejected_ads = rejected_ads.sum().sum()
        #print sum_rejected_ads

        prob_rejected = rejected(sum_rejected_ads,training_df,indices_words)



        if prob_accepted >= prob_rejected:
            label = 1
        else:
            label = 0

        if (label == df1.ix[int(j)][1]):
            count+=1
            
        acc.append(float(count)/numberofads - int(i*numberofads))
        print "Accuracy:",i,":",acc[x]

plt.plot(training_size, acc)
plt.xlabel(training_size)
plt.ylabel(acc)
plt.show()

