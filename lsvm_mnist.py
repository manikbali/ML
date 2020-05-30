# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:23:19 2020

@author: Manik Bali
"""

from mnist import MNIST
from skimage.color import rgb2grey
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import random

mndata = MNIST(r"C:/Users/Manik Bali/Documents/KD/SVM")
mndata.gz=True
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
np.array(range(1000,60000,1000) ).shape[0]
bias_array=np.array(range(1000,60000,1000) ).shape[0]
n=0
for nimages in range(1000,60000,1000):
    train_images=np.array(train_images[0:nimages]) 
    train_labels=np.array(train_labels[0:nimages]) 

#train_images=np.array(train_images) 
#train_labels=np.array(train_labels) 


    test_images=np.array(test_images) 
    test_labels=np.array(test_labels) 

    pca=PCA(n_components=120)
    pca_vectors = pca.fit_transform(train_images)

    cumulative_variance=np.cumsum(pca.explained_variance_ratio_)
#
    
    proj_data= np.matmul(train_images, pca.components_.T)

    test_proj_data = np.matmul(test_images, pca.components_.T)
#ssvm = svm.SVC(kernel='linear', probability=True, random_state=42)
#
    ssvm = svm.SVC(kernel='linear')
    a=ssvm.fit(proj_data, train_labels)
    y_pred = ssvm.predict(test_proj_data)
#The output array is test_labels
    bias=test_labels - y_pred
#    plt.plot(bias[np.where(bias ==0)].shape[0]))
    
    bias_array[n] =  np.array(np.where(bias ==0)).shape[1]
    n=n+1

#index = random.randrange(0, len(train_images))]=
#plt.imshow(np.array(images[index]).reshape(28,28))

#classes =train_labels
#clf=svm.SVC()
#clf.fit(train_images)
 
 
