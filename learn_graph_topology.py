#
# Created on Wed May 26 2021 5:43:10 PM
#
# The MIT License (MIT)
# Copyright (c) 2021 Ashwin De Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Objective : Learn the underlying graph topology from the multi-channel sEMG signals

# ////// libraries ///// 
# standard 
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.manifold import TSNE
import scipy.io as sio

# internal 
from tma.utils import load_tma_data, plot_latent_space
from graph_learning.methods import SmoothAutoregressGraphLearn, SmoothSignalGraphLearn, AutoregressGraphLearn, CorrelationGraphLearn
from graph_learning.methods import PartialCorrelationGraphLearn
from graph_learning.utils import createWeightedGraph, plotMultiChannelSignals, rescaleMCW

# ////// body ///// 

## user inputs
#//////////////////

# multi-channel dataset
data_path = 'data/subject_1001_Ashwin/trans_map_dataset_3.h5'
class_labels = ['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer']
dataset_type = "test"

# graph dataset
graph_data_path = 'graph_data/subject_1001_Ashwin'

# method
useCorrelationGraphLearn = False
usePartialCorrelationGraphLearn = True
useSmoothAutoregressGraphLearn = False
useSmoothSignalGraphLearn = False

# hyperparameters
alpha=0.05
beta=20 
gamma=0.01 

# visualize
visualizeLearntGraphs = False 
visualizeLatentSpace = True

#//////////////////

## loading the TMA map data
X, y = load_tma_data(data_path)

## extracting the first order terms of each TMA map.
X = X[:, :8, :]     # the multi-channel envelopes (8-channels of myo armband)  
L = X.shape[1]      # number of channels

## learn the graph
Ws = np.empty((0, L, L))

if useCorrelationGraphLearn:
    method_id = "correlation"
    print("Method : {}".format(method_id))
    for i in range(X.shape[0]):
        GL = CorrelationGraphLearn(
            X=rescaleMCW(X[i]),
            alpha=alpha
        )
        W = GL.findGraph()
        if visualizeLearntGraphs:
            createWeightedGraph(W, False, "Class : {}".format(class_labels[int(y[i])]))
        Ws = np.vstack((Ws, W.reshape(1, L, L)))
        print('Graph {:n} : Completed!'.format(i+1))

if usePartialCorrelationGraphLearn:
    method_id = "partial_correlation"
    print("Method : {}".format(method_id))
    for i in range(X.shape[0]):
        GL = PartialCorrelationGraphLearn(
            X=rescaleMCW(X[i]),
            alpha=alpha
        )
        W = GL.findGraph()
        if visualizeLearntGraphs:
            createWeightedGraph(W, False, "Class : {}".format(class_labels[int(y[i])]))
        Ws = np.vstack((Ws, W.reshape(1, L, L)))
        print('Graph {:n} : Completed!'.format(i+1))

if useSmoothAutoregressGraphLearn:
    method_id = "smoothAutoregression"
    print("Method : {}".format(method_id))
    for i in range(X.shape[0]):
        GL = SmoothAutoregressGraphLearn(
            rescaleMCW(X[i]), 
            beta=beta,
            gamma=gamma
        )
        W = GL.findGraph()
        W[W <= 1e-5] = 0
        if visualizeLearntGraphs:
            createWeightedGraph(W, "Class : {}".format(int(y[i])))
        Ws = np.vstack((Ws, W.reshape(1, L, L)))
        print('Graph {:n} : Completed!'.format(i+1))

if useSmoothSignalGraphLearn:
    method_id = "smoothSignal"
    print("Method : {}".format(method_id))
    for i in range(X.shape[0]):
        GL = SmoothSignalGraphLearn(
            rescaleMCW(X[i]), 
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        W = GL.findGraph()
        W[W <= 1e-5] = 0
        if visualizeLearntGraphs:
            createWeightedGraph(W, "Class : {}".format(int(y[i])))
        Ws = np.vstack((Ws, W.reshape(1, L, L)))
        print('Graph {:n} : Completed!'.format(i+1))

if visualizeLatentSpace:
        Xr = TSNE(n_components=2, perplexity=50).fit_transform(Ws.reshape(Ws.shape[0], L*L))
        plot_latent_space(Xr, y, labels=class_labels)

## save
if input("Save Graph Data? (y/n) : ") == "y":
    dic = {"W" : Ws, "y" : y}
    file_name = "{}_{}_graph_topologies.mat".format(method_id, dataset_type)
    file_path = os.path.join(graph_data_path, file_name)
    sio.savemat(file_path, dic)
