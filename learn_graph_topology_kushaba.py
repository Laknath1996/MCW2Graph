#
# Created on Wed Apr 14 2021 7:12:38 PM
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
# Objective : Use this script to run developer experiments 
#

#////// libraries //////

# standard 
import os
import numpy as np
import scipy.io as sio
from sklearn.manifold import TSNE

# internal
from graph_learning.methods import DiffusionGraphLearn, GraphicalLassoGraphLearn
from graph_learning.utils import createWeightedGraph
from data_handling.utils import create_dataset, plot_recording
from tma.utils import plot_latent_space

#////// body //////

## User Inputs
id = "kushaba"
s_id = 1
labels = ["M_M", "R_R", "I_I", "L_L", "HC_"] # "T_I", "T_M", "T_R", "T_L", "I_M", "M_R", "R_L", "IMR", "MRL"]
visualizeLearntGraphs = True
visualizeLatentSpace = True
L = 8

## Select the Graph Learning Method
useJustMCW = False
useGraphicalLassoGraphLearn = False
useDiffusionGraphLearn = True

## Create dataset
X, y = create_dataset(id, s_id, labels)

## collect the learnt the W matrices
Ws = np.empty((0, 8, 8))

if useJustMCW:
    Ws = X

if useDiffusionGraphLearn:
    # hyperparams
    p = 5
    beta_1 = 10
    beta_2 = 0.1

    for i in range(X.shape[0]):
        GL = DiffusionGraphLearn(
            1000*X[i],
            p=p, 
            beta_1=beta_1,
            beta_2=beta_2,
            verbosity=True
        )
        W = GL.findGraphLaplacian()
        Ws = np.vstack((Ws, np.expand_dims(W, 0)))
        if visualizeLearntGraphs:
            createWeightedGraph(W, "Class : {}".format(int(y[i])))
        print('Graph {:n} : Completed!'.format(i+1))

if useGraphicalLassoGraphLearn:
    # hyperparams
    beta = 0.5
    gamma = 0.0001
    imax = 2000
    epsilon = 0.01
    alpha = 0.05

    for i in range(X.shape[0]):
        GL = GraphicalLassoGraphLearn(
            X=10000*X[i],
            beta=beta,
            gamma=gamma,
            imax=imax,
            epsilon=epsilon,
            alpha=alpha,
            verbosity=False
        )
        W = GL.findGraph()
        if visualizeLearntGraphs:
            createWeightedGraph(W, "Class : {}".format(labels[int(y[i])]))
        Ws = np.vstack((Ws, W.reshape(1, L, L)))
        print('Graph {:n} : Completed!'.format(i+1))

## Visualize the Latent Space
if visualizeLatentSpace:
    if useJustMCW:
        Xr = TSNE(n_components=2, perplexity=60).fit_transform(Ws.reshape(Ws.shape[0], L*80)) 
    else:   
        Xr = TSNE(n_components=2, perplexity=60).fit_transform(Ws.reshape(Ws.shape[0], L*L))
    plot_latent_space(Xr, y, labels=labels)