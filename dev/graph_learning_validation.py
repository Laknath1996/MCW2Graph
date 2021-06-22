#
# Created on Thu May 27 2021 7:36:19 AM
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
# Objective : Test the validity of the graph learning algorithms

# ////// libraries ///// 

# standard 
from matplotlib.pyplot import sci
import numpy as np
import scipy

# internal 
from graph_learning.methods import SmoothSignalGraphLearn
from graph_learning.utils import createWeightedGraph

# ////// body ///// 

# generate a random geometric graph (RGG)
m = 8           #num Nodes
c = 2           #num Coordinates
np.random.seed(0)
Xc = np.random.rand(m, c)

# get the adjacency matrix according to the Gaussian function
sigma = 0.2
W = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        W[i, j] = np.exp(-(np.linalg.norm(Xc[i] - Xc[j], ord=2)**2)/(2*sigma**2))
np.fill_diagonal(W, 0)
W[W < 0.6] = 0
createWeightedGraph(W, semgConfig=False, seed=2)

## normalize laplacian

# degree matrix
D = np.zeros((m, m))
d = np.sum(W, axis=1)
np.fill_diagonal(D, d) 

# laplacian
L = D - W 

# normalized laplacian
L = L / np.linalg.norm(L, ord=2)

# find eigenvalues and eigenvectors
mu, U = np.linalg.eig(L)
idx = np.argsort(mu)
mu = mu[idx]            # sorted eigenvalues 
U = U[..., idx]         # sorted eigenvectors
h_mu = 1/(1 + 10*mu)    # filtered eigenvalues
H_mu = np.zeros((m, m))
np.fill_diagonal(H_mu, h_mu)

## create a smooth signal observations
n = 1000
X = np.zeros((m, n))

for k in range(n):
    # sample a gaussin iid vetcor
    np.random.seed(k+1)
    x_0 = np.random.randn(m, 1)
    # filter
    y = np.matmul(np.matmul(np.matmul(U, H_mu), U.T), x_0)
    # assign
    X[...,k] = y.squeeze()

## learn the graph on smooth signals
gl = SmoothSignalGraphLearn(X=X, 
                            alpha=30, 
                            beta=4, 
                            gamma=0.05, 
                            epsilon=1e-4, 
                            imax=2000,
                            verbosity=False)
W_pred = gl.findGraph()
W_pred[W_pred < 1e-5] = 0
createWeightedGraph(W_pred, semgConfig=False, seed=2)

## metrics
relativeEdgeErrorL2 = np.linalg.norm(W - W_pred, ord=2)
print("Relative Edge Error L2 = ", relativeEdgeErrorL2)




    