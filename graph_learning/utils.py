#
# Created on Wed May 26 2021 8:52:56 PM
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
# Objective : Contains functions for plotting a network given its Adjacency Matrix

# ////// libraries ///// 
# standard 
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils.loop import remove_self_loops
from torch_geometric.data import Data

# ////// body ///// 
def getEllipticalCoordinates():
    """Obtain the node coordinates

    Returns
    -------
    pos : dict
        A dictionary including the 2D coordinates for each node
    """
    pos = { 1: (-2*2**0.5, -1.5*2**0.5),
            2: (-4, 0),
            3: (-2*2**0.5, 1.5*2**0.5),
            4: (0, 3),
            5: (2*2**0.5, 1.5*2**0.5),
            6: (4,0),
            7: (2*2**0.5, -1.5*2**0.5),
            8: (0, -3)
    }
    return pos

def createWeightedGraphBeta(W, semgConfig=True, seed=1, **kwargs):
    """
    create a network topolgy given its Adjacency matrix
    (Depricated)
    """
    m = W.shape[0]
    G = nx.Graph()
    for i in range(0, m):
        for j in range(0, m):
            G.add_edge(i, j, weight=W[i, j])

    if semgConfig:
        pos = getEllipticalCoordinates()
    else:
        pos = nx.random_layout(G, seed=seed)

    weights = list(nx.get_edge_attributes(G,'weight').values())
    line_widths = (weights - min(weights))/(max(weights)-min(weights))*5
    edge_colors = (weights - min(weights))/(max(weights)-min(weights))

    # plot the graph
    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111)
    nx.draw(G, pos, 
        # width=list(line_widths),
        with_labels=True,
        arrows=True,
        arrowstyle='->',
        arrowsieze=20,
        edge_color=list(edge_colors),
        edge_cmap=plt.cm.Blues,
        edge_vmin=min(edge_colors),
        edge_vmax=max(edge_colors),
        node_color='lightgreen',
        ax=ax)
        
    ## add a colorbar
    norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    sm.set_array([])
    # plt.axis('on') # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.colorbar(sm)
    if kwargs['title'] is not None:
        plt.title(kwargs['title'])
    plt.show()

def plotMultiChannelSignals(X, Fs):
    """Plots the given multi-channel window

    Parameters
    ----------
    X : array
        Multi-channel window of shape (numChannels, numSamples)
    Fs : float
        Sampling freqency in Hertz
    """
    L = X.shape[0]
    N = X.shape[1]
    t = np.arange(0, N/Fs, 1/Fs)
    fig, axs = plt.subplots(L)
    for i in range(L):
        axs[i].plot(t, X[i, ...]) 
    plt.show()

def unvectorize(y):
    """
    Form a symmetrical matrix $Y \in \mathbb{R}^{m \times m}$ where
    diag(Y) = 0 given an array $y \in \mathbb{m(m-1)/2}$
    """
    L = 8
    Y = np.zeros((L, L))
    i, j = np.triu_indices(L, 1)
    Y[i, j] = y
    Y[j, i] = y
    return Y

def vectorize(Y):
    """
    Form an array $y \in \mathbb{L(L-1)/2}$ that only contains the upper 
    traingle indices of the input matrix $Y \in \mathbb{R}^{L \times L}$
    """
    L = Y.shape[0]
    idx = np.triu_indices(L, 1)
    return Y[idx]

def createGraphDataset(X, y):
    """
    Create the graph dataset for the multi-channel signals
    """
    Xg = []
    for i in range(X.shape[0]):
        W = X[i]
        A = coo_matrix(W)
        edge_index, edge_attr = from_scipy_sparse_matrix(A)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        x = torch.tensor(np.identity(8), dtype=torch.float)
        yg = torch.tensor([y[i]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=yg)
        Xg.append(data)
    return Xg

def createMultiGraphDataset(X1, X2, y):
    """
    Create the graph dataset for the multi-channel signals
    """
    Xg = []
    for k in range(X1.shape[0]):
        W1 = X1[k]
        W2 = X2[k]
        
        edge_index = np.empty((2, 0))
        edge_attr = np.empty((0, 2))
        for i in range(8):
            for j in range(8):
                if W1[i, j]==0 and W2[i, j]==0:
                    continue
                else:    
                    ei = np.array([i, j]).reshape(2, 1)
                    ea = np.array([W1[i, j], W2[i, j]]).reshape(1, 2)
                    edge_index = np.hstack((edge_index, ei))
                    edge_attr = np.vstack((edge_attr, ea))

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr.squeeze())
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)        
        x = torch.tensor(np.identity(8), dtype=torch.float)
        yg = torch.tensor([y[k]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=yg)
        Xg.append(data)
    return Xg    

def createWeightedGraph(W, remove_self_loops=True, title=None):
    """Plots the corresponding weighted undirecte or directed graph topology
    of the given adjacency matrix.

    Parameters
    ----------
    W : numpy array
        Adjacency matrix with shape (L, L). Currently, this function only
        supports L=8
    remove_self_loops : bool, optional
        if True, remove self loops in the graph, by default True
    title : str, optional
        Title of the plotted graph topology, by default None
    """
    if np.allclose(W, W.T):
        graphIsDirected = False
    else:
        graphIsDirected = True
    if remove_self_loops:
        np.fill_diagonal(W, 0)
    m = W.shape[0]
    pos = getEllipticalCoordinates()
    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111)

    if graphIsDirected:
        ## plot directed weighted graph
        G = nx.DiGraph()
        for i in range(0, m):
            for j in range(0, m):
                G.add_edge(i+1, j+1, weight=W[i, j])

        weights = list(nx.get_edge_attributes(G,'weight').values())
        weightColors = (weights - min(weights))/(max(weights)-min(weights))

        # plot the graph
        nx.draw_networkx_nodes(G, pos, 
                                with_labels=True, 
                                node_color='lightgreen', 
                                ax=ax)
        nx.draw_networkx_edges(G, pos, 
                                arrows=True, arrowsize=20, arrowstyle='-|>',
                                width=2,
                                edge_color=weightColors,
                                edge_cmap=plt.cm.Blues,
                                edge_vmin=min(weightColors),
                                edge_vmax=max(weightColors),
                                connectionstyle='arc3, rad = 0.1',
                                ax=ax)
        nx.draw_networkx_labels(G, pos)
            
        # add colorbar
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        weightMap = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
        plt.axis('on')
        plt.colorbar(weightMap)
        if title is not None:
            plt.title(title)
        plt.show()

    else:
        ## plot unidrected weighted graph
        G = nx.Graph()
        for i in range(0, m):
            for j in range(0, m):
                G.add_edge(i+1, j+1, weight=W[i, j])

        weights = list(nx.get_edge_attributes(G,'weight').values())
        weightWidths = (weights - min(weights))/(max(weights)-min(weights))*5
        weightColors = (weights - min(weights))/(max(weights)-min(weights))

        # plot the graph
        nx.draw(G, pos, 
                    width=weightWidths,
                    with_labels=True,
                    edge_color=weightColors,
                    edge_cmap=plt.cm.Blues,
                    edge_vmin=min(weightColors),
                    edge_vmax=max(weightColors),
                    node_color='lightgreen',
                    ax=ax)
            
        ## add a colorbar
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        weightMap = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
        plt.axis('on')
        plt.colorbar(weightMap)
        if title is not None:
            plt.title(title)
        plt.show()  

def rescaleMCW(X):
    """
    rescale a multi-channel window between [0, 1]
    """
    channel_max = np.max(X, axis=1, keepdims=True)
    return X / channel_max





