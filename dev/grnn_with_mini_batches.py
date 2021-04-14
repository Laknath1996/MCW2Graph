#
# Created on Mon Jan 04 2021 12:14:14 PM
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
# Objective : Implementing GRNN with mini-batches
#

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob.glob import global_mean_pool

class GraphConvRNN(torch.nn.Module):
    """
    h_t = \sigma( A(S) * x_t + B(S) * h_{t-1} )
    """
    def __init__(self, inChannels, outChannels):
        super(GraphConvRNN, self).__init__()
        self.A = GCNConv(inChannels, outChannels)
        self.B = GCNConv(outChannels, outChannels)

    def forward(self, X, h0):
        """
        forward propagation
        """
        x, edge_index, edge_weight = X.x, X.edge_index, X.edge_attr
        N = x.shape[0] # number of nodes
        T = x.shape[1] # sequence length 
        H = torch.empty(0)  # hidden state sequence

        h = h0
        for t in range(T):
            h = self.A(x[..., t].view(N, 1), edge_index, edge_weight) + self.B(h, edge_index, edge_weight)
            h = torch.sigmoid(h)
            H = torch.cat([H, h], 1)

        return H

class GCRNNMLP(torch.nn.Module):
    def __init__(self, inChannels, hiddenChannels, numNodes, numClasses, T=80):
        super(GCRNNMLP, self).__init__()
        self.numNodes = numNodes
        self.hiddenChannels = hiddenChannels
        self.gcrnn = GraphConvRNN(inChannels, hiddenChannels)
        self.fc = nn.Linear(hiddenChannels*T, numClasses)

    def forward(self, X):
        batchSize = len(X.y)
        h0 = self.init_hidden(batchSize)
        H = self.gcrnn(X, h0)
        H = global_mean_pool(H, X.batch, batchSize)
        y_hat = self.fc(H)
        return torch.softmax(y_hat, dim=1)

    def init_hidden(self,batchSize):
        return torch.zeros(self.numNodes*batchSize, self.hiddenChannels)