#
# Created on Wed Apr 14 2021 7:23:57 PM
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
# Objective : Define the Graph Recurrent Neual Network Architectures


# ////// libraries ///// 

# standard 
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.glob.glob import global_mean_pool

# ////// body ///// 

class GraphConvRNN(torch.nn.Module):
    """
    h_t = \sigma( A(S) * x_t + B(S) * h_{t-1} )
    """
    def __init__(self, inChannels, outChannels):
        super(GraphConvRNN, self).__init__()
        # self.A = GCNConv(inChannels, outChannels)
        # self.B = GCNConv(outChannels, outChannels)
        self.A = ChebConv(inChannels, outChannels, K=4)
        self.B = ChebConv(outChannels, outChannels, K=4)

    def forward(self, X, h0):
        """
        forward propagation
        """
        x, edge_index, edge_weight = X.x, X.edge_index, X.edge_attr 
        N = x.shape[0]              # number of nodes
        T = x.shape[1]              # sequence length 
        H = torch.empty(0)          # hidden state sequence

        # recurrent forward computation : h_t = \sigma( A(S) * x_t + B(S) * h_{t-1}
        h = h0
        for t in range(T): 
            h = self.A(x.select(-1, t).view(N, 1).float(), edge_index, edge_weight.float()) + self.B(h.float(), edge_index, edge_weight.float())
            h = torch.sigmoid(h)
            H = torch.cat([H, h], 1) # get the [h_t, h_{t-1},..., h_1] sequence
        
        return H


class GCRNNMLP(torch.nn.Module):
    """
    h_t = \sigma( A(S) * x_t + B(S) * h_{t-1} )
    y_hat = MLP(flatten([h_t, h_{t-1}, ..., h_1])
    """
    def __init__(self, inChannels, hiddenChannels, numNodes, numClasses, T=80):
        super(GCRNNMLP, self).__init__()
        self.numNodes = numNodes
        self.hiddenChannels = hiddenChannels
        self.gcrnn = GraphConvRNN(inChannels, hiddenChannels)
        self.fc = nn.Linear(hiddenChannels*T*numNodes, numClasses)

    def forward(self, X):
        h0 = self.init_hidden()             # initialize h_0
        H = self.gcrnn(X, h0)               # GCRNN layer
        H = H.view(-1)                      # flatten 
        y_hat = self.fc(H)                  # MLP layer       
        return torch.softmax(y_hat, dim=0)  

    def init_hidden(self):
        return torch.zeros(self.numNodes, self.hiddenChannels)


class GCRNNMLPh(torch.nn.Module):
    """
    h_t = \sigma( A(S) * x_t + B(S) * h_{t-1} )
    y_hat = MLP(flatten(h_t))
    """
    def __init__(self, inChannels, hiddenChannels, numNodes, numClasses):
        super(GCRNNMLPh, self).__init__()
        self.numNodes = numNodes
        self.hiddenChannels = hiddenChannels
        self.gcrnn = GraphConvRNN(inChannels, hiddenChannels)
        self.fc = nn.Linear(hiddenChannels*numNodes, numClasses)

    def forward(self, X):
        h0 = self.init_hidden()                                     # initialize h_0
        H = self.gcrnn(X, h0)                                       # GCRNN layer
        h_t = H.narrow(1,-self.hiddenChannels, self.hiddenChannels) # obtain h_t from the GCRNN output sequence
        h_t = h_t.reshape(-1)                                       # flatten
        y_hat = self.fc(h_t)                                        # MLP layer
        return torch.softmax(y_hat, dim=0)

    def init_hidden(self):
        return torch.zeros(self.numNodes, self.hiddenChannels)

class GCRNNGCN(torch.nn.Module):
    """
    h_t = \sigma( A(S) * x_t + B(S) * h_{t-1} )
    z_t = \phi(h_t) = ReLU (C(S) * h_t)
    y_t = MLP(z_t)
    """
    def __init__(self, inChannels, hiddenChannels, outChannels, numNodes, numClasses):
        super(GCRNNGCN, self).__init__()
        self.numNodes = numNodes
        self.hiddenChannels = hiddenChannels
        self.outChannels = outChannels
        self.gcrnn = GraphConvRNN(inChannels, hiddenChannels)
        self.phi1 = GCNConv(hiddenChannels, outChannels)
        self.phi2 = GCNConv(outChannels, int(outChannels/2))
        self.fc = nn.Linear(int(outChannels*numNodes/2), numClasses)

    def forward(self, X):
        batchSize, edge_index, edge_weight = len(X.y), X.edge_index, X.edge_attr
        h0 = self.init_hidden(batchSize)                                # initialize h_0
        H = self.gcrnn(X, h0)                                           # GCRNN layer
        h_t = H.narrow(1,-self.hiddenChannels, self.hiddenChannels)     # obtain h_t from the GCRNN output sequence
        z_t = self.phi1(h_t.float(), edge_index, edge_weight.float())                   # Graph Conv layer
        z_t = F.relu(z_t)                                               # ReLU
        z_t = self.phi2(z_t.float(), edge_index, edge_weight.float())                   # Graph Conv layer
        z_t = F.relu(z_t)                                               # ReLU
        z_t = z_t.view(batchSize, self.numNodes, int(self.outChannels/2))    # Rearrange
        z_t = z_t.view(batchSize,  int(self.numNodes * self.outChannels/2))  # flatten
        y_hat = self.fc(z_t)                                            # MLP layer
        return F.log_softmax(y_hat, dim=1)

    def init_hidden(self, batchSize):
        return torch.zeros(self.numNodes*batchSize, self.hiddenChannels) # zero initialization 
        # return torch.nn.init.xavier_normal_(torch.empty(self.numNodes*batchSize, self.hiddenChannels), gain=1.0) # Xavier intialization