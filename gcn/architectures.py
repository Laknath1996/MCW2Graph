#
# Created on Wed Jun 02 2021 5:36:56 PM
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
# Objective : 
# ////// libraries ///// 
# standard 
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob.glob import global_mean_pool

# internal 

# ////// body ///// 

class VanillaGCN(torch.nn.Module):
    """
    A Simple GCN archiecture
    """
    def __init__(self, inChannels, hiddenChannels, numNodes, numClasses):
        super(VanillaGCN, self).__init__()
        self.hiddenChannels = hiddenChannels
        self.numNodes = numNodes
        self.conv1 = GCNConv(inChannels, 2*hiddenChannels)
        self.conv2 = GCNConv(2*hiddenChannels, hiddenChannels)
        self.fc = nn.Linear(hiddenChannels*numNodes, numClasses)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = x.view(data.num_graphs, self.hiddenChannels*self.numNodes)
        x = self.fc(x)                  
        return F.softmax(x, dim=1)

