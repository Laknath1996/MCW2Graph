"""
objective :
author(s) : Ashwin de Silva
date      : 
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_add_pool, GCNConv

print("training...")

## loading the TMA map data
dataset1 = h5py.File(
    '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/src/gsp/subject_2001/trans_map_dataset_1_gsp.h5',
    mode='r')
dataset2 = h5py.File(
    '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/src/gsp/subject_2001/trans_map_dataset_2_gsp.h5',
    mode='r')

tma_data_1 = np.array(dataset1['data'])
labels_1 = np.array(dataset1['label'])
tma_data_2 = np.array(dataset2['data'])
labels_2 = np.array(dataset2['label'])

tma_data = np.concatenate((tma_data_1, tma_data_2), axis=0)
labels = np.concatenate((labels_1, labels_2), axis=0)

## extracting the first order terms of each TMA map.
tma_data_fo = tma_data[:, :8, :]

## adjacency matrix in the COO format
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                            5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7],
                           [0, 1, 2, 7, 0, 1, 2, 3, 7, 0, 1, 2, 3, 1, 2, 3, 4, 5, 3, 4, 5, 6, 3, 4,
                            5, 6, 7, 4, 5, 6, 7, 0, 1, 5, 6, 7]], dtype=torch.long)  # edges
edge_attr = torch.tensor([1.0000, 0.5790, 0.1845, 0.4516, 0.5790, 1.0000, 0.5790, 0.1013, 0.1013,
                          0.1845, 0.5790, 1.0000, 0.4516, 0.1013, 0.4516, 1.0000, 0.4516, 0.1013,
                          0.4516, 1.0000, 0.5790, 0.1845, 0.1013, 0.5790, 1.0000, 0.5790, 0.1013,
                          0.1845, 0.5790, 1.0000, 0.4516, 0.4516, 0.1013, 0.1013, 0.4516, 1.0000],
                         dtype=torch.float)  # edge weights

## creating the graph dataset using the TMA first order terms of each channel
data_list = []

for i in range(tma_data_fo.shape[0]):
    tma = tma_data_fo[i, ...].squeeze()
    x = torch.tensor(tma, dtype=torch.float)
    y = torch.tensor([labels[i]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data_list.append(data)

import random

random.shuffle(data_list)

# define data loaders
train_loader = DataLoader(data_list, batch_size=16, shuffle=True)
# train_loader = DataLoader(data_list[:1890], batch_size=32, shuffle=True)
# test_loader = DataLoader(data_list[1890:], batch_size=32, shuffle=True)


# define the network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(80, 16, K=2)
        self.conv2 = ChebConv(16, 32, K=2)
        self.conv3 = ChebConv(32, 64, K=2)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 5)

    def forward(self, data):
        batch_size = len(data.y)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


# # define GRNN
# class GRNNet(torch.nn.Module):
#     def __init__(self):
#         super(GRNNet, self).__init__()
#         self.conv1 = ChebConv(80, 40, K=2)
#         self.rnn = nn.RNN(40, 20)
#         self.fc = nn.Linear(20, 5)
#
#     def forward(self, data):
#         batch_size = len(data.y)
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = global_add_pool(x, data.batch, size=batch_size)
#         x = self.rnn(x)
#         x = self.fc(x)
#
#         return F.log_softmax(x, dim=1)


# train the network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in range(200):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Traning!')

# # evaluate model
# model.eval()
#
# with torch.no_grad():
#     correct = 0
#     num = 0
#     for data in test_loader:
#         _, pred = model(data).max(dim=1)
#         correct += int(pred.eq(data.y).sum().item())
#         num += int(len(data.y))
#
# acc = correct / num
# print('Accuracy : {:.4f}'.format(acc))

# save model
torch.save(model.state_dict(), '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/src/gsp/subject_2001/model_gnn_rnn.pt')
