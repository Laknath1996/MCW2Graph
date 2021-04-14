"""
objective :
author(s) : Ashwin de Silva
date      : 
"""

import os
from torch_geometric.utils.loop import remove_self_loops
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_add_pool, GCNConv

from src.grnn import *
from tma.utils import load_graph_data

## loading graph data
train_dataset = 'data/subject_1002_Malsha/graph_dataset_1.txt'
test_dataset = 'data/subject_1002_Malsha/graph_dataset_2.txt'

trainData = load_graph_data(train_dataset)

# # define data loaders
train_loader = DataLoader(trainData, batch_size=16, shuffle=True)

# # # forward pass test 
# print(trainData[0])
model = GCRNNGCN(1, 4, 8, 8, 5)
# out = model(trainData[0])
# print(out)

for data in train_loader:
    print(data)
    out = model(data)
    print(out)

# train the network
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = GCRNNMLP(1, 1, 8, 5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# model.train()
# for epoch in range(20):
#     running_loss = 0.0

#     for i, data in enumerate(data_list, 0):
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.cross_entropy(out.unsqueeze(0), data.y)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 10 == 0:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
#             running_loss = 0.0

# print('Finished Traning!')

# # # evaluate model
# model.eval()

# with torch.no_grad():
#     correct = 0
#     num = 0
#     for data in test_loader:
#         _, pred = model(data).max(dim=1)
#         correct += int(pred.eq(data.y).sum().item())
#         num += int(len(data.y))

# acc = correct / num
# print('Accuracy : {:.4f}'.format(acc))

# # save model
# torch.save(model.state_dict(), 'models/model_grnn.pt')

#/////////////// extra /////////////////

# # define the network
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = ChebConv(80, 16, K=2)
#         self.conv2 = ChebConv(16, 32, K=2)
#         self.conv3 = ChebConv(32, 64, K=2)
#         self.fc1 = nn.Linear(64, 100)
#         self.fc2 = nn.Linear(100, 20)
#         self.fc3 = nn.Linear(20, 5)

#     def forward(self, data):
#         batch_size = len(data.y)
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.relu(self.conv2(x, edge_index, edge_weight))
#         x = F.relu(self.conv3(x, edge_index, edge_weight))
#         x = global_add_pool(x, data.batch, size=batch_size)
#         x = F.dropout(x, training=self.training)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return F.log_softmax(x, dim=1)

# # define GRNN
# class MyoNet(torch.nn.Module):
#     def __init__(self):
#         super(MyoNet, self).__init__()
#         self.grnn = GRNN(256)
#         self.gconv1 = ChebConv(80, 40, K=2)
#         self.gconv2 = ChebConv(40, 20, K=2)
#         self.fc = nn.Linear(20, 5)

#     def forward(self, data):
#         batch_size = len(data.y)
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = self.grnn(x, edge_index)
#         x = F.relu(self.gconv1(x, edge_index, edge_weight))
#         x = F.relu(self.gconv2(x, edge_index, edge_weight))
#         x = global_add_pool(x, data.batch, size=batch_size)
#         x = self.fc(x)

#         return F.log_softmax(x, dim=1)
