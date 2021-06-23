import os
from numpy import random
from torch.nn.modules import normalization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import scipy.io as sio
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.glob.glob import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils.loop import remove_self_loops
from torch_geometric.data import Data, DataLoader

from graph_learning.utils import createGraphDataset, createMultiGraphDataset
from tma.utils import plot_latent_space

## User Inputs
class_labels = ['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer']
visualizeLatentSpace = False
epochs = 100
lr = 0.001

## define the network (simple graph)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(8, 16, K=4)
        self.conv2 = ChebConv(16, 32, K=4)
        self.conv3 = ChebConv(32, 64, K=4)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.relu(self.conv1(x.float(), edge_index, edge_weight.float()))
        x = F.relu(self.conv2(x.float(), edge_index, edge_weight.float()))
        x = F.relu(self.conv3(x.float(), edge_index, edge_weight.float()))

        x = x.view(data.num_graphs, -1)
        
        x = F.relu(self.fc1(x.float()))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1 , training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

## load data
dic1 = sio.loadmat('graph_data/de_silva/subject_1007_Jathu/diffusion_train_graph_topologies.mat')
X1 = dic1['W']
y1 = dic1['y'].squeeze()

dic2 = sio.loadmat('graph_data/de_silva/subject_1007_Jathu/diffusion_test_graph_topologies.mat')
X2 = dic2['W']
y2 = dic2['y'].squeeze()

## select data
X_train = np.empty((0, 8, 8))
X_test = np.empty((0, 8, 8))
y_train = np.empty((0, ))
y_test = np.empty((0, ))
for i in range(5):
    X1i = X1[y1==i]
    X2i = X2[y2==i]
    X_train = np.vstack((X_train, X1i[:21*7]))
    X_train = np.vstack((X_train, X2i[:21*7]))
    X_test = np.vstack((X_test, X1i[21*7:]))
    X_test = np.vstack((X_test, X2i[21*7:]))
    y_train = np.concatenate((y_train, i*np.ones((21*14, ))))
    y_test = np.concatenate((y_test, i*np.ones((21*6, ))))

X1, y1 = X_train, y_train
X2, y2 = X_test, y_test

if visualizeLatentSpace:
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    print(X.shape)
    print(y.shape)
    Xr = TSNE(n_components=2, perplexity=50).fit_transform(X.reshape(X.shape[0], 8*8))
    plot_latent_space(Xr, y, labels=class_labels)

## shuffle 
X1, y1 = shuffle(X1, y1, random_state=0)
X2, y2 = shuffle(X2, y2, random_state=0)

## create graph dataset
Xg1 = createGraphDataset(X1, y1)
Xg2 = createGraphDataset(X2, y2)

## define model
model = Net()

## define the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=lr)

## define the loader
train_loader = DataLoader(Xg1, batch_size=32, shuffle=True)
test_loader = DataLoader(Xg2, batch_size=1, shuffle=True)

## train
print('training...')
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += out.shape[0] * loss.item()
    print('epoch = {:n}, loss = {:.4f}'.format(epoch, epoch_loss / len(Xg1)))
print('Finished Traning!')

## validate model
print('validating model...')
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for data in test_loader:
        _, pred = model(data).max(dim=1)
        y_pred.append(pred.numpy()[0])
        y_true.append(data.y.numpy()[0])
print(classification_report(y_true, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))
print("Accuracy Score : {:.2f}".format(accuracy_score(y_true, y_pred)))

# print('validating model...')
# model.eval()
# y_pred = []
# y_true = []
# with torch.no_grad():
#     for data1, data2 in zip(test_loader_1, test_loader_2):
#         _, pred = model(data1, data2).max(dim=1)
#         y_pred.append(pred.numpy()[0])
#         y_true.append(data1.y.numpy()[0])
# print(classification_report(y_true, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))
# print("Accuracy Score : {:.2f}".format(accuracy_score(y_true, y_pred)))

# ## visualize latent space
# y_true = np.array(y_true)
# graph_embeddings[graph_embeddings < 0] = 0
# Xr = TSNE(n_components=2, perplexity=50).fit_transform(graph_embeddings)
# colors = ['r', 'b', 'g', 'k', 'm']
# classes = [0, 1, 2, 3, 4]
# for l, c in zip(classes, colors):
#     x1 = Xr[y_true == l, 0]
#     x2 = Xr[y_true == l, 1]
#     plt.scatter(x1, x2, c=c, label=l)
# plt.legend(['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer'])
# plt.show()