import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import scipy.io as sio
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from torch._C import device, dtype
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, max_pool, avg_pool, avg_pool_neighbor_x, SAGEConv
from torch_geometric.nn.glob.glob import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils.loop import remove_self_loops
from torch_geometric.data import Data, DataLoader

from graph_learning.utils import createGraphDataset

## define the network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = nn.Linear(256, 5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x.float(), edge_index, edge_weight))
        x = x.view(data.num_graphs, 256)
        x = self.fc(x.float())
        x = F.log_softmax(x, dim=1)
        return x

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

## load data
dic1 = sio.loadmat('graph_data/subject_1001_Ashwin/partial_correlation_train_graph_topologies.mat')
X1 = dic1['W']
y1 = dic1['y'].squeeze()

dic2 = sio.loadmat('graph_data/subject_1001_Ashwin/partial_correlation_test_graph_topologies.mat')
X2 = dic2['W']
y2 = dic2['y'].squeeze()

## shuffle 
X1, y1 = shuffle(X1, y1)
X2, y2 = shuffle(X2, y2)

## create graph dataset
Xg1 = createGraphDataset(X1, y1)
Xg2 = createGraphDataset(X2, y2)

## define model
model = Net()

## define the optimizer
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.001, 
                            weight_decay=0)

## define the loader
train_loader = DataLoader(Xg1, batch_size=32, shuffle=True)
test_loader = DataLoader(Xg2, batch_size=1, shuffle=True)

## train
epochs = 300
print('training...')
model.train()
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
    print('epoch = {:n}, loss = {:.4f}'.format(epoch, loss.item()))
print('Finished Traning!')

# validate model
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