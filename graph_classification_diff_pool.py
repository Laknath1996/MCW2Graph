import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ASAPooling, GraphConv, global_mean_pool, JumpingKnowledge
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE

from graph_learning.utils import createGraphDataset
from tma.utils import plot_latent_space


## User Inputs
class_labels = ['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer']
visualizeLatentSpace = False
epochs = 200
lr = 0.001

class ASAP(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden, ratio=0.8, dropout=0):
        super(ASAP, self).__init__()
        self.conv1 = ChebConv(num_features, hidden // 2, K=4, aggr='mean')
        self.conv2 = ChebConv(hidden // 2, hidden, K=4, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            ChebConv(hidden, hidden, K=2, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

## load data
dic1 = sio.loadmat('graph_data/subject_1001_Ashwin/diffusion_train_graph_topologies.mat')
X1 = dic1['W']
y1 = dic1['y'].squeeze()

dic2 = sio.loadmat('graph_data/subject_1001_Ashwin/diffusion_test_graph_topologies.mat')
X2 = dic2['W']
y2 = dic2['y'].squeeze()

if visualizeLatentSpace:
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    print(X.shape)
    print(y.shape)
    Xr = TSNE(n_components=2, perplexity=50).fit_transform(X.reshape(X.shape[0], 8*8))
    plot_latent_space(Xr, y, labels=class_labels)

## shuffle 
X1, y1 = shuffle(X1, y1, random_state=0)
X2, y2 = shuffle(X2, y2, random_state=1)

## create graph dataset
Xg1 = createGraphDataset(X1, y1)
Xg2 = createGraphDataset(X2, y2)

## define model
model = ASAP(8, 5, 4, 32, ratio=0.5, dropout=0.1)

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
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
    print('epoch = {:n}, loss = {:.4f}'.format(epoch, loss.item()))
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

