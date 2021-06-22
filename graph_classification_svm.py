import scipy.io as sio
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

## load data
dic1 = sio.loadmat('graph_data/subject_1002_Malsha/diffusion_train_graph_topologies.mat')
X1 = dic1['W']
y1 = dic1['y'].squeeze()

dic2 = sio.loadmat('graph_data/subject_1002_Malsha/diffusion_test_graph_topologies.mat')
X2 = dic2['W']
y2 = dic2['y'].squeeze()

## pick upper traingular indices
idx = np.triu_indices(8, k=1)

## shuffle 
X1, y1 = shuffle(X1, y1, random_state=0)
X2, y2 = shuffle(X2, y2, random_state=1)

X1 = X1[:, idx[0], idx[1]]
X2 = X2[:, idx[0], idx[1]]

# ## visualize latent space
# Xr = TSNE(n_components=2, perplexity=60).fit_transform(X)
# colors = ['r', 'b', 'g', 'k', 'm']
# classes = [0, 1, 2, 3, 4]
# for l, c in zip(classes, colors):
#     plt.scatter(Xr[y == l, 0], Xr[y == l, 1], c=c, label=l)
# plt.legend(['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer'])
# plt.show()

## train an SVM
clf = svm.SVC(C=0.1, kernel='rbf', gamma='scale', tol=1e-4)
clf.fit(X1, y1)

## predict
y_pred = clf.predict(X2)

## report
print(classification_report(y2, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))