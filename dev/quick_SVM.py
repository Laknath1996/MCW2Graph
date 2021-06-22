import scipy.io as sio
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

## load data
dic = sio.loadmat('/Users/ashwin/Current Work/GRNNmyo/smoothautoregressgl_outputs.mat')
X = dic['W']
y = dic['y'].squeeze()

## shuffle 
X, y = shuffle(X, y)

# ## visualize latent space
# Xr = TSNE(n_components=2, perplexity=60).fit_transform(X)
# colors = ['r', 'b', 'g', 'k', 'm']
# classes = [0, 1, 2, 3, 4]
# for l, c in zip(classes, colors):
#     plt.scatter(Xr[y == l, 0], Xr[y == l, 1], c=c, label=l)
# plt.legend(['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer'])
# plt.show()

## split
X_train, y_train = X[:600], y[:600]
X_test, y_test = X[600:], y[600:]

## train an SVM
clf = svm.SVC()
clf.fit(X_train, y_train)

## predict
y_pred = clf.predict(X_test)

## report
print(classification_report(y_test, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))