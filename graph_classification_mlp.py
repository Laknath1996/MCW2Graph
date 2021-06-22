from keras import Input
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

plotLatentSpace = True
epochs = 60

def nn(shape, no_classes):
    """neural network model"""

    model = Sequential()

    model.add(InputLayer(input_shape=shape))
    
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(8))
    model.add(Activation('relu'))

    model.add(Dense(no_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['categorical_accuracy'])

    return model

## load data
dic1 = sio.loadmat('graph_data/subject_1001_Ashwin/diffusion_train_graph_topologies.mat')
X1 = dic1['W']
y1 = dic1['y'].squeeze()

dic2 = sio.loadmat('graph_data/subject_1001_Ashwin/diffusion_test_graph_topologies.mat')
X2 = dic2['W']
y2 = dic2['y'].squeeze()

## pick upper traingular indices
idx = np.triu_indices(8, k=1)

## shuffle and reshape
X1, y1 = shuffle(X1, y1, random_state=0)
X2, y2 = shuffle(X2, y2, random_state=1)

# X1 = X1[:, idx[0], idx[1]]
# X2 = X2[:, idx[0], idx[1]]

X1 = X1.reshape(X1.shape[0], 64)
X2 = X2.reshape(X2.shape[0], 64)

## scale
# X1 = X1 / np.max(X1, axis=-1, keepdims=True)
# X2 = X2 / np.max(X2, axis=-1, keepdims=True)

## plot latent space
if plotLatentSpace:
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    Xr = TSNE(n_components=2, perplexity=60).fit_transform(X)
    colors = ['r', 'b', 'g', 'k', 'm']
    classes = [0, 1, 2, 3, 4]
    for l, c in zip(classes, colors):
        plt.scatter(Xr[y == l, 0], Xr[y == l, 1], c=c, label=l)
    plt.legend(['Middle_Flexion', 'Ring_Flexion', 'Hand_Closure', 'V_Flexion','Pointer'])
    plt.show()

## to categorical
y1_cat = to_categorical(y1, num_classes=5)

## model
model = nn((64,), 5)

## train
model.fit(
        X1, 
        y1_cat,
        epochs=epochs,
        batch_size=32
)

## validate
y_pred = np.argmax(model.predict(X2), axis=-1)

## report
print(classification_report(y2, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))



