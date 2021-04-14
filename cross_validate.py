#
# Created on Wed Apr 14 2021 7:25:25 PM
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
# Objective : train, save, and validate the models 
#

#////// libraries ///////

## standard
import os
from pickle import load
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import h5py
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

## internal
from tma.utils import load_tma_data, load_graph_data
from tma.models.nn_models import cnn
from grnn.architectures import GCRNNGCN, GCRNNMLP, GCRNNMLPh

#////// user inputs ///////

## methods
tmacnnMethod = False  # 1 : TMA Map + CNN
gcrnnmlpMethod = True  # 2 : GCRNN + MLP


## datasets

if tmacnnMethod:
    train_dataset = 'data/subject_1001_Ashwin/trans_map_dataset_2.h5'
    test_dataset = 'data/subject_1001_Ashwin/trans_map_dataset_3.h5'

if gcrnnmlpMethod:
    train_dataset = 'data/subject_1001_Ashwin/graph_dataset_1.txt'
    test_dataset = 'data/subject_1001_Ashwin/graph_dataset_2.txt'
    model_path = 'data/subject_1001_Ashwin/grnn_models/gcrnn_gcn_weights.txt'

## training parameters
epochs = 200
saveModel = True
trainModel = True

#////// body ///////

if tmacnnMethod:
    ## TMA Map + CNN method

    # load training data
    X_train, y_train = load_tma_data(train_dataset)
    X_train, y_train = shuffle(X_train, y_train)
    
    # load testing data
    X_test, y_test = load_tma_data(test_dataset)
    X_test, y_test = shuffle(X_test, y_test)

    # train the model 
    num_classes = len(np.unique(y_train))
    if num_classes > 2:
        y_train = to_categorical(y_train, num_classes=num_classes)
    else:
        y_train = y_train

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    model = cnn((X_train.shape[1], X_train.shape[2], 1), num_classes)
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=16,
              verbose=1)

    # validate
    y_pred = model.predict(X_test.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.array(y_test, dtype='int')
    print(classification_report(y_true, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))

if gcrnnmlpMethod:
    ## GCRNN + MLP Method

    # define the model
    print('defining model and optimizer...')
    # model = GCRNNMLP(inChannels=1, 
    #                 hiddenChannels=1, 
    #                 numNodes=8,
    #                 numClasses=5)

    # model = GCRNNMLPh(inChannels=1, 
    #                 hiddenChannels=4, 
    #                 numNodes=8,
    #                 numClasses=5)

    model = GCRNNGCN(inChannels=1, 
                    hiddenChannels=4,
                    outChannels=8,
                    numNodes=8,
                    numClasses=5)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001, 
                                weight_decay=5e-4)

    ## train model                            
    if trainModel:
        # load training data 
        print('loading training data...')
        trainData = load_graph_data(train_dataset)
        train_loader = DataLoader(trainData, batch_size=16, shuffle=True)

        # shuffle training data
        # random.shuffle(trainData)    

        # commence training
        print('training...')
        model.train()
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                out = model(data)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], "
                        f"Step [{i + 1}/{len(trainData)}], "
                        f"Loss: {loss.item():.4f}"
                    )

        print('Finished Traning!')

        if saveModel:
            # save the trained model
            print('saving model...')
            torch.save(model.state_dict(), model_path)

    else:
        # load the specifiec model
        print('loading model...')
        model.load_state_dict(torch.load(model_path))

    # load testing data
    print('loading testing data...')
    testData = load_graph_data(test_dataset)
    test_loader = DataLoader(testData, batch_size=1, shuffle=True)

    # validate model
    print('validating model...')
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in test_loader:
            _, pred = model(data).max(dim=1)
            y_pred.append(pred)
            y_true.append(data.y)
    print(classification_report(y_true, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))



    