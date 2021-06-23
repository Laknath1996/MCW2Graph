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
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score

## internal
from tma.utils import load_tma_data, load_graph_data
from tma.models.nn_models import cnn
from grnn.architectures import GCRNNGCN, GCRNNMLP, GCRNNMLPh
from gcn.architectures import VanillaGCN

#////// user inputs ///////

## methods
tmacnnMethod = True            # 1 : TMA Map + CNN
vanillagcnMethod = False        # 2 : Multi-Channel Window + Vanilla GCN
gcrnnMethod = False             # 3 : GCRNN + MLP
graphLearnGCNMethod = False      # 4 : Multi-Channel WIndow + Graph Learning + GCN

## datasets

if tmacnnMethod:
    train_dataset = 'data/de_silva/subject_1006_Ramith/trans_map_dataset_1.h5'
    test_dataset = 'data/de_silva/subject_1006_Ramith/trans_map_dataset_2.h5'

if gcrnnMethod or vanillagcnMethod:
    train_dataset = 'data/de_silva/subject_1001_Ashwin/graph_dataset_1n.txt'
    test_dataset = 'data/de_silva/subject_1001_Ashwin/graph_dataset_2n.txt'
    model_path = 'data/de_silva/subject_1001_Ashwin/grnn_models/gcrnn_gcn_weights_e200_clip0.5.txt'

if graphLearnGCNMethod:
    train_dataset = 'graph_data/de_silva/subject_1001_Ashwin/correlation_train_graph_topologies.mat'
    test_dataset = 'graph_data/de_silva/subject_1001_Ashwin/correlation_test_graph_topologies.mat'

## training parameters
epochs = 20
saveModel = False
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

    ## select data
    X1 = np.empty((0, X_train.shape[1], X_train.shape[2]))
    X2 = np.empty((0, X_train.shape[1], X_train.shape[2]))
    y1 = np.empty((0, ))
    y2 = np.empty((0, ))
    for i in range(5):
        X_train_i = X_train[y_train==i]
        X_test_i = X_test[y_test==i]
        X1 = np.vstack((X1, X_train_i[:21*7]))
        X1 = np.vstack((X1, X_test_i[:21*7]))
        X2 = np.vstack((X2, X_train_i[21*7:]))
        X2 = np.vstack((X2, X_test_i[21*7:]))
        y1 = np.concatenate((y1, i*np.ones((21*14, ))))
        y2 = np.concatenate((y2, i*np.ones((21*6, ))))

    X_train, y_train = X1, y1
    X_test, y_test = X2, y2

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
    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.array(y_test, dtype='int')
    print(classification_report(y_true, y_pred, target_names=['M', 'R', 'HC', 'V', 'PO']))
    print("Accuracy Score : {:.2f}".format(accuracy_score(y_true, y_pred)))

if vanillagcnMethod:
    ## Multi-Channel Window + Vanilla GCN Method
    
    # define model
    model = VanillaGCN(
        inChannels=80,
        hiddenChannels=20,
        numNodes=8,
        numClasses=5
    )

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001, 
                                betas=(0.9, 0.999),
                                weight_decay=0)

    if trainModel:
        ## train the model
        # load training data 
        print('loading training data...')
        trainData = load_graph_data(train_dataset)
        print("Training Examples = ", len(trainData))

        # shuffle training data
        random.shuffle(trainData)    

        # define the loader
        train_loader = DataLoader(trainData, batch_size=8, shuffle=True)

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
            print('epoch = {:n}, loss = {:.4f}'.format(epoch, loss.item()))
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
    print("Number of Testing Examples = {:n}".format(len(testData)))
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
    print("Accuracy Score : {:.2f}".format(accuracy_score(y_true, y_pred)))

if gcrnnMethod:
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
                    hiddenChannels=16,
                    outChannels=8,
                    numNodes=8,
                    numClasses=5)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001, 
                                betas=(0.9, 0.999),
                                weight_decay=0)

    ## train model                            
    if trainModel:
        # load training data 
        print('loading training data...')
        trainData = load_graph_data(train_dataset)
        print("Number of Training Examples : {:n}".format(len(trainData)))

        # shuffle training data
        random.shuffle(trainData)    

        # define the loader
        train_loader = DataLoader(trainData, batch_size=32, shuffle=True)

        # commence training
        print('training...')
        model.train()
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                out = model(data)
                loss = F.cross_entropy(out, data.y)
                loss.backward()

                # gradient clipping in GCRNN layer
                for count, param in enumerate(model.parameters(), 1):
                    if count == 5:
                        break
                    else:
                        torch.nn.utils.clip_grad_norm_(param, 0.5)

                optimizer.step()
            print('epoch = {:n}, loss = {:.4f}'.format(epoch, loss.item()))
                

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
    print("Testing Examples = ", len(testData))
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
    print("Accuracy Score : {:.2f}".format(accuracy_score(y_true, y_pred)))

if graphLearnGCNMethod:
    ## Multi-Channel Window + Graph Learning + GCN Method
    
    # define model
    model = VanillaGCN(
        inChannels=80,
        hiddenChannels=20,
        numNodes=8,
        numClasses=5
    )

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001, 
                                betas=(0.9, 0.999),
                                weight_decay=0)

    if trainModel:
        ## train the model
        # load training data 
        print('loading training data...')
        trainData = load_graph_data(train_dataset)
        print("Training Examples = ", len(trainData))

        # shuffle training data
        random.shuffle(trainData)    

        # define the loader
        train_loader = DataLoader(trainData, batch_size=8, shuffle=True)

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
            print('epoch = {:n}, loss = {:.4f}'.format(epoch, loss.item()))
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
    print("Number of Testing Examples = {:n}".format(len(testData)))
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
    print("Accuracy Score : {:.2f}".format(accuracy_score(y_true, y_pred)))


    