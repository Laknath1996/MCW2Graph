#
# Created on Wed Apr 14 2021 7:12:38 PM
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
# Objective : Use this script to run developer experiments 
#

#////// libraries //////

# standard 
import os
import numpy as np
import scipy.io as sio
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# internal
from graph_learning.methods import DiffusionGraphLearn, GraphicalLassoGraphLearn
from graph_learning.utils import createWeightedGraph, plotMCWandGraph
from data_handling.utils import create_dataset, plot_recording_same_axis, preprocess, read_from_csv
from tma.utils import plot_latent_space

#////// body //////

## User Inputs
id = "de_silva"
data_path = "data/de_silva/subject_1001_Ashwin"
label = "M_2"
title = "Middle Flexion"
onsets_file = "data/de_silva/subject_1001_Ashwin/onsets.csv"
fs = 200
path = "figures/subject_1002"

## Graph Learn Params
p = 2
beta_1 = 10
beta_2 = 5

## Flags
getOnsets = True
plotGraphs = False
reduceDims = False
combinedReduceDims = False

# //////////////////////

if getOnsets or plotGraphs:
    file = os.path.join(data_path, "{}.csv".format(label))
    data = read_from_csv(id, file)
    data = preprocess(data, fs, lpfreq=1)
    
if getOnsets:
    plot_recording_same_axis(data, fs)

if plotGraphs:
    df = pd.read_csv(onsets_file, delimiter=',')
    onsets = df[[label]].values.squeeze()

    for i in range(len(onsets)):
        start = int((onsets[i])*200)
        end = int((onsets[i]+1.5)*200)
        X = data[:, start:end]
        
        GL = DiffusionGraphLearn(
                X,
                p=p, 
                beta_1=beta_1,
                beta_2=beta_2,
                verbosity=True
        )
        W = GL.findGraphLaplacian()
        W /= np.max(W)
        # W[W<0.4] = 0
        save_path = os.path.join(path, "{}_{}.png".format(label, i))
        plotMCWandGraph(X, fs, W, title=title, save_path=save_path)

if reduceDims:
    df = pd.read_csv(onsets_file, delimiter=',')
    Ws = np.empty((0, 8, 8))
    y = []
    labels = ["M_1", "R_1", "HC_1", "V_1", "PO_1"]
    for i in range(len(labels)):
        file = os.path.join(data_path, "{}.csv".format(labels[i]))
        data = read_from_csv(id, file)
        data = preprocess(data, fs, lpfreq=1)
        onsets = df[[labels[i]]].values.squeeze()
        for j in range(len(onsets)):
            print("Class {} - Graph {}".format(i, j))
            start = int((onsets[j])*200)
            end = int((onsets[j]+1.5)*200)
            X = data[:, start:end]
            
            GL = DiffusionGraphLearn(
                    X,
                    p=p, 
                    beta_1=beta_1,
                    beta_2=beta_2,
                    verbosity=True
            )
            W = GL.findGraphLaplacian()
            W /= np.max(W)
            # W[W<0.4] = 0
            Ws = np.vstack((Ws, np.expand_dims(W, 0)))
            y.append(i)

    Xr = PCA(n_components=2).fit_transform(Ws.reshape(Ws.shape[0], 8*8))
    plot_latent_space(Xr, y, labels=labels)

if combinedReduceDims:
    df = pd.read_csv(onsets_file, delimiter=',')
    Ws = np.empty((0, 8, 8))
    y = []
    labels = ["M", "R", "HC", "V", "PO"]
    for i in range(len(labels)):
        file1 = os.path.join(data_path, "{}_2.csv".format(labels[i]))
        file2 = os.path.join(data_path, "{}_3.csv".format(labels[i]))
        data1 = read_from_csv(id, file1)
        data1 = preprocess(data1, fs, lpfreq=1)
        data2 = read_from_csv(id, file2)
        data2 = preprocess(data2, fs, lpfreq=1)
        onsets1 = df[['{}_2'.format(labels[i])]].values.squeeze()
        onsets2 = df[['{}_3'.format(labels[i])]].values.squeeze()
        onsets = np.concatenate((onsets1, onsets2))
        for j in range(len(onsets)):
            print("Class {} - Graph {}".format(i, j))
            start = int((onsets[j])*200)
            end = int((onsets[j]+1.5)*200)
            if j < 10:
                X = data1[:, start:end]
            else:
                X = data2[:, start:end]
            
            GL = DiffusionGraphLearn(
                    X,
                    p=p, 
                    beta_1=beta_1,
                    beta_2=beta_2,
                    verbosity=True
            )
            W = GL.findGraphLaplacian()
            W /= np.max(W)
            W[W<0.4] = 0
            Ws = np.vstack((Ws, np.expand_dims(W, 0)))
            y.append(i)

    Xr = PCA(n_components=2).fit_transform(Ws.reshape(Ws.shape[0], 8*8))
    plot_latent_space(Xr, y, labels=labels)
