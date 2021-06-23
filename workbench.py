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
import pandas as pd

# internal
from graph_learning.methods import DiffusionGraphLearn, GraphicalLassoGraphLearn
from graph_learning.utils import createWeightedGraph, plotMCWandGraph
from data_handling.utils import create_dataset, plot_recording_same_axis, preprocess, read_from_csv
from tma.utils import plot_latent_space

#////// body //////

## User Inputs
id = "de_silva"
file = "data/de_silva/subject_1001_Ashwin/R_2.csv"
onsets_file = "data/de_silva/subject_1001_Ashwin/onsets.csv"
fs = 200
path = "figures/subject_1001"

data = read_from_csv(id, file)
data = preprocess(data, fs, lpfreq=1)
# plot_recording_same_axis(data, fs)

df = pd.read_csv(onsets_file, delimiter=',')
onsets = df[['R_2']].values.squeeze()

p = 5
beta_1 = 10
beta_2 = 0.1

for i in range(len(onsets)):
    start = int((onsets[i])*200)
    end = int((onsets[i]+1)*200)
    X = data[:, start:end]
    
    GL = DiffusionGraphLearn(
            X,
            p=p, 
            beta_1=beta_1,
            beta_2=beta_2,
            verbosity=True
    )
    W = GL.findGraphLaplacian()
    save_path = os.path.join(path, "R_1_{}.png".format(i))
    plotMCWandGraph(X, fs, W, title="Ring Flexion", save_path=save_path)




