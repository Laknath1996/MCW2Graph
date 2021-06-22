#
# Created on Tue Jun 22 2021 8:33:52 AM
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
# Objective : Some data handling scripts for the available datasets

# ////// libraries ///// 

# standard 
import os
import numpy as np
import h5py
import pandas as pd
from scipy.signal import decimate, butter, sosfilt
import matplotlib.pyplot as plt

# ////// body ///// 

def read_from_csv(dataset_id, file):
    """Reads the multi-channl data from .csv files

    Parameters
    ----------
    dataset_id : str
        A unique identifier allocated for each mulit-channel dataset
    file : str
        The path to the .csv file

    Returns
    -------
    data : numpy array
        The multi-channel data with the shape (num_channels, num_samples)
    """
    if dataset_id == "kushaba":
        df = pd.read_csv(file, delimiter=',')
        data = df.values
        data = data.T
    if dataset_id == "de_silva":
        df = pd.read_csv(file, delimiter='\t')
        data = df.values
        data = data[:, 1:data.shape[1]]
        data = data.T
    return data

def downsample(data, fs, factor):
    """Downsample a mulit-channel lsignal by the given factor

    Parameters
    ----------
    data : numpy array 
        Multi-channl signals of the shape (num_channels, num_samples)
    fs : float
        sampling freqeuncy of the multi-channel signals
    factor : list
        decimating factors as a list. For example if the decimating factor is 20, then
        factor = [4, 5]

    Returns
    -------
    data : numpy array
        decimated multi-channel signals
    
    fs_new : float
        new sampling frequency after decimation
    """
    if len(factor) > 1:
        tot_factor = 1
        for i in range(len(factor)):
            data = decimate(data, factor[i], axis=-1)
            tot_factor *= factor[i]
        fs_new = fs/tot_factor
    else:
        data = decimate(data, factor[0], axis=-1)
        fs_new = fs/factor[0]
    return data, fs_new

def preprocess(data, fs, lpfreq=1):
    """Preprocess the mulit-channel signals by first rectifying and then low pass filtering
    the signals

    Parameters
    ----------
    data : numpy array
        Multi-channel signal data of shape (num_channels, num_samples)
    fs : float
        Sampling frequency
    lpfreq : int, optional
        Low pass cut off frequency in Hz, by default 1

    Returns
    -------
    filt_data : numpy array
        Preprocessed multi-channel signals
    """
    # rectify the signals
    data = abs(data)

    # low pass filtering
    lpf = butter(2, lpfreq, 'lowpass', analog=False, fs=fs, output='sos') 
    filt_data = sosfilt(lpf, data)

    return filt_data
    
def plot_recording(data, fs):
    """Plots the multi-channl signals

    Parameters
    ----------
    data : numpy array
        Multi-channl signals of shape (num_channels, num_samples)
    fs : float
        sampling frequency
    """
    fig = plt.figure()
    num_channels = data.shape[0]
    num_samples = data.shape[1]
    axes = [fig.add_subplot('%i1' % num_channels + str(i)) for i in range(0, num_channels)]
    t = np.arange(0, num_samples, 1) / fs
    for i in range(num_channels):
        axes[i].plot(t, data[i])
        i += 1
    plt.show()

def create_dataset(dataset_id, subject_id, labels, num_channels=8, num_samples=80):
    """Create the datasent of multi-channel windows and their respecitve labels

    Parameters
    ----------
    dataset_id : str
        A unique identifier allocated for each mulit-channel dataset
    subject_id : str
        A unique identifier allocated for each subject
    labels : list
        An array including the file names of the .csv files containing the multi-channel signals
    num_channels : int, optional
        Number of channels, by default 8
    num_samples : int, optional
        Number of samples per each multi-channel window, by default 80

    Returns
    -------
    X : numpy array
        An array of the shape (num_windows, num_channels, num_samples) that includes the 
        multi-channel windows
    y : numpy array
        An array of the size (num_windows, ) that includes the labels of each multi-channel
        window
    """
    if dataset_id == "kushaba":
        dataset_path = "data/Delsys_8Chans_15Classes"
        subject_folder = "S{}-Delsys-15Class".format(subject_id)
        X = np.empty((0, num_channels, num_samples))
        y = np.empty((0, ))
        for k in range(len(labels)):
            label = labels[k]
            for i in range(1, 4):
                csv_file = label + "{}.csv".format(i)
                file = os.path.join(dataset_path, subject_folder, csv_file)
                data = read_from_csv(dataset_id, file)
                data, fs = downsample(data, 4000, [4, 5])
                data = preprocess(data, fs)
                for j in range(12):
                    X = np.vstack((X, np.expand_dims(data[:, j*80:(j+1)*80], 0)))
                y = np.concatenate((y, k*np.ones((12, ))))
    return X, y                                                

