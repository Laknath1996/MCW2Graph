"""
objective :
author(s) : Ashwin de Silva
date      : 
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from threading import Lock
import myo
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_add_pool

from gsp.functions import EmgGraphLearn

# adjacency matrix in the COO format
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                            5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7],
                           [0, 1, 2, 7, 0, 1, 2, 3, 7, 0, 1, 2, 3, 1, 2, 3, 4, 5, 3, 4, 5, 6, 3, 4,
                            5, 6, 7, 4, 5, 6, 7, 0, 1, 5, 6, 7]], dtype=torch.long)  # edges
edge_attr = torch.tensor([1.0000, 0.5790, 0.1845, 0.4516, 0.5790, 1.0000, 0.5790, 0.1013, 0.1013,
                          0.1845, 0.5790, 1.0000, 0.4516, 0.1013, 0.4516, 1.0000, 0.4516, 0.1013,
                          0.4516, 1.0000, 0.5790, 0.1845, 0.1013, 0.5790, 1.0000, 0.5790, 0.1013,
                          0.1845, 0.5790, 1.0000, 0.4516, 0.4516, 0.1013, 0.1013, 0.4516, 1.0000],
                         dtype=torch.float)  # edge weights


# define the network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(80, 40, K=2)
        self.conv2 = ChebConv(40, 20, K=2)
        self.fc = nn.Linear(20, 5)

    def forward(self, data):
        batch_size = len(data.y)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


class EmgCollectorPrediction(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = []  # deque(maxlen=n)
        self.emg_stream = []

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))
            self.emg_stream.append((event.timestamp, event.emg))


class Predict(object):
    """
    onset detection and plotting
    """

    def __init__(self, listener, emgLearn, gesture_dict, model, thresh):
        # connection with the device
        self.n = listener.n
        self.listener = listener
        self.emgLearn = emgLearn

        # prediction properties
        self.prediction = 'No Gesture'
        self.thresh = thresh
        self.gesture_dict = dict([(value, key) for key, value in gesture_dict.items()])
        self.model = model
        self.start = time.time()

    def main(self):
        prev_pred = 6
        start_time = 0
        prev_O = 0
        start = True

        while True:
            emg_data = self.listener.get_emg_data()
            emg_data = np.array([x[1] for x in emg_data]).T

            if emg_data.shape[0] == 0:
                continue

            if emg_data.shape[1] >= self.n:
                if start:
                    print("statring..")
                    start = False

                emg_data = self.emgLearn.filter_signals(emg_data)

                obs = emg_data[:, -int(self.emgLearn.obs_dur*self.emgLearn.fs):]  # M = 80 (obs_dur = 0.4)

                O = self.emgLearn.non_linear_transform(obs)

                diff = np.linalg.norm(prev_O - O, ord=2)

                if diff > self.thresh and time.time() - start_time >= 3:
                    start_time = time.time()

                    # logic
                    if prev_pred == 0:
                        pred = 5
                    elif prev_pred == 1:
                        pred = 5
                    elif prev_pred == 2:
                        pred = 5
                    elif prev_pred == 3:
                        pred = 5
                    elif prev_pred == 4:
                        pred = 5
                    else:
                        ### GNN architecture
                        x = torch.tensor(O[:8, :], dtype=torch.float)
                        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=[0])

                        # define data loaders
                        test_loader = DataLoader([data], batch_size=1, shuffle=True)

                        with torch.no_grad():
                            for data in test_loader:
                                _, pred = self.model(data).max(dim=1)
                        pred = pred.numpy()[0]

                    self.prediction = self.gesture_dict[pred]
                    print(self.prediction)

                    prev_pred = pred

                prev_O = O
                time.sleep(0.100)  # k value between two adjacent TMA maps


def main():
    """
    execute
    """

    # load the model
    model_path = '/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/src/gsp/subject_2001/model.pt'
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    myo.init('/Users/ashwin/Current Work/Real-Time Hand Gesture Recognition with TMA Maps/sdk/myo.framework/myo')
    hub = myo.Hub()

    el = EmgGraphLearn(fs=200,
                       no_channels=8,
                       obs_dur=0.400)
    listener = EmgCollectorPrediction(n=512)

    gesture_dict = {
        'Middle_Flexion': 0,
        'Ring_Flexion': 1,
        'Hand_Closure': 2,
        'V_Flexion': 3,
        'Pointer': 4,
        'Neutral': 5,
        'No_Gesture': 6
    }

    live = Predict(thresh=4,
                   listener=listener,
                   emgLearn=el,
                   gesture_dict=gesture_dict,
                   model=model)

    with hub.run_in_background(listener.on_event):
        live.main()

    print("Closing...")


if __name__ == '__main__':
    main()
