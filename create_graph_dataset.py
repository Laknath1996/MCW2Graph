#////// libraries ///////

# standard
import pickle
import torch
from torch_geometric.utils.loop import remove_self_loops
from torch_geometric.data import Data

# internal 
from tma.utils import load_tma_data

#////// code ///////

## inputs
train_dataset = 'data/subject_1001_Ashwin/trans_map_dataset_2.h5'
test_dataset = 'data/subject_1001_Ashwin/trans_map_dataset_3.h5'

graph_train_dataset = 'data/subject_1001_Ashwin/graph_dataset_1.txt'
graph_test_dataset = 'data/subject_1001_Ashwin/graph_dataset_2.txt'

## loading the TMA map data
X_train, y_train = load_tma_data(train_dataset)
X_test, y_test = load_tma_data(test_dataset)

## extracting the first order terms of each TMA map.
X_train = X_train[:, :8, :]
X_test = X_test[:, :8, :]

## adjacency matrix in the COO format
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                            5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7],
                           [0, 1, 2, 7, 0, 1, 2, 3, 7, 0, 1, 2, 3, 1, 2, 3, 4, 5, 3, 4, 5, 6, 3, 4,
                            5, 6, 7, 4, 5, 6, 7, 0, 1, 5, 6, 7]], dtype=torch.long)  # edges

edge_attr = torch.tensor([1.0000, 0.5790, 0.1845, 0.4516, 0.5790, 1.0000, 0.5790, 0.1013, 0.1013,
                          0.1845, 0.5790, 1.0000, 0.4516, 0.1013, 0.4516, 1.0000, 0.4516, 0.1013,
                          0.4516, 1.0000, 0.5790, 0.1845, 0.1013, 0.5790, 1.0000, 0.5790, 0.1013,
                          0.1845, 0.5790, 1.0000, 0.4516, 0.4516, 0.1013, 0.1013, 0.4516, 1.0000],
                         dtype=torch.float)  # edge weights

## remove self-loops of the adjacent matrix (graph)
edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

## creating the graph dataset using the TMA first order terms of each channel
graphTrainDataset = []

for i in range(X_train.shape[0]):
    tma = X_train[i, ...].squeeze()
    x = torch.tensor(tma, dtype=torch.float)
    y = torch.tensor([y_train[i]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graphTrainDataset.append(data)

graphTestDataset = []

for i in range(X_test.shape[0]):
    tma = X_test[i, ...].squeeze()
    x = torch.tensor(tma, dtype=torch.float)
    y = torch.tensor([y_test[i]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graphTestDataset.append(data)

## save the graph datasets using pickle
with open(graph_train_dataset, "wb") as fp:   #Pickling
    pickle.dump(graphTrainDataset, fp)

with open(graph_test_dataset, "wb") as fp:   #Pickling
    pickle.dump(graphTestDataset, fp)




