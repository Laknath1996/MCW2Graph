# MCW2Graph

## Installation Guide

### Step 1. Clone the respository

Clone the repository using the following command.

```
$ git clone https://github.com/Laknath1996/MCW2Graph.git
```

### Step 1. Install the dependencies

Use the `requirements.txt` file given in the repository to install the dependencies via `pip` in a newly created conda environment with python 3.6.

```
$ pip install -r requirements.txt 
```

## Documentation

To access the documentation, open `/docs/build/html/index.html` using a web browser.

## Project Organization

------------
    ├── README.md                       <- Project Details
    ├── requirements.txt                <- Dependencies
    ├── docs                            <- Documentation
    ├── experiments                     <- Experimental Log
    ├── data                            <- Datasets
    │   ├── kushaba                     <- Dataset from Kushaba et al.
    │   ├── de_silva                    <- Dataset from De Silva et al.
    ├── graph_data                      <- Graph Datasets
    │   └── de_silva
    ├── data_handling                   <- Data handling scripts
    │   └── utils.py
    ├── gcn                             <- GCN architectures
    │   └── architectures.py
    ├── graph_learning                  <- Graph learning methods
    │   ├── methods.py
    │   └── utils.py
    ├── grnn                            <- GRNN architectures
    │   └── architectures.py
    ├── optimizers                      <- Some solvers from cvxopt
    │   ├── l1.py
    │   └── l1regls.py
    ├── real_time_scripts               <- Real time scripts from De Silva et al.
    │   ├── real_time_prediction.py
    │   ├── real_time_prediction_gfreq_onsets.py
    │   └── record_training_data.py
    ├── tma                             <- Scripts from De Silva et al.
    │   ├── functions.py
    │   ├── models
    │   │   ├── classifiers.py
    │   │   └── nn_models.py
    │   └── utils.py
    ├── dev                              <- Dev folder (test scripts)
    │   ├── graph_learning_validation.py
    │   ├── grnn_with_mini_batches.py
    │   └── quick_SVM.py
    ├── dev.ipynb                        <- Dev notebook (reserved for testing)
    ├── create_graph_dataset.py          <- Create distance based graph dataset
    ├── learn_graph_topology_kushaba.py   <- Graph learning for Kushaba et al. 
    ├── learn_graph_topology_desilva.py      <- Graph learning for De Silva et al.
    ├── graph_classification_diff_pool.py  <- Diff Pool paper implementation
    ├── graph_classification_mlp.py        <- MLP implementation
    ├── graph_classification_svm.py        <- SVM implementation
    ├── graph_classification_vanilla.py    <- GCN implementation (*)
    └── experiments.py                     <- Script to run the final experiments
-------------


