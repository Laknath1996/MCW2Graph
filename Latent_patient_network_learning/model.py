"""

This code segment contains the model for Latent Patient network
based on :
Latent Patient Network Learning for Automatic Diagnosis
Luca Cosmo, Anees Kazi, Seyed-Ahmad Ahmadi, Nassir Navab, and Michael Bronstein

"""
import os
import random
import numpy as np
from numpy.linalg import norm

from keras.models import Sequential
from keras.layers import Dense


def multi_layer_perceptron(x, lower_dimension, num_layers):
    # Set up the number of perceptron per each layer:
    num_in_initial_layer = 12  # Layer 1
    num_in_other_layer = 8  # Layer 2

    #TODO: might have to build this MLP from scratch as its learnt E2E

    # define the keras model - not training to act as a MLP
    model = Sequential()
    model.add(Dense(num_in_initial_layer, input_dim=x.Shape[0], activation='relu'))

    for _ in (num_layers - 1):
        model.add(Dense(num_in_other_layer, activation='relu'))
    model.add(Dense(lower_dimension, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model.predict(x)

def sigmoid_func(x_i, x_j, t, theta):
    norm_val = np.sqrt(norm(x_i - x_j))
    return 1 / (1 + (np.exp(-t * (norm_val + theta))))

def adjacency_matrix(x_dash, t, theta):
    a_ij = [[sigmoid_func(i, j, t, theta) for j in x_dash] for i in x_dash]
    return a_ij

def latent_graph_learn(x, t, theta, lower_dim, num_layers):
    x_dash = []
    for x_i in x:
        x_i_dash = multi_layer_perceptron(x_i, lower_dim, num_layers)
        x_dash.append(x_i_dash)

    A = adjacency_matrix(x_dash, t, theta)
    return A

def gc_layer(H_l, input_feature):
    # TODO : what is the dimension of H_l

    lower_dimension = 3        # lower Dimension of the euclidean space
    num_perceptron_layers = 4  # number of perceptron layers
    t_value = 0.25             # this value is required to get the sigmoid function (learnable weigh)
    theta_value = 0.3          # this value is required to get the sigmoid function (learnable weigh)
    x = input_feature          # Input set of nodes

    A = latent_graph_learn(x, t_value, theta_value, lower_dimension, num_perceptron_layers) #return is a 2D list

    A_metrix = np.array(A)

    #Normalization of matrix A happens here
    row_sums = A_metrix.sum(axis=1)
    A_normalized = A_metrix / row_sums[:, np.newaxis]

    # TODO : how to incorporate MLP's weights over here ? is it required ? if so how
    W = np.arange(t_value,theta_value).reshape(2,1)

    H_l_plus1 = np.matmul(A_normalized, H_l, W)
    return H_l_plus1


### set randome seed

seed_value = 10
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# TODO : loop the gc_layers with training specifics and weight updates

