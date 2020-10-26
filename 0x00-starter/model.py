#!/usr/bin/env python3
"""contains the keras nn model"""

import keras
import yaml


def build_model(nx, layers, activations, lambtha, keep_prob,
                alpha, beta1, beta2, verbose):
    """
    builds a neural network with the Keras library
    :param nx: number of input features to the network
    :param layers: list containing the number of nodes
        in each layer of the network
    :param activations: list containing the activation
        functions used for each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :param alpha: learning rate
    :param beta1: first Adam optimization parameter
    :param beta2: second Adam optimization parameter
    :param verbose: show model
    :return: keras model
    """
    # input placeholder
    inputs = keras.Input(shape=(nx,))

    # regularization scheme
    reg = keras.regularizers.L1L2(l2=lambtha)

    # a layer instance is callable on a tensor, and returns a tensor.
    # first densely-connected layer
    my_layer = keras.layers.Dense(units=layers[0],
                                  activation=activations[0],
                                  kernel_regularizer=reg,
                                  input_shape=(nx,))(inputs)

    # subsequent densely-connected layers:
    for i in range(1, len(layers)):
        my_layer = keras.layers.Dropout(1 - keep_prob)(my_layer)
        my_layer = keras.layers.Dense(units=layers[i],
                                      activation=activations[i],
                                      kernel_regularizer=reg,
                                      )(my_layer)

    network = keras.Model(inputs=inputs, outputs=my_layer)

    network.compile(optimizer=keras.optimizers.Adam(alpha, beta1, beta2),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    if verbose:
        network.summary()

    return network


def create_callbacks(early_stopping, patience,
                     learning_rate_decay,
                     alpha, decay_rate,
                     save_best, filepath,
                     verbose):
    callback_list = []

    # decay formula
    def learning_rate(epoch):
        return alpha / (1 + decay_rate * epoch)

    # learning rate decay callback
    if learning_rate_decay:
        lrd = keras.callbacks.LearningRateScheduler(learning_rate,
                                                    verbose)
        callback_list.append(lrd)

    # models save callback
    if save_best:
        mcp_save = keras.callbacks.ModelCheckpoint(filepath,
                                                   save_best_only=True,
                                                   monitor='accuracy',
                                                   mode='max')
        callback_list.append(mcp_save)

    # early stopping callback
    if early_stopping:
        es = keras.callbacks.EarlyStopping(monitor='accuracy',
                                           mode='max',
                                           patience=patience,
                                           restore_best_weights=True)
        callback_list.append(es)

    return callback_list


def create_model():
    # load parameters
    yaml_file = open("params.yaml")
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    model_dict = parsed_yaml_file["model_dictionary"]

    # build model
    model = build_model(**model_dict)
    return model
