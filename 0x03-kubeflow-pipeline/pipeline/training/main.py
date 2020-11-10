#!/usr/bin/env python3
"""contains the training experiment"""

import warnings

warnings.filterwarnings('ignore')

# UTILITY FUNCTIONS************************************************************
# *****************************************************************************
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def load(filename):
    """
    unzip folder and load raw dara
    :param filename: file_ path
    :return: raw_data
        Raw data set
    """
    data = pd.read_csv(filename)
    # print('data set shape: ', data.shape, '\n')
    # print(data.head())
    return data


def unzip_and_load(zipfolder, filename):
    """
    unzip folder and load raw dara
    :param zipfolder: zip folder path
    :param filename: file_ path
    :return: raw_data
        Raw data set
    """
    with zipfile.ZipFile(zipfolder, 'r') as zip_ref:
        zip_ref.extractall()

    data = pd.read_csv(filename)

    print('data set shape: ', data.shape, '\n')
    print(data.head())

    return data


def csv_from_url(csv_url):
    """
    loads data from csv url
    :param csv_url: example:
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    :return: data
    """
    try:
        data = pd.read_csv(csv_url, sep=";")
        # print('data set shape: ', data.shape, '\n')
        # print(data.head())
        return data
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, "
            "check your internet connection. Error: %s", e
        )


def fig_to_html(fig: plt.Figure) -> str:
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}'>"


# PRE-PROCESSING **************************************************************
# *****************************************************************************
import numpy as np
import warnings
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# Custom Transformer that fills missing ages
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.age_means_ = {}

    def fit(self, X, y=None):
        self.age_means_ = X.groupby(['Pclass', 'Sex']).Age.mean()

        return self

    def transform(self, X, y=None):
        # fill Age
        for key, value in self.age_means_.items():
            X.loc[((np.isnan(X["Age"])) & (X.Pclass == key[0]) & (X.Sex == key[1])), 'Age'] = value

        return X


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.means_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = X.to_numpy()
        self.means_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)

        return self

    def transform(self, X, y=None):
        X[:] = (X.to_numpy() - self.means_) / self.std_

        return X


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Helper function that converts values to Binary depending on input
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

    # Transformer method for this transformer
    def transform(self, X, y=None):
        # Categorical features to pass down the categorical pipeline
        return X[['Sex']].values


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Numerical features to pass down the numerical pipeline
        X = X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.values


# Defining the steps in the categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ('cat_transformer', CategoricalTransformer()),
    ('one_hot_encoder', OneHotEncoder(sparse=False))])

# Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline(steps=[
    ('num_transformer', NumericalTransformer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())])

# Combining numerical and categorical pipeline into one full big pipeline horizontally
# using FeatureUnion
union_pipeline = FeatureUnion(transformer_list=[
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline)])

# Combining the custom imputer with the categorical and numerical pipeline
preprocess_pipeline = Pipeline(steps=[('custom_imputer', CustomImputer()),
                                      ('full_pipeline', union_pipeline)])

# MODEL ***********************************************************************
# *****************************************************************************
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

    # history log
    hlog = keras.callbacks.CSVLogger('training_log.csv', append=True)
    callback_list.append(hlog)

    return callback_list


def create_model():
    # load parameters
    yaml_file = open("./training/params.yaml")
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    model_dict = parsed_yaml_file["model_dictionary"]

    # build model
    model = build_model(**model_dict)
    return model


# TRAINING ********************************************************************
# *****************************************************************************
import yaml
import typer
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

import json
import os
from pathlib import Path

from matplotlib import pyplot as plt
from tensorflow.python.lib.io import file_io

app = typer.Typer()
ROOT = Path(os.environ.get("ROOT", "/"))


@app.command()
def train(epochs: int = 200,
          batch_size: int = 125,
          csv_file: str = './training/train.csv',
          params_file: str = './training/params.yaml'):
    print(f"Epochs: {epochs}")

    # parameters
    yaml_file = open(params_file)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    dataset_dict = parsed_yaml_file["dataset_dictionary"]
    callback_dict = parsed_yaml_file["callback_dictionary"]
    classifier_dict = parsed_yaml_file["classifier_dictionary"]

    classifier_dict['epochs'] = epochs
    classifier_dict['batch_size'] = batch_size

    # data loading
    data = load(csv_file)

    # input and labels
    X = data.drop(dataset_dict['label'], axis=1)
    y = data[dataset_dict['label']].values

    # dataset split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=dataset_dict['test_size'],
                         random_state=42)

    # training
    callback_list = create_callbacks(**callback_dict)

    my_clasiffier = KerasClassifier(build_fn=create_model,
                                    callbacks=callback_list,
                                    **classifier_dict)

    # define full pipeline --> pre-processing + model
    full_pipeline = Pipeline(steps=[
        ('preprocess_pipeline', preprocess_pipeline),
        ('model', my_clasiffier)])

    # fit on the complete pipeline
    training = full_pipeline.fit(X_train, y_train)

    df = pd.read_csv('training_log.csv')
    print(df)

    # training history (accuracy and loss)
    with plt.xkcd():
        fig = plt.figure()
        plt.title('train accuracy and loss')
        plt.plot(df['accuracy'], label='accuracy')
        plt.plot(df['loss'], label='loss')
        plt.ylabel('accuracy and loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    metadata = {
        "inputs": [
            {
                "dataset": dataset_dict,
                "callbacks": callback_dict,
                "training": classifier_dict,
            },
        ],
        "outputs": [
            {
                "type": "web-app",
                "storage": "inline",
                "source": fig_to_html(fig),
            },
        ]
    }

    json_file = str(ROOT / "mlpipeline-ui-metadata.json")
    print('json file path:', json_file)

    with file_io.FileIO(json_file, "w") as f:
        json.dump(metadata, f)

    # metrics
    score_train = \
        round(training.score(X_train, y_train) * 100, 2)
    print(f"\nTrain Accuracy: {score_train}")

    score_test = \
        round(training.score(X_test, y_test) * 100, 2)
    print(f"\nTest Accuracy: {score_test}")

    metrics = {
        "metrics": [
            {
                "training_accuracy": score_test,
                "test_accuracy": score_test,
            },
        ]
    }

    json_file = str(ROOT / "mlpipeline-metrics.json")
    print('json file path:', json_file)

    with file_io.FileIO(json_file, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    app()
