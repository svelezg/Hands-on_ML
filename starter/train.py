#!/usr/bin/env python3
"""contains the training experiment"""

import pandas as pd
import numpy as np
import keras
import warnings
import yaml
import typer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import mlflow

load = __import__('utils').load
preprocess_pipeline = __import__('preprocess').preprocess_pipeline
create_model = __import__('model').create_model
create_callbacks = __import__('model').create_callbacks

warnings.filterwarnings('ignore')
#mlflow.sklearn.autolog()
mlflow.keras.autolog()
app = typer.Typer()


@app.command()
def train(csv_file: str = 'train.csv',
          params_file: str = 'params.yaml'):
    # parameters
    yaml_file = open(params_file)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    dataset_dict = parsed_yaml_file["dataset_dictionary"]
    callback_dict = parsed_yaml_file["callback_dictionary"]
    classifier_dict = parsed_yaml_file["classifier_dictionary"]

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

    # define full pipeline --> preprocessing + model
    full_pipeline = Pipeline(steps=[
        ('preprocess_pipeline', preprocess_pipeline),
        ('model', my_clasiffier)])

    # fit on the complete pipeline
    training = full_pipeline.fit(X_train, y_train)

    # metrics
    score_test = \
        round(training.score(X_test, y_test) * 100, 2)
    print(f"Test Accuracy: {score_test}")

    # mlflow.log_metric('test_accuracy', score_test)

    """
    scores = cross_val_score(full_pipeline, X_train.values, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    
    num_folds = 10
    kfold = StratifiedKFold(n_splits=num_folds,
                            shuffle=True)
    
    
    
    cv_results = cross_val_score(full_pipeline,
                              X_train, y_train,
                              cv=kfold,
                              verbose=2)
    """


if __name__ == "__main__":
    app()
