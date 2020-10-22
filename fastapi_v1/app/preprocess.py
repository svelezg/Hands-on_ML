#!/usr/bin/env python3
"""contains the preprocessing pipeline"""

import numpy as np
import warnings
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# Custom Transformer that fills missing fares and ages
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
