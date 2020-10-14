#!/usr/bin/env python3
"""contains the preprocessing pipeline"""

import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# Custom Transformer that fills missing fares and ages
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # fill Fare
        X['Fare'] = X.groupby(['Pclass', 'Sex'])['Fare'] \
            .transform(lambda x: x.fillna(x.mean()))

        # fill Age
        X['Age'] = X.groupby(['Pclass', 'Sex'])['Age'] \
            .transform(lambda x: x.fillna(x.mean()))

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Helper function that converts values to Binary depending on input
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        return X.values


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.values


# Categorical features to pass down the categorical pipeline
categorical_features = ['Sex']

# Numerical features to pass down the numerical pipeline
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Defining the steps in the categorical pipeline
categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
                                       ('cat_transformer', CategoricalTransformer()),
                                       ('one_hot_encoder', OneHotEncoder(sparse=False))])

# Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                     ('num_transformer', NumericalTransformer()),
                                     ('imputer', SimpleImputer(strategy='median')),
                                     ('std_scaler', StandardScaler())])

# Combining numerical and categorical pipeline into one full big pipeline horizontally
# using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline)])

# Combining the custum imputer with the categorical and numerical pipeline
preprocess_pipeline = Pipeline(steps=[('custom_imputer', CustomImputer()),
                                      ('full_pipeline', full_pipeline)])
