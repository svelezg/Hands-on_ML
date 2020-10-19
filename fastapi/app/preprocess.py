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
        # fill Age

        X.loc[((np.isnan(X["Age"])) & (X.Pclass == 1) & (X.Sex == 'female')), 'Age'] = 35.197183
        X.loc[((np.isnan(X["Age"])) & (X.Pclass == 1) & (X.Sex == 'male')), "Age"] = 41.364270
        X.loc[((np.isnan(X["Age"])) & (X.Pclass == 2) & (X.Sex == 'female')), "Age"] = 28.689394
        X.loc[((np.isnan(X["Age"])) & (X.Pclass == 2) & (X.Sex == 'male')), "Age"] = 30.91663
        X.loc[((np.isnan(X["Age"])) & (X.Pclass == 3) & (X.Sex == 'female')), "Age"] = 21.478261
        X.loc[((np.isnan(X["Age"])) & (X.Pclass == 3) & (X.Sex == 'male')), "Age"] = 26.709258

        # fill Fare
        X.loc[((np.isnan(X["Fare"])) & (X.Pclass == '1') & (X.Sex == 'female')), "Fare"] = 107.52020875000001
        X.loc[((np.isnan(X["Fare"])) & (X.Pclass == '1') & (X.Sex == 'male')), "Fare"] = 69.94506203703703
        X.loc[((np.isnan(X["Fare"])) & (X.Pclass == '2') & (X.Sex == 'female')), "Fare"] = 22.20925294117647
        X.loc[((np.isnan(X["Fare"])) & (X.Pclass == '2') & (X.Sex == 'male')), "Fare"] = 19.0605198019802
        X.loc[((np.isnan(X["Fare"])) & (X.Pclass == '3') & (X.Sex == 'female')), "Fare"] = 16.688050
        X.loc[((np.isnan(X["Fare"])) & (X.Pclass == '3') & (X.Sex == 'male')), "Fare"] = 12.652100

        return X


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

    # Transformer method for this transformer
    def transform(self, X, y=None):
        # Categorical features to pass down the categorical pipeline
        return X[['Sex']].values


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

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
