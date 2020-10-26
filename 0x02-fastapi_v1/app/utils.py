#! /usr/bin/env python3
"""
utility functions
"""

import pandas as pd
import zipfile


def load(filename):
    """
    unzip folder and load raw dara
    :param zipfolder: zip folder path
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
