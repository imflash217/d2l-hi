import numpy as np
import pandas as pd
import torch
from torch import nn

import utils


def main():
    ## downloading the data from Kaggle
    ## data_train = pd.read_csv(utils.download("kaggle_house_train"))
    ## data_test = pd.read_csv(utils.download("kaggle_house_test"))
    data_train = pd.read_csv(
        "../data/house-prices-advanced-regression-techniques/train.csv"
    )
    data_test = pd.read_csv(
        "../data/house-prices-advanced-regression-techniques/test.csv"
    )
    print(f"train.shape = {data_train.shape}\ntest.shape = {data_test.shape}")
    print(data_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])

    ######################################################################################
    ##
    ## removing the ID from the features set
    ## and also removing the "labels" for the trainign data
    all_features = pd.concat((data_train.iloc[:, 1:-1], data_test.iloc[:, 1:]))
    print(f"all_features.shape = {all_features.shape}")
    print(all_features.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])

    ## Step-1: Dealing with NUMERIC FEATURES & N/A
    ##
    ## calculating the mean & std for normaalization of all numeric features
    ## print(all_features.dtypes)
    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    ## normalization
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    ## after normalizing the data, the mean is ZERO now, so w set the missing values to 0 (i.e. the current mean)
    all_features[numeric_features] = all_features[numeric_features].fillna(0.0)
    ## print(all_features[numeric_features].iloc[:10, 0:15])

    ## Step-2: Dealing with Descrete values
    ## replacing them by 1-hot encoding
    ## and replacing the categorical values into 1-hot encoded labels
    ##
    all_features = pd.get_dummies(all_features, dummy_na=True)
    ## print(f"all_features.shape = {all_features.shape}")
    print(all_features.dtypes)

    ## Step-3: Converting the Pandas DataFrame into Tensor, we we can train
    ##
    n_train = data_train.shape[0]
    train_features = torch.tensor(all_features.iloc[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    print(train_features.shape, test_features.shape)
    ## print(data_train[:3])
    train_labels = torch.tensor(data_train.SalePrice.values, dtype=torch.float32).reshape((-1,1))
    print(train_labels.shape)

    ######################################################################################
    ##
    ## Step-3: Training
    ##

