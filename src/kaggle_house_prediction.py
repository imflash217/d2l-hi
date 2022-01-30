import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import utils


def get_model(in_features):
    ## defining a linear regression model
    model = nn.Sequential(nn.Linear(in_features, 1))
    return model


def log_rmse(model, features, labels):
    ## to further stabilize the value when the logrithm is taken
    ## clamp the value less than 1 to 1 BEFORE LOGRITHM
    clamp_preds = model(features).clamp(min=1.0, max=float("inf"))
    rmse = torch.sqrt(loss_fn(torch.log(clamp_preds), torch.log(labels)))
    return rmse.item()


def get_k_fold_data(k, i, x, y):
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part = x[idx, :]
        y_part = y[idx, :]
        if j == i:
            x_valid = x_part
            y_valid = y_part
        elif x_train is None:
            x_train = x_part
            y_train = y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return x_train, y_train, x_valid, y_valid


def train(
    model,
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_epochs,
    lr,
    wd,
    bs,
):
    loss_train = []
    loss_test = []
    train_iter = d2l.load_array((train_features, train_labels), bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(num_epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        loss_train.append(log_rmse(model, train_features, train_labels))
        if test_labels is not None:
            loss_test.append(log_rmse(model, test_features, test_labels))
    return loss_train, loss_test


def k_fold(k, x_train, y_train, num_epochs, lr, wd, bs):
    loss_tr_sum = 0.0
    loss_val_sum = 0.0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        model = get_model()
        loss_tr, loss_val = train(model, *data, num_eopchs, lr, wd, bs)
        loss_tr_sum += loss_tr[-1]  ## adding only the loss from the final epoch
        loss_val_sum += loss_val[-1]
    return loss_tr_sum / k, loss_val_sum / k

def train_and_pred(x_train, y_train, x_test, num_epochs, lr, wd, bs):
    model = get_model()
    loss_tr, *_ = train(model, x_train, y_train, None, None, num_epochs, lr, wd, bs)
    preds = model(x_test).detach().numpy()
    
    ## reformat it to export ot Kaggle
    x_test["SalePrice"] = pd.Series(preds.reshape((1,-1))[0])
    submission = pd.concat((test_data["Id"], test_data["SalePrice"]), axis=1)
    submission.to_csv("submission.csv", index=False)

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
    train_features = torch.tensor(
        all_features.iloc[:n_train].values, dtype=torch.float32
    )
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    print(train_features.shape, test_features.shape)
    ## print(data_train[:3])
    train_labels = torch.tensor(
        data_train.SalePrice.values, dtype=torch.float32
    ).reshape((-1, 1))
    print(train_labels.shape)

    ######################################################################################
    ##
    ## Step-3: Training
    ##
    loss_fn = nn.MSELoss()
    in_features = train_features.shape[1]
    
    k = 5
    num_epochs = 100
    lr = 5
    wd = 0.0
    bs = 64
    
    loss_train, loss_val = k_fold(k, train_features, train_labels, num_epochs, lr, wd, bs)
    print(loss_train, loss_val)

    










