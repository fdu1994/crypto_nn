import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import compute_indicators_labels_lib
import imbalanced_lib
from Pytorch_NNModel import NNModel
import torch
from torch.utils.data import DataLoader
from CoinDataset import CustomDataset
import tensorflow as tf
from sklearn.utils import shuffle
import random
import torch.nn as nn
import torch.optim as optim

from config import RUN as run_conf
from numpy.random import seed
from tensorflow import random as tf_rand
from imbalanced_lib import get_sampler


def train_test(RUN, save_to="model.pt"):
    random.seed(RUN["seed"])
    seed(42)

    if torch.cuda.is_available():
        print("GPU is available!")
    scaler = StandardScaler()
    sampler = get_sampler(run_conf["balance_algo"])
    data = compute_indicators_labels_lib.get_dataset(RUN)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[
        (data["Date"] < RUN["back_test_start"]) | (data["Date"] > RUN["back_test_end"])
    ]  # exclude backtest data from trainig/test set

    data = data[data["pct_change"] < RUN["beta"]]  # remove outliers

    labels = data["label"].copy()
    print(f"Label value counts: {labels.value_counts()}")
    data.drop(
        columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Asset_name",
            "label",
        ],
        inplace=True,
    )
    columns = data.columns
    index = data.index
    X = scaler.fit_transform(data.values)

    data = pd.DataFrame(X, columns=columns, index=index)
    data["label"] = labels
    data.dropna(inplace=True)

    data = shuffle(data, random_state=RUN["seed"])
    data = sampler(data)

    # Xs, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    train_set, test_set = train_test_split(
        data, test_size=0.3, random_state=RUN["seed"]
    )

    # print(train_set)
    print(f"train set shape 1: {train_set.shape[1]}")
    print(f"train set columns: {train_set.columns}")

    model = NNModel(train_set.shape[1] - 1, 3).to("cuda")
    # Define your loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters())
    train_loader = DataLoader(CustomDataset(train_set), batch_size=32)
    test_loader = DataLoader(CustomDataset(test_set), batch_size=32)
    model.print_num_parameters()
    model.train_model(train_loader, criterion, optimizer, RUN["epochs"], "torch_model")
    torch.save(model.state_dict(), save_to)
    # model.dummy_train(train_set, y_train)

    preds_test = model.predict(test_loader)
    preds_train = model.predict(train_loader)

    test_rep = classification_report(
        test_set["label"], preds_test, digits=2, output_dict=True
    )
    print(test_rep)
    train_rep = classification_report(
        train_set["label"], preds_train, digits=2, output_dict=True
    )
    print("================================================")
    print(train_rep)

    rep_fields = [
        "",
        "bw",
        "fw",
        "tf",
        "alpha",
        "beta",
        "fee",
        "epchs",
        "bt_start",
        "bt_end",
        "ba_alg",
        "acc",
        "wa_prec",
        "wa_rec",
        "wa_f1",
        "-1 prec",
        "0 prec",
        "1 prec",
        "-1 rec",
        "0 rec",
        "1 rec",
        "-1 f1",
        "0 f1",
        "1 f1",
        "-1 supp",
        "0 supp",
        "1 supp",
    ]
    rep_rows = []

    ds = {"TestRep": test_rep, "TrainRep": train_rep}  # , "DummyRep": dummy_rep}
    for d in ds:
        rep = ds[d]
        row = [
            d,
            RUN["b_window"],
            RUN["f_window"],
            RUN["folder"],
            RUN["alpha"],
            RUN["beta"],
            RUN["commission fee"],
            RUN["epochs"],
            RUN["back_test_start"],
            RUN["back_test_end"],
            RUN["balance_algo"],
            rep["accuracy"],
            rep["weighted avg"]["precision"],
            rep["weighted avg"]["recall"],
            rep["weighted avg"]["f1-score"],
        ]
        for k in ["precision", "recall", "f1-score", "support"]:
            row.append(rep[str(-1)][k])
            row.append(rep[str(0)][k])
            row.append(rep[str(1)][k])

        rep_rows.append(row)

    return rep_rows, rep_fields


if __name__ == "__main__":
    train_test(run_conf)
