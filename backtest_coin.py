import pandas as pd
import numpy as np

import compute_indicators_labels_lib
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from sklearn.preprocessing import StandardScaler
from Pytorch_NNModel import NNModel
from sklearn.model_selection import train_test_split
import os
import traceback
import sys
from config import RUN as run_conf
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import torch
from torch.utils.data import DataLoader
from CoinDataset import CustomDataset
from technical_analysis_lib import TecnicalAnalysis


def calc_cum_ret_s1(x, stop_loss, fee):
    """
    compute cumulative return strategy s1.
    compute the multiplication factor of original capital after n iteration,
    reinvesting the gains.
    It search for BUY order, execute them and close them when stop_loss hits happen or when reversal/SELL signal.
    Finds the pairs BUY/SELL orders and compute the cumulative return.
    If no BUY order is encountered, it does nothing.
    If no SELL order is encountered, once a BUY order was issued, a dummy SELL order of the position is issued at the end of the period.
    :param x: ordered prices timeseries of a coin and labels
    :param stop_loss:
    :param fee: commission fee applied
    :return: history, capital, num_op, min_drowdown, max_gain
    """

    order_pending = 0
    price_start = 0
    price_end = 0
    capital = 1
    history = []
    labs = x["label"].values  # debug

    min_drowdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0
    for row in x.itertuples():

        # handle stop loss
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                order_pending = 0

                price_end = price_start * (1 - stop_loss)

                pct_chg = (price_end - price_start) / price_start
                if pct_chg < min_drowdown:
                    min_drowdown = pct_chg

                capital *= 1 + (
                    ((price_end * (1 - fee)) - (price_start * (1 + fee)))
                    / (price_start * (1 + fee))
                )
                price_start = price_end = 0

        history.append(capital)

        if row.label == BUY:
            if order_pending:
                continue

            else:
                order_pending = 1
                price_start = row.Close
                num_ops += 1
                continue

        if row.label == HOLD:
            continue

        if row.label == SELL:
            if order_pending:
                price_end = row.Close
                pct_chg = (price_end - price_start) / price_start
                if pct_chg > 0:
                    good_ops += 1

                if pct_chg < min_drowdown:
                    min_drowdown = pct_chg

                if pct_chg > max_gain:
                    max_gain = pct_chg

                order_pending = 0
                capital *= 1 + (
                    ((price_end * (1 - fee)) - (price_start * (1 + fee)))
                    / (price_start * (1 + fee))
                )
                price_start = price_end = 0
                continue

            else:
                continue

    # handle last candle
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)

        if pct_chg < min_drowdown:
            min_drowdown = pct_chg

        if pct_chg > max_gain:
            max_gain = pct_chg

        capital *= 1 + (
            ((price_end * (1 - fee)) - (price_start * (1 + fee)))
            / (price_start * (1 + fee))
        )

    return history, capital, num_ops, min_drowdown, max_gain, good_ops


def backtest_single_coin(RUN, filename, mdl_name="torch_model/epoch_32.pt", suffix=""):
    """
    Backtest a coin whose timeseries is contained in filename.
    It uses last model trained.
    Backtest period selected in RUN config dictionary
    :param suffix:
    :param mdl_name:
    :param RUN:
    :param filename:
    :return: a dictionary with neural net (nn) statistic of backtest
    tuple composed by: (final capital, num operations completed, min drowdown, max_gain, positive ops)
    """
    try:
        nr = StandardScaler()
        data_val = pd.read_csv(f"{RUN['folder']}{filename}")

        data_val.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_val = data_val.dropna()
        data_val["Date"] = pd.to_datetime(data_val["Date"])

        data_val = data_val[data_val["Date"] >= RUN["back_test_start"]]
        data_val = data_val[data_val["Date"] <= RUN["back_test_end"]]
        data_val = TecnicalAnalysis.compute_oscillators(data_val)
        data_val = TecnicalAnalysis.find_patterns(data_val)
        data_val = TecnicalAnalysis.add_timely_data(data_val)
        data_val["label"] = TecnicalAnalysis.assign_labels(
            data_val, 5, 2, RUN["alpha"], RUN["beta"]
        )
        data = pd.read_csv(f"{RUN['folder']}{filename}")

        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)
        data["label"] = TecnicalAnalysis.assign_labels(
            data, 5, 2, RUN["alpha"], RUN["beta"]
        )
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.dropna()
        data["Date"] = pd.to_datetime(data["Date"])

        data = data[data["Date"] >= RUN["back_test_start"]]
        data = data[data["Date"] <= RUN["back_test_end"]]
        if len(data.index) == 0:
            raise ValueError("Void dataframe")

        labels = data["label"].copy()
        ohlc = data[["Date", "Open", "High", "Low", "Close", "label"]].copy()
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
        if len(data.index) == 0:
            raise ValueError("Void dataframe")
        columns = data.columns
        index = data.index
        nr.fit(data)
        X = nr.transform(data)
        data = pd.DataFrame(X, columns=columns, index=index)
        data["label"] = ohlc["label"]

        print(f"train set shape 1: {data.shape[1]}")
        print(f"train set columns: {data.columns}")
        train_loader = DataLoader(CustomDataset(data), batch_size=16)
        model = NNModel(data.shape[1] - 1, 3).to("cuda")
        # model.dummy_train(X_train, y_train)
        model.load_state_dict(torch.load(mdl_name))
        labels = model.predict(train_loader)
        print(labels)
        print(len(labels))
        data["label"] = labels
        data["Open"] = ohlc["Open"]
        data["High"] = ohlc["High"]
        data["Low"] = ohlc["Low"]
        data["Close"] = ohlc["Close"]
        data["Date"] = ohlc["Date"]
        hist_nn, cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn = (
            calc_cum_ret_s1(data, RUN["stop_loss"], RUN["commission fee"])
        )

        dates = list(data.index)

        ya = np.array(hist_nn)
        ya = np.log(ya)

        prices = data["Close"]

        plt.rcParams["font.size"] = 14
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        fig, axs = plt.subplots(2, 1)
        fig = plt.gcf()
        fig.set_size_inches(8, 8)

        f = RUN["f_window"]
        b = RUN["b_window"]

        ax = axs[0]
        ax.set_facecolor("#eeeeee")
        box = ax.get_position()
        box.y0 = box.y0 + 0.03
        ax.set_position(box)
        ax.plot(dates, ya, label="MLP", color="green")
        ax.set(
            xlabel="",
            ylabel="Log(Return)",
            title=filename.split(".")[0] + " " + "backW=%d, forW=%d" % (b, f),
        )
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
        ax.grid()
        ax.legend()

        ax = axs[1]
        ax.set_facecolor("#eeeeee")
        ax.plot(dates, prices)
        ax.set(xlabel="", ylabel="Price", title="")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
        ax.grid()

        fig.savefig(
            RUN["reports"] + filename.split(".")[0] + "_b%d_f%d_%s.png" % (b, f, suffix)
        )
        # plt.show()

        return {
            "nn": (cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn),
        }

    except Exception:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)


if __name__ == "__main__":
    report = []
    res = backtest_single_coin(run_conf, "ETHUSDT.csv")
    columns = [
        "model",
        "period",
        "bw",
        "fw",
        "coin",
        "final cap",
        "num op",
        "min drawdown ",
        "max gain ",
        "good ops",
    ]
    rep = [
        "NN",
        "short",
        5,
        2,
        "ETHUSDT",
        res["nn"][0],
        res["nn"][1],
        res["nn"][2],
        res["nn"][3],
        res["nn"][4],
    ]
    report.append(rep)

    rep = pd.DataFrame(report, columns=columns)
    print("report to excel")
    print(rep)
    rep.to_excel(run_conf["reports"] + "backtest_final1.xlsx")
