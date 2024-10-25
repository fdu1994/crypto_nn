from pandas import Timestamp

run1 = {
    "folder": "raw_data_4_hour_BTCUSDT/",
    # 'folder' : 'raw_data_1_hour/',
    # 'folder' : 'raw_data_30_min/',
    #'folder' : 'raw_data_1_day/',
    "reports": "reports/",
    "alpha": 0.05,  # computed in determine_alpha.py
    "beta": 0.1627,  # ignore sample greater than beta in percent of change
    "seed": 353598215,
    "commission fee": 0.001,  # 0.0004,  # 0.001,
    "b_window": 2,
    "f_window": 2,
    # used in define the grid for searching backward and forward window
    "b_lim_sup_window": 6,
    "f_lim_sup_window": 6,
    "back_test_start": Timestamp("2023-10-01"),
    "back_test_end": Timestamp("2024-01-10"),
    "suffix": "ncr",
    "stop_loss": 0.05,
    "off_label_set": [],  # ['BTCUSDT', 'ETHUSDT', 'ALGOUSDT']  # list of coin to be excluded from training/test set. Used in backtesting
    "balance_algo": "srs",  # 'ncr', 'srs', None
    "loss_func": "categorical",  # 'focal', 'categorical'
    "epochs": 32,  # how many epochs spent in training neural network
}

RUN = run1
