# crypto_nn
Rough implementation of the paper "A profitable trading algorithm for cryptocurrencies using a Neural Network model"
I partly took some of their code I found online and adjusted it for my project.

The pipeline code is constituted by a series of scripts to run in sequence.

- config.py: script configurations.

- run_preprocess_dataset.py: 
	Creates the preprocessed dataset and saves it into a csv file in the folder processed_data/

- run_data_stats.py:
	 Plots the charts of time data distribution.

- run_alpha_beta.py: 
	Computes alpha and beta, (the computed values must be copied and pasted into config.py).

- run_search_bw_fw.py: 
	The grid search for backward and forward windows.

- torch_training.py:
	The training of PyTorch model. The output saves reports into torch_model/final_model_*_*.xlsx. 

- backtest_coin.py:
	The backtest and saves reports into reports/backtest_final.xlsx.
