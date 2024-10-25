from coinbase.rest import RESTClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go


class DataLoader:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)

    def get_portfolios(self):
        portfolios = self.client.get_portfolios()
        return portfolios

    def portfolio_breakdown(self, portfolio_id):
        portfolio_breakdowns = self.client.get_portfolio_breakdown(portfolio_id)
        df = pd.DataFrame.from_dict(portfolio_breakdowns["breakdown"]["spot_positions"])
        print(df)
        costs = df["cost_basis"]
        balances_fiat = df["total_balance_fiat"]
        total_balance = balances_fiat.sum()

        # Extract 'value' and 'currency' from each dictionary
        values = [float(item["value"]) for item in costs]
        currencies = [item["currency"] for item in costs]

        # Create DataFrame
        df = pd.DataFrame({"value": values, "currency": currencies})
        print(df["value"].sum())
        total_costs = df["value"].sum()
        ratio = total_balance / total_costs
        print(
            f"Total cost: {total_costs}, total balance: {total_balance}, ratio: {ratio}"
        )

    def fetch_market_data(self, product, start_time, end_time, chunk_size_hours=600):
        self.client.market_order()
        dfs = []
        current_time = start_time
        while current_time < end_time:
            chunk_end_time = min(current_time + (chunk_size_hours * 3600), end_time)
            market_candles = self.client.get_candles(
                product_id=product,
                start=str(current_time),
                end=str(chunk_end_time),
                granularity="TWO_HOUR",
            )
            df_chunk = pd.DataFrame.from_dict(market_candles)
            df_chunk[["start", "low", "high", "open", "close", "volume"]] = df_chunk[
                "candles"
            ].apply(pd.Series)
            df_chunk.drop(columns=["candles"], inplace=True)
            df_chunk["Date"] = pd.to_datetime(df_chunk["start"].astype(int), unit="s")
            df_chunk["Low"] = df_chunk["low"].astype(float)
            df_chunk["High"] = df_chunk["high"].astype(float)
            df_chunk["Open"] = df_chunk["open"].astype(float)
            df_chunk["Close"] = df_chunk["close"].astype(float)
            df_chunk["Volume"] = df_chunk["volume"].astype(float)
            dfs.append(df_chunk)
            current_time = chunk_end_time
        df_res = pd.concat(dfs, ignore_index=True)
        df_res.drop(
            columns=["start", "low", "high", "open", "close", "volume"],
            inplace=True,
        )
        df_res["Asset_name"] = "BTCUSDT"
        return df_res

    def fetch_market_data_six_hour(
        self, product, start_time, end_time, chunk_size_hours=1800
    ):
        dfs = []
        current_time = start_time
        while current_time < end_time:
            chunk_end_time = min(current_time + (chunk_size_hours * 3600), end_time)
            market_candles = self.client.get_candles(
                product_id=product,
                start=str(current_time),
                end=str(chunk_end_time),
                granularity="SIX_HOUR",
            )
            df_chunk = pd.DataFrame.from_dict(market_candles)
            df_chunk[["start", "low", "high", "open", "close", "volume"]] = df_chunk[
                "candles"
            ].apply(pd.Series)
            df_chunk.drop(columns=["candles"], inplace=True)
            df_chunk["Date"] = pd.to_datetime(df_chunk["start"].astype(int), unit="s")
            df_chunk["Low"] = df_chunk["low"].astype(float)
            df_chunk["High"] = df_chunk["high"].astype(float)
            df_chunk["Open"] = df_chunk["open"].astype(float)
            df_chunk["Close"] = df_chunk["close"].astype(float)
            df_chunk["Volume"] = df_chunk["volume"].astype(float)
            dfs.append(df_chunk)
            current_time = chunk_end_time
        df_res = pd.concat(dfs, ignore_index=True)
        df_res.drop(
            columns=["start", "low", "high", "open", "close", "volume"],
            inplace=True,
        )
        df_res["Asset_name"] = "BTCUSDT"
        return df_res

    def fetch_market_data_daily(
        self, product, start_time, end_time, chunk_size_hours=7200
    ):
        dfs = []
        current_time = start_time
        while current_time < end_time:
            chunk_end_time = min(current_time + (chunk_size_hours * 3600), end_time)
            market_candles = self.client.get_candles(
                product_id=product,
                start=str(current_time),
                end=str(chunk_end_time),
                granularity="ONE_DAY",
            )
            df_chunk = pd.DataFrame.from_dict(market_candles)
            df_chunk[["start", "low", "high", "open", "close", "volume"]] = df_chunk[
                "candles"
            ].apply(pd.Series)
            df_chunk.drop(columns=["candles"], inplace=True)
            df_chunk["Date"] = pd.to_datetime(df_chunk["start"].astype(int), unit="s")
            df_chunk["Low"] = df_chunk["low"].astype(float)
            df_chunk["High"] = df_chunk["high"].astype(float)
            df_chunk["Open"] = df_chunk["open"].astype(float)
            df_chunk["Close"] = df_chunk["close"].astype(float)
            df_chunk["Volume"] = df_chunk["volume"].astype(float)
            dfs.append(df_chunk)
            current_time = chunk_end_time
        df_res = pd.concat(dfs, ignore_index=True)
        df_res.drop(
            columns=["start", "low", "high", "open", "close", "volume"],
            inplace=True,
        )
        df_res["Asset_name"] = product
        return df_res

    def get_orders(self):
        self.client.get_fills()

    def get_products(self):
        products = self.client.get_products(limit=300, offset=2)
        return products

    def load_and_check_latest_date(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            latest_date = int(pd.to_datetime(df["start"].max()).timestamp())
            now = int((datetime.now(timezone.utc) + timedelta(hours=2)).timestamp())
            if now - latest_date > (timedelta(hours=2).total_seconds()):
                # Data is not up to date, fetch market data
                print("data not up to date")
                return latest_date
            else:
                # Data is up to date
                print("data up to date")
                return None
        except FileNotFoundError:
            # File doesn't exist, fetch market data
            return None

    # TODO: don't update too much. fix!
    def update_data(self, csv_file, product):
        latest_date = self.load_and_check_latest_date(csv_file)
        if latest_date is not None:
            # Fetch market data up to now
            current_time = latest_date
            end_time = int(
                (datetime.now(timezone.utc) + timedelta(hours=2)).timestamp()
            )
            print(end_time)
            new_data = self.fetch_market_data(product, current_time, end_time)
            if new_data.empty:
                print("No new data available.")
            else:
                # Append new data to the CSV file
                new_data.to_csv(csv_file, mode="a", header=False, index=False)
                print("Data updated successfully.")
        else:
            print("Data is up to date.")

    def convert_date_to_unix_timestamp(self, date_string):
        date_obj = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        unix_timestamp = int(date_obj.timestamp())
        return unix_timestamp

    def plot_candlechart(self, df):
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                )
            ]
        )

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    def to_csv(self, df, name):
        df.to_csv(f"training_data_daily/{name}.csv", index=False)
        print(f"{name} data saved to {name}.csv")
