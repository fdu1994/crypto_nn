from data_loader import DataLoader
import os
from tqdm import tqdm


api_key = ""
api_secret = ""
data_loader = DataLoader(api_key, api_secret)


def main():
    start_time = "2022-01-01 00:00:00"
    end_time = "2024-05-10 00:00:00"

    data_loader = DataLoader(api_key, api_secret)
    # Open the text file in read mode

    txt_file_path = "product_names.txt"
    with open(txt_file_path, "r") as file:
        # Read all lines from the file
        filenames = [line.strip() for line in file.readlines()]
    try:
        for file in filenames[::2]:
            try:
                file_name = os.path.splitext(file)[0]
                print(file_name)
                market_data = data_loader.fetch_market_data_daily(
                    file,
                    data_loader.convert_date_to_unix_timestamp(start_time),
                    data_loader.convert_date_to_unix_timestamp(end_time),
                )
                market_data.sort_values(by="Date", ascending=True, inplace=True)
                data_loader.to_csv(market_data, file)
                # data_loader.plot_candlechart(btc_data)
                # data_loader.plot_candlechart(market_data)
            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")
                continue  # Skip to the next iteration if an error occurs
    except Exception as e:
        print(f"An error occurred: {e}")


def portfolio_breakdown_default():
    portfolios = data_loader.get_portfolios()
    data_loader.portfolio_breakdown(portfolios["portfolios"][0]["uuid"])


def portfolio_breakdown_perp():
    portfolios = data_loader.get_portfolios()
    data_loader.portfolio_breakdown(portfolios["portfolios"][1]["uuid"])


def get_orders():
    orders = data_loader.get_orders()
    return orders


def getfilenames():

    # Directory path
    directory = "/raw_data_4_hour"

    # Get all filenames in the directory
    filenames = os.listdir(directory)

    # Specify the path for the text file to save the filenames
    txt_file_path = "filenames.txt"

    # Open the text file in write mode
    with open(txt_file_path, "w") as file:
        # Write each filename to the text file
        for filename in filenames:
            file.write(filename + "\n")

    print("Filenames saved to", txt_file_path)


def get_products():
    data_loader = DataLoader(api_key, api_secret)
    products = data_loader.get_products()
    print(products["products"])
    # Specify the path for the text file to save the filenames
    txt_file_path = "product_names.txt"

    # Open the text file in write mode
    with open(txt_file_path, "w") as file:
        # Write each filename to the text file
        for filename in products["products"]:
            file.write(filename["product_id"] + "\n")
            print(filename["product_id"])


def check_files():
    # API credentials
    api_key = ""
    api_secret = ""

    data_loader = DataLoader(api_key, api_secret)
    data_loader.update_data("render_ytd_data.csv", "RNDR-USD")
    # print(data_loader.load_and_check_latest_date("render_ytd.csv"))


if __name__ == "__main__":
    # check_files()
    # get_products()
    # main()
    # getfilenames()
    # print(get_orders())
    portfolio_breakdown_default()
