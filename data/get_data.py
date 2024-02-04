import yfinance as yf
import pandas as pd

def get_yahoo_finance_data(ticker, start_date, end_date):
    try:
        # Download historical data
        data = yf.download(ticker, start=start_date, end=end_date)
        return data["Close"]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def export_to_csv(data, filename):
    try:
        data.to_csv(filename)
        print(f"Data exported to {filename} successfully.")
    except Exception as e:
        print(f"Error exporting data to CSV: {e}")

if __name__ == "__main__":
    # Specify the stock symbol, start date, and end date
    ticker_symbol = "^GSPC"  # S&P 500
    start_date = "2009-05-01"
    end_date = "2018-12-01"

    # Fetch data
    stock_data = get_yahoo_finance_data(ticker_symbol, start_date, end_date)

    if stock_data is not None:
        # Export data to CSV
        csv_filename = f"{ticker_symbol}_historical_data_{start_date}_to_{end_date}.csv"
        export_to_csv(stock_data, csv_filename)