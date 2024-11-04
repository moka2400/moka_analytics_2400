from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Heavily inspired by fatiherik @ Kaggle

# Monte Carlo simulations are used here to generate multiple possible future price paths for the stock based on historical behavior.
# This method helps in estimating the range of potential outcomes and their probabilities.

def load_data() -> pd.DataFrame:
    # Let first column (dates) be row indices.
    data = pd.read_csv('../data/stock_data.csv', sep=",", header=0, decimal=".", index_col=0)

    df_Stock = data
    df_Stock = df_Stock.rename(columns={'Close(t)':'Close'})
    prices = df_Stock['Close'] # Focus on Close prices

    return prices

# The normal distribution is used to simulate price movements because stock returns are assumed to be normally distributed here.
# The formula price * (1 + np.random.normal(0, daily_vol)) generates a new price by applying a random return drawn from a normal distribution 
# with mean 0 and standard deviation equal to the daily volatility.
def get_randomly_distributed_new_price(latest_price: any, daily_vol: any):
    return latest_price * ( 1 + np.random.normal(0,daily_vol))

def run_simulations(prices: pd.Series) -> Tuple[pd.DataFrame, any]:
    returns = prices.pct_change() # Calculate daily returns. Financial models often work with returns rather than absolute prices. 

    latest_price = prices.iloc[-1] # Future projections are typically based on the most recent data point.
    num_simulations=1000
    num_days=200
    daily_vol = returns.std() # Volatility is calculated as the standard deviation of these returns. It represents the typical daily fluctuation of the stock price.

    simulation_list = []

    # Each simulation here is one possible future path
    for _ in range(num_simulations):
        count=0
        price_series=[]

        price = get_randomly_distributed_new_price(latest_price, daily_vol)
        price_series.append(price)
    
        # Each simulation here projects each scenario above forward in time
        for _ in range(num_days):
            if count == num_days - 1:
                break

            price = price_series[count] * ( 1 + np.random.normal(0,daily_vol))
            price_series.append(price)
            count+= 1

        simulation_list.append(pd.Series(price_series))

    simulation_df = pd.concat(simulation_list, axis=1)
    simulation_df.columns = range(num_simulations)

    return simulation_df, latest_price

def monte_carlo_simulations():
    prices = load_data()
    simulation_data, latest_price = run_simulations(prices)

    simulation_last_day = simulation_data.iloc[199,:]
    simulation_last_day_mean = simulation_last_day.mean()

    # Plot histogram
    plt.hist(simulation_last_day, bins=10, color='b')
    plt.axvline(x = simulation_last_day_mean, linewidth=2)
    plt.show()

    # Plot future scenarios
    plt.figure(figsize=(19,8))
    plt.plot(simulation_data, linewidth=2)
    plt.title('Monte Carlo Simulation IBM stocks')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.axhline(y=latest_price, color='r', linestyle='-')
    plt.show()

if __name__ == "__main__":
    monte_carlo_simulations()