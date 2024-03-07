import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def run_monte_carlo_simulation(start_price, daily_volatility, forecast_horizon, num_simulations):
    # Initialize a list to hold all the price series
    simulation_list = []

    # Run simulations
    for x in range(num_simulations):
        price_series = [start_price]
        for _ in range(forecast_horizon):
            price = price_series[-1] * (1 + np.random.normal(0, daily_volatility))
            price_series.append(price)
        simulation_list.append(price_series)

    return pd.DataFrame(simulation_list).T

# Use historical data up to the start of the backtest to get last price and volatility
historical_data = yf.download('BTC-USD', start='2020-01-01', end='2023-11-01')
historical_volatility = historical_data['Close'].pct_change().std()
last_price_before_backtest = historical_data['Close'].iloc[-1]

# Run simulation for a 3-month forecast horizon with 1000 simulations
forecast_horizon = 90  # 3 months ~ 90 days
num_simulations = 1000
simulated_prices = run_monte_carlo_simulation(last_price_before_backtest, historical_volatility, forecast_horizon, num_simulations)

# Download actual Bitcoin data for the backtest period
btc_backtest = yf.download('BTC-USD', start='2023-11-01', end='2024-02-01')
actual_prices = btc_backtest['Close']

# Ensure the actual prices have the same number of data points as the simulated data
# If the actual data is less than the simulation, it could be due to non-trading days
if len(actual_prices) < forecast_horizon + 1:
    # If actual prices are less, pad the actual prices array with NaNs at the end
    padded_actual_prices = np.pad(actual_prices.values, (0, forecast_horizon + 1 - len(actual_prices)), 'constant', constant_values=np.NaN)
else:
    # If actual prices are more, truncate the array to the length of the simulation
    padded_actual_prices = actual_prices.values[:forecast_horizon + 1]

# Plotting the results with the 95% confidence interval
plt.figure(figsize=(14, 7))

# Calculate 95% confidence interval bounds
lower_bounds = np.percentile(simulated_prices, 2.5, axis=1)
upper_bounds = np.percentile(simulated_prices, 97.5, axis=1)

# Calculate mean and median paths
mean_path = simulated_prices.mean(axis=1)
median_path = np.median(simulated_prices, axis=1)

# Plot the mean simulated price path
plt.plot(mean_path, label='Simulated Mean Price Path', color='blue')

# Plot the median simulated price path
plt.plot(median_path, label='Simulated Median Price Path', color='green')

# Plot the actual price path
plt.plot(padded_actual_prices, label='Actual Price Path', color='orange', alpha=0.7)

# Fill between the lower and upper bounds for the 95% confidence interval
plt.fill_between(range(forecast_horizon + 1), lower_bounds, upper_bounds, color='blue', alpha=0.1, label='95% Confidence Interval')

# Add labels and title
plt.title('Monte Carlo Simulation vs Actual Data (3-Month Backtest)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

