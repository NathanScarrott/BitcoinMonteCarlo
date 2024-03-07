import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Download Bitcoin data
btc = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01')

# Calculate daily returns and drop the first NA value
daily_returns = btc['Close'].pct_change().dropna()

# Set the number of simulations and the forecast horizon
num_simulations = 1000
forecast_horizon = 365  # days

# Get the last closing price
last_price = btc['Close'].iloc[-1]

# Initialize an empty DataFrame to hold all simulation results with pre-allocation
simulation_df = pd.DataFrame(index=range(forecast_horizon), columns=range(num_simulations))

# Run simulations using the Pareto distribution
for i in range(num_simulations):
    # The 'scale' parameter would be related to the scale of your returns
    # The 'shape' parameter controls the shape of the distribution
    scale = daily_returns.std()
    shape = 3  # This is an example value; you might want to fit this to your data
    
    # Generate random returns using Pareto distribution
    random_returns = (np.random.pareto(shape, size=forecast_horizon) + 1) * scale
    price_series = last_price * (1 + random_returns).cumprod()

    simulation_df[i] = price_series

# Plot the simulation results
plt.figure(figsize=(14,7))
plt.plot(simulation_df, color='blue', alpha=0.05)

# Calculate the mean and median of the simulations
mean_path = simulation_df.mean(axis=1)
median_path = simulation_df.median(axis=1)

# Calculate the 95% confidence interval bounds
lower_bound = simulation_df.quantile(0.025, axis=1)
upper_bound = simulation_df.quantile(0.975, axis=1)

# Plot the mean and median paths
plt.plot(mean_path, label='Mean Simulated Path', color='red', linewidth=2)
plt.plot(median_path, label='Median Simulated Path', color='green', linewidth=2)

# Fill between the lower and upper bounds for the 95% confidence interval
plt.fill_between(simulation_df.index, lower_bound, upper_bound, color='orange', alpha=0.3, label='95% Confidence Interval')

# Add labels and title
plt.title('Monte Carlo Simulation for Bitcoin using Pareto Distribution')
plt.xlabel('Days')
plt.ylabel('Simulated Price')
plt.legend()
plt.show()
