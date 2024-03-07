import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

################# Data Colletion and Simulation #####################

# Download Bitcoin data
btc = yf.download('BTC-USD', start='2020-01-01', end='2024-02-15')  # Adjust dates as needed

# Calculate daily returns
daily_returns = btc['Close'].pct_change()

# Set the number of simulations and the forecast horizon
num_simulations = 10000
forecast_horizon = 365 # days

# Get the last closing price
last_price = btc['Close'].iloc[-1]

# Initialize a list to hold all the price series
simulation_list = []

# Run simulations
for x in range(num_simulations):
    count = 0
    daily_volatility = daily_returns.std()
    
    price_series = [last_price]
    
    # Generate price forecast
    for y in range(forecast_horizon):
        if count == forecast_horizon - 1:
            break
        price = price_series[-1] * (1 + np.random.normal(0, daily_volatility))
        price_series.append(price)
        count += 1
    
    simulation_list.append(price_series)

# Once all simulations are done, convert the list of lists into a DataFrame
simulation_df = pd.DataFrame(simulation_list).T

################# Plotting of simulation #####################
# Plot the simulation results
plt.figure(figsize=(10,5))
plt.plot(simulation_df)
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Monte Carlo Simulation for Bitcoin')
plt.show()

# Analyze the final day's results
final_day = simulation_df.iloc[-1, :]
# Calculate potential outcomes, e.g., median, mean, confidence intervals
median_price = final_day.median()
mean_price = final_day.mean()
print(f"Median predicted price: {median_price}")
print(f"Mean predicted price: {mean_price}")

################# Calculation and Testing #####################
# Calculate the 95% confidence interval
confidence_level = 0.95
lower_bound = final_day.quantile((1 - confidence_level) / 2)
upper_bound = final_day.quantile(1 - (1 - confidence_level) / 2)

print(f"The 95% confidence interval for the Bitcoin price is: {lower_bound:.2f} - {upper_bound:.2f}")

from scipy.stats import skew

# Calculate skewness
skewness = skew(final_day)

print(f"Skewness of the distribution: {skewness}")

# Interpretation
if abs(skewness) < 0.05:  # This threshold can vary depending on the source; some use 0.5.
    print("The distribution is approximately symmetric.")
elif skewness > 0:
    print("The distribution is positively skewed.")
else:
    print("The distribution is negatively skewed.")

from scipy.stats import normaltest

# Perform the D'Agostino's K-squared test
stat, p = normaltest(final_day)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# Interpretation
alpha = 0.05
if p > alpha:
    print('The distribution looks Gaussian (fail to reject H0)')
else:
    print('The distribution does not look Gaussian (reject H0)')

# Bootstrapping confidence interval
bootstrap_samples = 10000
bootstrap_sample_size = len(final_day)
bootstrap_means = np.zeros(bootstrap_samples)

for i in range(bootstrap_samples):
    bootstrap_sample = np.random.choice(final_day, size=bootstrap_sample_size, replace=True)
    bootstrap_means[i] = np.mean(bootstrap_sample)

# Compute the 95% confidence interval from the bootstrap distribution of means
boot_lower = np.percentile(bootstrap_means, 2.5)
boot_upper = np.percentile(bootstrap_means, 97.5)

print(f"The 95% confidence interval for the Bitcoin price using bootstrapping is: {boot_lower:.2f} - {boot_upper:.2f}")



# Calculate the sample kurtosis (Fisher’s definition, the one used by pandas, subtracts 3 from the Pearson definition so that the kurtosis of a normal distribution is zero).
sample_kurtosis = final_day.kurtosis()

# Calculate the sample skewness
sample_skewness = final_day.skew()  # You've already provided this value.

# The degrees of freedom for the t-distribution can be estimated from the excess kurtosis
# Reverse calculate the degrees of freedom 'v' from the sample excess kurtosis
estimated_v = 6 / sample_kurtosis + 4

# Now generate a t-distribution with the estimated degrees of freedom
t_distribution = st.t(df=estimated_v)

# Create a Q-Q plot to visualize how well the t-distribution fits the simulated data
st.probplot(final_day, dist=t_distribution, plot=plt)
plt.title('Q-Q Plot of Simulated Prices against t-Distribution')
plt.show()


# Assuming 'final_day' contains the final day prices from all simulations
data = final_day

# Estimate Pareto parameters from data
shape, loc, scale = st.pareto.fit(data, floc=0)

# Generate a range of values from the minimum data point up to the maximum
pareto_quantiles = np.linspace(min(data), max(data), 1000)

# Calculate the CDF of the Pareto distribution with the estimated parameters
pareto_cdf = st.pareto.cdf(pareto_quantiles, shape, loc=loc, scale=scale)

# Calculate the empirical CDF
data_sorted = np.sort(data)
ecdf = np.arange(1, len(data) + 1) / len(data)

# Plot the empirical CDF against the theoretical CDF of the Pareto distribution
plt.plot(data_sorted, ecdf, label='Empirical CDF')
plt.plot(pareto_quantiles, pareto_cdf, label='Theoretical Pareto CDF')
plt.xlabel('Data points')
plt.ylabel('CDF')
plt.legend()
plt.show()

# Log-log plot for the survival function
plt.loglog(data_sorted, 1 - ecdf, marker='.', linestyle='none', label='Empirical Survival Function')
pareto_survival = 1 - st.pareto.cdf(data_sorted, shape, loc=loc, scale=scale)
plt.loglog(data_sorted, pareto_survival, label='Theoretical Pareto Survival Function')
plt.xlabel('Log of data points')
plt.ylabel('Log of Survival Function')
plt.legend()
plt.show()

"""
from scipy.stats import levy_stable

# Your data from the Monte Carlo simulations
data = final_day.values  # assuming 'final_day' is a pandas Series

# Fit the stable distribution to the data
alpha, beta, gamma, delta = levy_stable._fitstart(data)
param = levy_stable.fit(data)

# Generate a range of values for the quantile function
percentiles = np.linspace(0.01, 0.99, 100)
# Calculate the quantiles of the fitted stable distribution
fitted_quantiles = levy_stable.ppf(percentiles, *param)

# Calculate the empirical quantiles
empirical_quantiles = np.percentile(data, percentiles * 100)

# Plot the empirical quantiles against the fitted quantiles
plt.plot(empirical_quantiles, fitted_quantiles, 'o')
plt.plot(empirical_quantiles, empirical_quantiles, 'r--')  # Line y=x for reference
plt.xlabel('Empirical Quantiles')
plt.ylabel('Fitted Stable Quantiles')
plt.title('Q-Q Plot of Empirical Data against Fitted Stable Distribution')
plt.show()
"""
var_95 = np.percentile(final_day, 5)
print(f"Value at Risk (95% confidence): {var_95}")

################# Plotting #####################
# Plotting the results with the 95% confidence interval
plt.figure(figsize=(14, 7))

# Calculate mean and median paths from the simulation DataFrame
mean_path = simulation_df.mean(axis=1)
median_path = simulation_df.median(axis=1)

# Plot the mean simulated price path
plt.plot(mean_path, label='Simulated Mean Price Path', color='blue')

# Plot the median simulated price path
plt.plot(median_path, label='Simulated Median Price Path', color='green')

# [You would need to plot the actual price path here if you have the actual price data for the same period]

# Add labels and title
plt.title('Monte Carlo Simulation for Bitcoin')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()






# Download Bitcoin data
btc = yf.download('BTC-USD', start='2020-01-01', end='2024-02-15')

# Calculate daily returns
daily_returns = btc['Close'].pct_change()

# Set the number of simulations and the forecast horizon
num_simulations = 1000
forecast_horizon = 365  # days

# Get the last closing price
last_price = btc['Close'].iloc[-1]

# Initialize a list to hold all the price series
simulation_list = []

# Assume 'b' is the shape parameter for the Pareto distribution
b = 2.62  # Example value, you'll need to fit this to your data

# Run simulations using the Pareto distribution
for x in range(num_simulations):
    price_series = [last_price]
    
    # Generate price forecast using a right-tail heavy distribution like Pareto
    for y in range(forecast_horizon):
        # Scale Pareto distribution by daily volatility and shift it by the mean return
        price_change = (np.random.pareto(b) + 1) * daily_returns.mean() * daily_volatility
        price = price_series[-1] * (1 + price_change)
        price_series.append(price)
    
    simulation_list.append(price_series)

# Once all simulations are done, convert the list of lists into a DataFrame
simulation_df = pd.DataFrame(simulation_list).T

# Plot the simulation results
plt.figure(figsize=(10,5))
plt.plot(simulation_df, alpha=0.1, color='blue')  # Set a low alpha to see the density of the paths
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Monte Carlo Simulation for Bitcoin using Pareto Distribution')
plt.show()

# Analyze the final day's results
final_day = simulation_df.iloc[-1, :]

# Calculate mean and median paths
mean_path = simulation_df.mean(axis=1)
median_path = simulation_df.median(axis=1)

# Plot the mean and median simulated price paths
plt.figure(figsize=(14,7))
plt.plot(mean_path, label='Simulated Mean Price Path', color='blue')
plt.plot(median_path, label='Simulated Median Price Path', color='green')
plt.title('Mean and Median of Simulated Bitcoin Prices !!!')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()


