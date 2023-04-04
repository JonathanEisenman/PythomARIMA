import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


# Load the data from a CSV file
df = pd.read_csv('data.csv')

# Convert the date column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Set the date column as the index
df.set_index('date', inplace=True)

# Resample the data to a daily frequency
df = df.resample('D').sum()

# Fill any missing values with zeros
df.fillna(0, inplace=True)


# Plot the time series
plt.plot(df)
plt.xlabel('Date')
plt.ylabel('Variable')
plt.title('Time Series')
plt.show()


# Define the ARIMA model parameters
p = 2  # Order of the AR term
d = 1  # Order of differencing
q = 2  # Order of the MA term

# Create the ARIMA model
model = ARIMA(df, order=(p, d, q))

# Fit the model to the data
results = model.fit()



# Make predictions for the next 30 days
forecast = results.forecast(steps=30)

# Plot the predicted values
plt.plot(df.index, df, label='Observed')
plt.plot(forecast.index, forecast[0], label='Forecast')
plt.xlabel('Date')
plt.ylabel('Variable')
plt.title('Time Series Forecast')
plt.legend()
plt.show()
