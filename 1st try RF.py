import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Load CSV file into a DataFrame
df = pd.read_csv('btc2020.csv')

# Display the first few rows of the DataFrame
#print(df.head())

#print(df.info())

#check is any missing data
#print(df.isnull().sum())

#make sure data types are appropriate in each column
#print(df.dtypes)

#Convert date to Datetime: You can convert the date column to a datetime type, which will allow you to perform time-based operations more easily
df['date'] = pd.to_datetime(df['date'])

#Convert unix to Datetime (Optional): If you want to use the UNIX timestamp as a datetime index, you can also convert it to a datetime object. (not sure on this yet come back)
#df['unix'] = pd.to_datetime(df['unix'], unit='s')  # unit depends on the granularity: 's' for seconds, 'ms' for milliseconds

#Set Date as Index 
df.set_index('date', inplace=True)

#a summary of the data 
#print(df.describe())

# Using OHLC columns as features
X = df[['open', 'high', 'low', 'close']]

# Assuming you want to predict the next minute's closing price
y = df['close'].shift(-1)

# Dropping the last row as it will have a NaN value after shifting
X = X[:-1]
y = y[:-1]

# Determining the split index for 70-30 split
split_index = int(len(df) * 0.7)

# Splitting the data chronologically
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Ensure the test set doesn't start with the last value of the training set
assert X_train.iloc[-1].name != X_test.iloc[0].name  # This line ensures that there's no overlap in timestamps

# Training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print("Code is running")

# Extract the dates corresponding to your test set
test_dates = X_test.index

# Print the range of dates in the test set
print(test_dates.min(), test_dates.max())

# Print the length of predictions and y_test (both should be same)
#print(len(predictions))
#print(len(y_test))

#plt.figure(figsize=(10, 5))
#plt.plot(y_test.index, y_test.values, label='Actual')
#plt.plot(y_test.index, predictions, label='Predicted')
#plt.legend()
#plt.title('Actual vs. Predicted Closing Prices')
#plt.show()