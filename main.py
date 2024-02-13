import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras. layers import LSTM, Dense

# Load Data Set
url = "/content/Avg_temperatur.csv"  # Replace with the actual path to your dataset
df = pd.read_csv(url, delimiter=',')

# Combine year, month, and day columns into a datetime column
df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MO'].astype(str) + '-' + df['DY'].astype(str))
# Drop the original year, month, and day columns
df = df.drop(['YEAR', 'MO', 'DY'], axis=1)

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)
print(df)
temperatures = df["T2M"].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
normalized_temps = scaler.fit_transform(temperatures)

train_size = int(len(normalized_temps) * 0.8)
train_data = normalized_temps[:train_size]
test_data = normalized_temps[train_size:]

def create_lstm_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)
look_back = 100 # Number of previous time steps to use as input
train_x, train_y =create_lstm_dataset(train_data, look_back)
test_x, test_y = create_lstm_dataset(test_data, look_back)

train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))

model = Sequential()
model.add(LSTM(units=1000,input_shape=(look_back,1)))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(train_x,train_y,epochs=50,batch_size=128,verbose=2)

train_predictions = model.predict(train_x)
test_predictions = model.predict(test_x)
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

train_rsme = np.sqrt(np.mean((train_predictions - temperatures[look_back:train_size])**2))
test_rsme = np.sqrt(np.mean((test_predictions - temperatures[train_size+look_back:])**2))
print("Train RSME : ",train_rsme)
print("Test RSME : ",test_rsme)
future_inputs = normalized_temps[-look_back:]
future_inputs = np.reshape(future_inputs,(1,look_back,1))
future_predictions = []
 # Generate 100 future predictions
for _ in range(600):
    prediction = model.predict(future_inputs)
    future_predictions.append(prediction)
    future_inputs = np.append(future_inputs[: , 1: , :], np.reshape(prediction, (1, 1, 1)), axis=1)

future_predictions = np.array(future_predictions)
future_predictions = np.squeeze(future_predictions)
future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

print("Future predictions:")
for prediction in future_predictions:
    print(prediction)

import matplotlib.pyplot as plt
# Plotting the actual temperatures
plt.plot(temperatures, label =  'Actual Temperatures')
# Plotting the future predictions
plt.plot(range(len(temperatures), len(temperatures) + len(future_predictions)), future_predictions, label='Future Predictions')
# Set plot Labels and title
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Temperature Trends')
# Add Legend
plt.legend()
# Show the plot
plt. show()
