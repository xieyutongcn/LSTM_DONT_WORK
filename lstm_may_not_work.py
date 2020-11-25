#%%
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# %%
# Download the data
df = web.DataReader('SPY', data_source = 'yahoo', start = '1990-01-01', end = '2020-10-01') # get the stock price data
data = df.filter(['Close']) # create a new dataframe with only the 'Close' column

#%%
dataset = data.values # convert the dataframe to a numpy array

training_data_len = math.ceil(len(dataset)*0.9)
pstep = 30 # This is how far into the futhre I want to predict. 


# Scale the data for machine learning
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data  = scaler.fit_transform(dataset)


# Create the training data 
# Create the scaled training data
train_data = scaled_data[:training_data_len, :]
# Split the data into x_train and y_train
x_train = []
y_train = []

# This look is going to create many x_train-by-y_train pairs. Within each pair, there are
# 180 price data points in x_train and 1 data point in y_train. We are using past 
# 180 (selected using train_data[i-180:i,0]) prices to predict the next 1 (selected 
# using train_data[i,0]). 
for i in range(180 + pstep, len(train_data)):
    x_train.append(train_data[i-180 - pstep:i - pstep,0]) # This is selecting rows i - 180 to i not including i; 0 means the first column. 
    y_train.append(train_data[i,0])
    # These created objects are lists. 


# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train) # This is converting lists into np arrays
x_train.shape # This tells us what the shape of the training data is    

# Reshape the data
# The reason for reshaping: LSTM expects the data to be 3-dimentional -- Number of Samples by Number of time stpes by Number of Features. 
# Number of sampels -- 1702 -- range(180, len(train_data))
# Number of times steps -- 180 -- using 180 obs to predict next 1
# Number of features -- 1 -- only closing price as input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# Build the LSTM model
model = Sequential()
model.add(LSTM(180, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(180, return_sequences = False))
model.add(Dense(180))
model.add(Dense(1))

# Compile the model
# The loss function can be 
# - mean_squared_error
# - mean_squared_logarithmic_error
# - mean_absolute_error
# - categorical_crossentropy
# - 
# - 

model.compile(optimizer = 'adam', loss = 'mean_absolute_error')


# Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)


# %%
# Apply compiled value to test dataset
# Create a new array containing scaled values from index 1702 to the last
test_data = scaled_data[training_data_len - 180 - pstep: , :]
test_data
# Create x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(180 + pstep, len(test_data)):
    x_test.append(test_data[i - 180 - pstep:i - pstep, 0])
x_test = np.array(x_test)

# convert to np array and reshape
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# %%
# Get the model predicions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize = (16,8))
plt.title('Model')
#plt.xlable('date')
#plt.ylable('price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
plt.show()

# %%

signal = data[training_data_len - pstep:data.shape[0]-pstep]

future_price = []
real_future_price = []
for i in range(0, len(signal)):
    future_price.append(predictions[i, 0])
    real_future_price.append(y_test[i, 0])

signal['future_price'] = future_price
signal['real_future_price'] = real_future_price

# buy = []
# sell = []
# for i in range(0, len(signal)):
#     if signal['Close'][i] < signal['future_price'][i]:
#         buy.append(1)
#     else: 
#         buy.append(0)
#     if signal['Close'][i] > signal['future_price'][i]:
#         sell.append(1)
#     else:
#         sell.append(0)

# signal['buy'] = buy
# signal['sell'] = sell
# signal
# %%
# signal = signal[:300]
plt.figure(figsize = (16,8))
plt.title('Model')
#plt.xlable('date')
#plt.ylable('price')
plt.plot(signal['Close'])
plt.plot(signal['future_price'])
plt.plot(signal['real_future_price'])
#plt.scatter(signal.loc[signal['buy'] ==1 , 'Date'].values,signal.loc[signal['buy'] ==1, 'Close'].values, label='skitscat', color='green', s=25, marker="^")
#plt.scatter(signal.loc[signal['sell'] ==1 , 'Date'].values,signal.loc[signal['sell'] ==1, 'Close'].values, label='skitscat', color='red', s=25, marker="v")
plt.legend(['Current Price', 'Predicted Future', 'Real Future'], loc = 'lower right')
plt.show()
# %%
