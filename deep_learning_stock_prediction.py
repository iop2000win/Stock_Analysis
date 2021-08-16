'''
기본적으로 시계열 데이터인 주가에 대해서, 딥러닝을 활용해보는 기본적인 소스 코드이다.
매우 기초적인 수준의 코드로, 시계열 데이터에 대해서 이런 모델을 사용하고, 이런 방식으로 접근한다는 개념 정도만 포함하고 있는 코드이다.
'''

# basic library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from matplotlib import font_manager, rc
plt.rcParams['font.family'] = 'Malgun Gothic'
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# stock related library
import FinanceDataReader as fdr
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc

# ML related library
import sklearn as sc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ------------------------------------------------------------------------------

code_list = ['149980'] # input을 어떻게 받을 것인지??

df = fdr.DataReader(code_list[0], '2019')
display(df.head())

# data preprocessing
minmax = MinMaxScaler().fit(df)
df = minmax.transform(df)

X = df[['Open', 'High', 'Low', 'Volume', 'Close']].values.tolist()
y = df[['Close'].values.tolist()

# data set: 이전 10일 동안의 Open - High - Low - Volume - Close 값을 통해 다음 날 Close 값을 예측
X_data = []
y_data = []
window_size = 10

for i in range(len(y) - window_size):
    _x = X[i : i + window_size]
    _y = y[i + window_size]
    X_data.append(_x)
    y_data.append(_y)


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, shuffle = False, random_state = 23)
print(f'X_train.shape: {X_train.shape}\ny_train.sahpe: {y_train.shape}\nX_test.shape: {X_test.shape}\y_test.shape: {y_test.shape}')


# modeling
model = Sequentail()

model.add(LSTM(units = 10, activation = 'relu', return_sequences = True, input_shape = (window_size, 5)))
model.add(Dropout(0.1))
model.add(LSTM(units = 10, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 1))

model.summary()


# train and prediction
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 60, batch_size = 30)

y_pred = model.predict(X_test)


# Compare with graph
plt.figure()

plt.plot(y_test, color = 'red', label = 'real stock price')
plt.plot(y_pred, color = 'blue', label = 'predicted stock price')

plt.title('stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()

plt.show()


# 다음 날의 예측 종가
print('''tomorrow's price : ''', df['Close'][-1] * y_pred[-1] / y['Close'][-1])
