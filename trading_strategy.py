'''
다양한 분석기법을 활용하여, 성공적인 시스템 트레이딩을 하기 위한 학습 과정이다.
'''

# basic library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from datetime import timedelta

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


# Step 0. Basic

def get_code(input_name):
    stock_df = pd.read_csv()
    code = stock_df.loc[stock_df['Name'] == input_name]['Symbol'][0]

    return code


# Step 1. 투자가치가 있는 종목 선정
'''
듀얼 모멘텀 기법을 활용하여 투자 가치가 있다고 판단되는 상위 n개의 종목을 추린다.
'''
class DualMomentum:
    def __init__(self):
        '''
        FinanceDataReader 패키지에서 제공해주는 stock list를 받아서, krx 상장 종목 전체에 대한 상대 모멘텀을 계산하고,
        상대 모멘텀 값을 기준으로 투자 가치가 높다고 판단되는 종목을 추리기 위함이다.
        종목코드 상에 숫자만으로 구성되지 않은 종목 코드의 경우 에러를 동반하여 우선적으로 정규표현식을 통해 종목을 걸러내었다.
        '''
        self.stock_df = fdr.StockListing('KRX')[['Symbol', 'Name']]
        self.stock_df = self.stock_df.loc[self.stock_df['Symbol'].apply(lambda x: re.search('[^0-9]', x) == None)]

    # 상대 모멘텀 계산
    def get_rltv_momentum(self, start_date, end_date = None, stock_count = 300):
        '''
        상대 모멘텀을 계산하기 위한 코드
        - start_date : 상대 모멘텀을 구할 시작일자
        - end_date : 상대 모멘텀을 구할 종료일자
        - stock_count : 상대 모멘텀을 구할 종목수, 디폴트 값은 300
        '''
        rows = []
        columns = ['Symbol', 'Name', 'old_price', 'new_price', 'earning_rate']
        if end_date == None:
            end_date = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')

        for (symbol, name) in self.stock_df.values:
            df = fdr.DataReader(symbol, start_date, end_date).sort_index()
            if df.empty:
                continue

            old_price = df['Close'].values[0]
            new_price = df['Close'].values[-1]

            er = (new_price / old_price -1) * 100

            rows.append([symbol, name, old_price, new_price, er])

        df = pd.DataFrame(rows, columns = columns)
        df = df.sort_values(by = 'earning_rate', ascending = False)
        df = df.head(stock_count)
        df.index = pd.Index(range(stock_count))

        print(f'Absolute momentum ({start_date} ~ {end_date}) : {df['er'].mean():.2f%}')

        return df

    # def get_abs_momentum(self, rltv_momentum, start_date, end_date):
    #     '''
    #     절대 모멘텀을 계산하기 위한 코드
    #     * 절대 모멘텀 : 특정 기간 동안 상대 모멘텀에 투자했을 때의 평균 수익률
    #     * 즉, 상대 모멘텀 계산에서 지정해준 stock_count 값만큼의 종목에 대해서만 수익률을 계산하는 것
    #     - start_date : 매수일
    #     - end_data : 매도일
    #     '''
    #     stock_list = rltv_momentum[['Symbol', 'Name']].values.tolist()
    #
    #     rows = []
    #     columns = ['Symbol', 'Name', 'old_price', 'new_price', 'earning_rate']
    #     for (symbol, name) in stock_list:
    #         df = fdr.DataReader(symbol, start_date, end_date).sort_index()
    #         if df.empty:
    #             continue
    #
    #         old_price = df['Close'].values[0]
    #         new_price = df['Close'].values[-1]
    #
    #         er = (new_price / old_price - 1) * 100
    #
    #         rows.append([symbol, name, old_price, new_price, er])
    #
    #     df = pd.DataFrame(rows, columns = columns)
    #     df = df.sort_values(by = 'earning_rate', ascending = False)
    #
    #     return df


# Step 2. 종목들에 대해서 추가 feature 생성

def get_features(input_df):
    '''
    볼린저 밴드 값을 구하기 위한 함수
    * 볼린저 밴드 : 주가의 20일 이동 평균선을 기준으로, 상대적인 고점을 나타내는 상단 밴드와 상대적인 저점을 나타내는 하단 밴드로 구성
    * %b : 주가가 볼린저 밴드 어디에 위치하는지를 나타내는 지표
    * 밴드폭 : 밴드의 너비를 수치로 나타낸 것. 추세의 시작과 끝을 포착하는 역할
    '''
    df = input_df.copy()

    df['MA20'] = df['Close'].rolling(window = 20).mean() # 20일간의 이동평균
    df['std'] = df['Close'].rolling(window = 20).std()
    df['upper_band'] = df['MA20'] + (df['std'] * 2)
    df['lower_band'] = df['MA20'] - (df['std'] * 2)
    df['PB'] = (df['Close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
    df['band_width'] = (df['upper_band'] - df['lower_band']) / df['MA20'] * 100

    ### 추세 추종 매매기법을 위한 지표
    # MONEY FLOW INDEX (현금흐름지표) - 추세를 확인하는 용도로 사용할 수 있다.
    # 매수 : (%b > 0.8) & (MFI > 80)
    # 매도 : (%b < 0.2) & (MFI < 20)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['PMF'] = 0
    df['NMF'] = 0
    for i in range(len(df['Close']) -1):
        if df['TP'].values[i] < df['TP'].values[i+1]:
            df['PMF'].values[i+1] = df['TP'].values[i+1] * df['Volume'].values[i+1]
            df['NMF'].values[i+1] = 0
        else:
            df['NMF'].values[i+1] = df['TP'].values[i+1] * df['Volume'].values[i+1]
            df['PMF'].values[i+1] = 0

    df['MFR'] = df['PMF'].rolling(window = 10).sum() / df['NMF'].rolling(window = 10).sum()
    df['MFI10'] = 100 - 100 / (1 + df['MFR'])

    ### 반전 매매기법을 위한 지표
    # INTRADAY INTENSITY (일중강도) - 거래 범위에서 종가의 위치를 토대로 주식 종목의 자금 흐름을 설명
    # 매수 : (%b < 0.05) & (II% > 0)
    # 매도 : (%b > 0.95) & (II% < 0)
    df['II'] = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    df['IIP21'] = df['II'].rolling(window = 21).sum() / df['Volume'].rolling(window = 21).sum() * 100
    df = df.dropna()

    ### 삼중창 매매 시스템을 위한 지표
    # 1. 추세 판단 지표
    df['EMA60'] = df['Close'].ewm(span = 60).mean()
    df['EMA130'] = df['Close'].ewm(span = 130).mean()
    df['MACD'] = df['EMA60'] - df['EMA130']
    df['signal_line'] = df['MACD'].ewm(span = 45).mean()
    df['MACD_hist'] = df['MACD'] - df['signal_line']

    # 2. 오실레이터 확인 지표
    df['ndays_high'] = df['High'].rolling(window = 14, min_periods = 1).max()
    df['ndays_low'] = df['Low'].rolling(window = 14, min_periods = 1).min()
    df['fast_k'] = (df['Close'] - df['ndays_low']) / (df['ndays_high'] - df['ndays_low']) * 100
    df['slow_d'] = df['fast_k'].rolling(window = 3).mean()


    df = df[19:]

    return df


# Step 3. 투자 전략 (삼중창 매매 시스템)

def tr_strategy(input_df):
    df = input_df.copy()

    # 매매기법 별 매수/매도 사인
    df['bollinger_trend_sign'] = 0
    df['bollinger_reversal_sign'] = 0
    df['triple_screen_sign'] = 0

    for i in range(1, len(df['Close'])):
        if (df['PB'].values[i] > 0.8) & (df['MFI10'].values[i] > 80):
            df['bollinger_trend_sign'].values[i] = 1
        elif (df['PB'].values[i] < 0.2) & (df['MFI10'].values[i] < 20):
            df['bollinger_trend_sign'].values[i] = -1
        else:
            pass

        if (df['PB'].values[i] < 0.05) & (df['IIP21'].values[i] > 0):
            df['bollinger_reversal_sign'].values[i] = 1
        elif (df['PB'].values[i] > 0.95) & (df['IIP21'].values[i] < 0):
            df['bollinger_reversal_sign'].values[i] = -1
        else:
            pass

        if (df['EMA130'].values[i-1] < df['EMA130'].values[i]) & (df['slow_d'].values[i-1] >= 20) & (df['slow_d'].values[i] < 20):
            df['triple_screen_sign'].values[i] = 1
        elif (df['EMA130'].values[i-1] > df['EMA130'].values[i]) & (df['slow_d'].values[i-1] <= 80) & (df['slow_d'].values[i] > 80):
            df['triple_screen_sign'].values[i] = -1

    return df


# Step 4. Plotting

def plotting(input_df):
    df = input_df.copy()
    df['number'] = df.index.map(mdates.date2num)
    ohlc = df[['number', 'Open', 'High', 'Low', 'Close']]

    plt.figure(figsize = (16, 40))

    # --------------------------------------------------------------------------
    p1 = plt.subplot(5, 1, 1)
    candlestick_ohlc(p1, ohlc.values, width = 0.6, colorup = 'red', colordown = 'blue')
    p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.plot(df['number'], df['EMA130'], color = 'c', label = 'EMA130')

    plt.plot(df['number'], df['upper_band'], 'r--', label = 'Upper band')
    plt.plot(df['number'], df['lower_band'], 'c--', label = 'Lower band')
    plt.fill_between(df['number'], df['upper_band'], df['lower_band'], color = '0.9')

    plt.grid()
    plt.legend(loc = 'upper right')

    # --------------------------------------------------------------------------
    plt.subplot(5, 1, 2)
    plt.plot(df['number'], df['PB'] * 100, color = 'b', label = '%B x 100')
    plt.plot(df['number'], df['MFI10'], 'g--', label = 'MFI(10 day)')
    plt.yticks([-20, 0, 20, 40, 60, 80, 100, 120])

    plt.grid()
    plt.legend(loc = 'upper right')

    # --------------------------------------------------------------------------
    plt.subplot(5, 1, 3)
    plt.bar(df['number'], df['IIP21'], color = 'g', label = 'II% 21day')

    plt.grid()
    plt.legend(loc = 'upper right')

    # --------------------------------------------------------------------------
    plt.subplot(5, 1, 4)
    plt.bar(df['number'], df['MACD_hist'], color = 'm', label = 'MACD-Hist')
    plt.plot(df['number'], df['MACD'], color = 'b', label = 'MACD')
    plt.plot(df['number'], df['signal_line'], 'g--', label = 'MACD-Signal')

    plt.grid()
    plt.legend(loc = 'upper right')

    # --------------------------------------------------------------------------
    plt.subplot(5, 1, 5)
    plt.plot(df['number'], df['fast_k'], color = 'c', label = '%K')
    plt.plot(df['number'], df['slow_d'], color = 'k', label = '%D')
    plt.yticks([0, 20, 80, 100])

    plt.grid()
    plt.legend(loc = 'upper right')

    plt.show()
