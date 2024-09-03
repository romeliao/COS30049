import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import mplfinance as mpf

import os 
import time 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, InputLayer

#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
COMPANY = "Weather Prediction"
ROWID_START = 1
ROWID_END = 100
FEATURES = []
nan_strategy = 'ffill'
SPLIT_METHOD = 'ratio'
TEST_SIZE = 0.2
SCALE = 'True'
FEATURE_RANGE=(0,1)
FILE_PATH = './Weather Training Data.csv'

# save processes data file path
date_now = time.strftime("%Y-%m-%d")
data_filename = os.path.join("data",f"{COMPANY}_{date_now}.csv")
file_name = f"{date_now}_{COMPANY}-{SCALE}"


#------------------------------------------------------------------------------
# load and process data function
#------------------------------------------------------------------------------
def load_and_process_dataset(rowID_start,rowID_end,features,nan_strategy = 'ffill',
                             file_path = None, split_method = "ratio", test_size=0.2,
                             random_state=None, scale=False, feature_range=(0,1)):
    
    #load data from a file 
    weather_data = pd.read_csv(FILE_PATH)

    #this is to select the desired features of the data
    data = data [features]

    #handling the missing values in the data set 

    if nan_strategy =="ffill":
        data.fillna



    return 
