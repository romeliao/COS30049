import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import mplfinance as mpf
import seaborn as sns

import os 
import time 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, InputLayer


df = pd.read_csv('data/weatherAUS.csv')

