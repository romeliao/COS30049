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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, InputLayer


#TO DO:

#Data Analysis: Analyze the preprocessed data to uncover patterns and insights. 
#Utilize statistical methods and visualizations to better understand the data 
#and inform the selection of the machine learning model. 
#Model Selection: Choose an appropriate machine learning model based on the data 
#and the problem you are addressing. Consider models such as regression, classification, 
#clustering, etc. 


#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
COMPANY = "Weather Prediction"
#TRAIN_START = '2008-12-01'
#TRAIN_END = '2010-12-01'
FILL_METHOD = 'ffill'
SPLIT_METHOD = 'ratio'
DROP_THRESHOLD = 0.7
FEATURES = ['Location', 'MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir']
NORMALIZE_COLUMN = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir',
            'WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm',
            'Temp9am','Temp3pm','RainToday','RainTomorrow']

FILE_PATH = df = pd.read_csv('data/weatherAUS.csv') 

# save processes data file path
date_now = time.strftime("%Y-%m-%d")
data_filename = os.path.join("data",f"{COMPANY}_{date_now}.csv") #processed data
file_name = f"{date_now}_{COMPANY}"

#------------------------------------------------------------------------------
# format lables in the dataset 
#------------------------------------------------------------------------------
def encode_categorical(df, categorical_columns):
    # Initialize LabelEncoder for categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings for consistent encoding
        label_encoders[col] = le  # Store the label encoders for possible inverse_transform later
    return df, label_encoders

#------------------------------------------------------------------------------
# load and process data function
#------------------------------------------------------------------------------
def process_weather_data(file_path,fill_method = 'ffill', drop_threshold = 0, normalize_columns=None, features=None):
    df = pd.read_csv('data/weatherAUS.csv') 

    #handling missing values
    if fill_method =='mean':
        df.fillna(df.mean(), inplace=True)
    elif fill_method == 'ffill':
        df.fillna(method = 'ffill', inplace=True)
    elif fill_method =='drop':
        df.dropna(inplace=True)
    else:
        raise ValueError("Invalid fill method. Choose 'mean', 'ffill', or 'drop'.")

    # Drop columns with more missing data than the threshold
    df.dropna(thresh=int(drop_threshold * len(df)), axis=1, inplace=True)

    if features:
        features_to_encode = [col for col in features if col not in 
                              ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday','RainTomorrow']]
        df, label_encoders = encode_categorical(df, features_to_encode)

    if normalize_columns:
        continuous_feats = [col for col in normalize_columns if col not in features]   

        numeric_columns = [col for col in normalize_columns if col in df.columns]

    # Normalize specified numerical columns if provided
    if normalize_columns:
        # Filter only numerical columns for normalization
        numeric_columns = df[normalize_columns].select_dtypes(include=[np.number]).columns.tolist()

        # Normalize numerical columns
        if numeric_columns:
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        else:
            print("No numerical columns to normalize.")

    # Filter the DataFrame to include only the columns specified in DISPLAY_COLUMNS
    filtered_df = df[FEATURES]

    return filtered_df
#------------------------------------------------------------------------------
# Call Processing data function 
#------------------------------------------------------------------------------
processed_df = process_weather_data(file_path=FILE_PATH,fill_method=FILL_METHOD,
    drop_threshold=DROP_THRESHOLD,normalize_columns=NORMALIZE_COLUMN,features=FEATURES
)

#------------------------------------------------------------------------------
# Data Analysis
#------------------------------------------------------------------------------
#data analysis

print(processed_df.head(20))

print("\n",df.columns)                      #show what columns are there 
df.info()                                   #show columns info


