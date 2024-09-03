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

#TO DO:
#Data Processing: Clean and preprocess the collected data. This may include handling missing values, 
#normalizing data, and feature extraction to prepare the data for analysis. 
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
TRAIN_START = '2008-12-01'
TRAIN_END = '2010-12-01'
FEATURES = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir',
            'WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm',
            'Temp9am','Temp3pm','RainToday','RainTomorrow'] 
NAN_STRATEGY = 'ffill'
SPLIT_METHOD = 'ratio'
TEST_SIZE = 0.2
SPLIT_DATE = '2009-12-01'
SCALE = 'False'
FEATURE_RANGE=(0,1)
FILE_PATH = 'data/weatherAUS.csv' #to read the unprocesses data 

# save processes data file path
date_now = time.strftime("%Y-%m-%d")
data_filename = os.path.join("data",f"{COMPANY}_{date_now}.csv") #processed data
file_name = f"{date_now}_{COMPANY}-{SCALE}"


#------------------------------------------------------------------------------
# load and process data function
#------------------------------------------------------------------------------
def load_and_process_dataset(start_date,end_date,features,nan_strategy = 'ffill',
                             file_path = None, split_method = "ratio", test_size=0.2,
                             split_date=None,random_state=None, scale=False, feature_range=(0,1)):
    
    #to ensure that the start date and end date are called as objects and not text 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    #load data from a file 
    df = pd.read_csv('data/weatherAUS.csv')

    #this is to select the desired features of the data
    data = df [features]

    #handling the missing values in the data set 
    if nan_strategy =="ffill":
        data.fillna(method = 'ffill', inplace = True)
    
    elif nan_strategy =="bfill":
        data.fillna(method = 'bfill', inplace = True)

    elif nan_strategy =="drop":
        data.dropna(inplace = True)
    else:
        raise ValueError("Invalid NaN handling strategy. Choose from 'ffill', 'bfill', or 'drop'.")

    #if a file path is provided save the data 
    #this code takes the file and save it to the designated file path 
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path)

    #initialize a dictionary to store the scalers if scaling is enabled 
    column_scaler = {}

    #what this function does is that its a loop that iterates each column that is 
    # specified in the features list 
    #is then creates a scaler for each feature a new MinMaxScaler is created. the feature_range will specifies
    #the range to which the data is scaled to.
    #the data is then reshaped into a 2D array with one column and multiple rows
    #the fit transform is applied to the data which scales the data and replaces the original values in the dataset 
    #The values is then sotred in the column_Scaler dictionary, this allows it to reverse the transformation later if needed
    if scale:
        for column in features:
            scaler = MinMaxScaler(feature_range=feature_range)
            data[column] = scaler.fit_transform(data[column].values.reshape(-1,1))
            column_scaler[column] = scaler

    #split the processed data into training and testing data sets.
    if split_method == 'ratio':
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    elif split_method == 'random':
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("Invalid split method. choose from 'ratio' or 'random'.")
    

    return train_data, test_data
#------------------------------------------------------------------------------
# Data Analysis
#------------------------------------------------------------------------------
#data analysis
pd.set_option('max_columns', 26)
df = pd.read_csv('data/weatherAUS.csv')
print("\n",df.head(10))#show 
print(df.columns) #show what columns are there 
df.info()         #show columns info


#graphs to use histogram, correlation Matrix, scatter plot graph, bar graph 
# 
