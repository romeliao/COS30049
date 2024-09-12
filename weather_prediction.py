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
FEATURES = ['Location', 'MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','RainToday','RainTomorrow']
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
        numeric_columns = df.select_dtypes(include=[np.number]).columns.intersection(normalize_columns)

        # Normalize numerical columns
        if not numeric_columns.empty:
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
# print("\n",df.columns)                      #show what columns are there 
# df.info()                                   #show columns info
#------------------------------------------------------------------------------
print("\n")
pd.set_option('display.max_columns', None)
print(processed_df.head(10))

#------------------------------------------------------------------------------
# Histogram function
#------------------------------------------------------------------------------
def plot_histogram(df, output_dir ='Diagrams', figsize=(12,18)):

    #create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Get the continuous 
    continuous_columns = [col for col in df.columns if df[col].dtype != object]

    #set up the subplot grid 
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=figsize)
    axes = axes.reshape(-1)

    #plot a histogram for each of the columns called 
    for i , col in enumerate(continuous_columns):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')

    #adjust layout and save the figure 
    fig.tight_layout(pad=2.0)
    plt.suptitle("Histogram of Columns", fontsize = 16, y =1.02)

    #save the figure 
    output_path = os.path.join(output_dir, 'histograms.png')
    plt.savefig(output_path, facecolor='white',dpi=100)
    plt.show()
#------------------------------------------------------------------------------
# Bar graph function function for rainy days
#------------------------------------------------------------------------------
def anayze_rainy_days_by_location(df, output_dir='Diagrams'):
    #ensure the RainToday column is in binary format (1 for 'Yes', 0 for 'No')
    df['RainToday'] = df['RainToday'].apply(lambda x: 1 if x== 'Yes' else 0)

    #group by location an sum the rainToday column to get the number of rainy days per location 
    df_rain_by_location = df.groupby(by='Location').sum()
    df_rain_by_location = df_rain_by_location[['RainToday']]

    #plotting the bar graph 
    plt.figure(figsize=(8,12))
    sns.barplot(x='RainToday', y=df_rain_by_location.index,
                data=df_rain_by_location.sort_values('RainToday', ascending=False),
                orient='h', palette='crest')
    plt.xlabel('Number of Rainy Days')
    plt.title('Rainy Days by Location')
    plt.tight_layout()

    #save the figure 
    output_path = os.path.join(output_dir,'rainy_days_by_loc.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

#------------------------------------------------------------------------------
# Heatmap function
#------------------------------------------------------------------------------
def plot_heatmap(df, output_dir='Diagrams', figsize=(12,10), focus_cols=None):
    if focus_cols:
        df = df[focus_cols]

    corr_matrix = df.corr()

    #plotting the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    
    #save heatmap diagram to the right path
    plt.title('Correlation Heatmap')
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

#------------------------------------------------------------------------------
# scatter plot diagram
#------------------------------------------------------------------------------
def scatter_plot(df,x_col,y_col ,output_dir='Diagrams', figsize=(8,6), hue=None):

    #create directory if it doesn't exist 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #plotting scatter plot 
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col,hue=hue, palette='viridis')
    plt.title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout

    #save scatter plot diagram to the right path
    output_path=os.path.join(output_dir, f'Scatter_{x_col}_VS__{y_col}.png')
    plt.savefig(output_path, facecolor='white', dpi=100)

    #display the scatter plot graph
    plt.show()



#------------------------------------------------------------------------------
# Calling graph functions
#------------------------------------------------------------------------------
plot_histogram(processed_df)                    # calling Histogram graph.
anayze_rainy_days_by_location(processed_df)     # calling graph for rainy days 
plot_heatmap(processed_df)                      # calling heat map graph for the functiong being called

#calling scatter plot diagram 
#X_col and y_col can be changed to any funtions 
#example like x_col='Sunshine', y_col='Rainfall'

#scatter plot for min temp and max temp
scatter_plot(df=processed_df, x_col='MinTemp', y_col='MaxTemp')
#scatter plot for rainfall vs evaporation          
scatter_plot(df=processed_df, x_col='Rainfall', y_col='Evaporation')