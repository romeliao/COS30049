import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 


#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
FILL_METHOD = 'ffill'           #this is to specify how to handle missing values in the dataset
SPLIT_METHOD = 'ratio'          #this indicates that the dataset will be split into training and testing based on the specified ratio
DROP_THRESHOLD = 0.7            #if the column is more than 70% missing value it will be dropped from the dataset 
Output_DIagram = 'Diagrams'     #file to save diagrams to
Bar_GraphFeature = 'RainToday'  # to choose what feature to show on the bar graph
RegressionTarget = 'RainTomorrow'

#Scatter plot variables 
X_COL='MinTemp'
Y_COL='MaxTemp'

#features that user wants to call and analyse 
#this is to represent the specific columns from the dataset that will be extracted for analysis
FEATURES = ['Date','Location', 'MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','RainToday','RainTomorrow']

#columns to normalize
#this refers to the columns in the dataset that will be normalized and adjuust the values to a standard cale
# to improve the performace of the machine learning algorithms
NORMALIZE_COLUMN = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir',
            'WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm',
            'Temp9am','Temp3pm','RainToday','RainTomorrow']

#file path is to indicate where the dataset in being read from
#the dataset will be stored in the variable df for further processing 
FILE_PATH = df = pd.read_csv('data/weatherAUS.csv') 

n_clusters = 3  # Choose the number of clusters



#------------------------------------------------------------------------------
# format lables in the dataset 
#------------------------------------------------------------------------------
#this function takse two arguments which is DF and Categorical columns 
# df is the datafram containing the dateset 
# categorical columns is a list of column names that contain categorical data non numeric
def encode_categorical(df, categorical_columns):

    # Initialize LabelEncoder for categorical columns
    #this is to create to store the lable encoder objets for each categorical column
    label_encoders = {}

    #for each column in the list the lable encoder will be created 
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings for consistent encoding
        label_encoders[col] = le  # Store the label encoders for possible inverse_transform later
    
    return df, label_encoders

#------------------------------------------------------------------------------
# load and process data function
#------------------------------------------------------------------------------
def process_weather_data(file_path,fill_method = 'ffill', drop_threshold = 0, normalize_columns=None, features=None):
    
    #the dataset is loaded into the dataframe 
    df = pd.read_csv('data/weatherAUS.csv') 

    #handling missing values
    # mean is to replace missing values with the column mean 
    # ffill is foward fill the missing values 
    # drop is to drop the missing rows with missing values 
    if fill_method =='mean':
        df.fillna(df.mean(), inplace=True)
    elif fill_method == 'ffill':
        df.fillna(method = 'ffill', inplace=True)
    elif fill_method =='drop':
        df.dropna(inplace=True)
    else:
        raise ValueError("Invalid fill method. Choose 'mean', 'ffill', or 'drop'.")

    # Drop columns with more missing data than the threshold
    #the columns with more missing data than the specified treshold are dropped
    #if the threshold is 0.7 any column with more than 70% missing values will be remoced
    df.dropna(thresh=int(drop_threshold * len(df)), axis=1, inplace=True)

     # Ensure the Date column is in datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Coerce errors to handle invalid dates

    #this is the columns to be encoded as categorical features 
    # the encode_categorical function is to transform these categorical columns into numerical lables usign the encoder 
    if features:
        features_to_encode = [col for col in features if col not in 
                              ['Date','Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday','RainTomorrow']]
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
    # This function is used to filter the data fram to include only the specified 
    # Columns in features if is not provided it will return the entire dataframe 
    filtered_df = df[features] if features else df

    return filtered_df
#------------------------------------------------------------------------------
# Call Processing data function 
# this is to load, clean and process the weather data bsed on the parameters provided 
# the file path is which the data set file is found 
# the fill method is for filling the missing vales 
# the drop threshold is to drop the columns with excessive missing data 
# the normalize argument is to ensure the columns are normalize 
# the features will indicate which columns are being called
#------------------------------------------------------------------------------
processed_df = process_weather_data(file_path=FILE_PATH, fill_method=FILL_METHOD, 
                                    drop_threshold=DROP_THRESHOLD, 
                                    normalize_columns=NORMALIZE_COLUMN, 
                                    features=FEATURES)


#------------------------------------------------------------------------------
# preparing data for logistic regression model
#------------------------------------------------------------------------------
df_final = processed_df.copy()

# Define X and y for machine learning
# X will include all features except 'RainTomorrow', and y will be the 'RainTomorrow' column (target)
X = df_final.drop(columns=RegressionTarget)
y = df_final[RegressionTarget]


X = X.select_dtypes(include=[np.number])

y = df_final[RegressionTarget].map({'Yes': 1, 'No': 0})
# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Output sizes of the split datasets
print('\nTrain size:', X_train.shape[0])
print('Test size: ', X_test.shape[0])

# Initialize and train the Logistic Regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred

#------------------------------------------------------------------------------
# logictic regression predictions
#------------------------------------------------------------------------------
def conf_matrix(model, X_test, y_test, cmap='Blues',output_dir =Output_DIagram):
    plot_confusion_matrix(model,X_test,y_test, cmap=cmap)
    plt.title(f'Confusion-Matrix:{RegressionTarget} ', fontsize=14)
    plt.grid()
    output_path = os.path.join(output_dir, 'Confusion_Matrix.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

def roc_curve_custom(model,X_test, y_test,output_dir =Output_DIagram):
    plot_roc_curve(model,X_test,y_test)
    plt.title(f'ROC-Curve:{RegressionTarget} ', fontsize=14)
    plt.plot([0,1], [0,1], color = 'black', linestyle='--')
    output_path = os.path.join(output_dir, 'ROC_Curve.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

def evaluate(model,X_train,X_test, y_train, y_test):
    
    #compute predictions 
    y_pred = model.predict(X_test)

    #confusion matrix 
    print('Confusion Matrix')
    print('-'*53)
    conf_matrix(model,X_test, y_test)
    print('\n')

    #classification Report 
    print('Classification Report') 
    print('-'*53)
    print(classification_report(y_test, y_pred))
    print('\n')

    # ROC Curve
    print('ROC Curve')
    print('-'*53)
    roc_curve_custom(model, X_test, y_test)
    print('\n')
    
    # Checking model fitness
    print('Checking model fitness') 
    print('-'*53)
    print('Train score:', round(model.score(X_train, y_train), 4))
    print('Test score: ', round(model.score(X_test, y_test), 4))
    print('\n')

#------------------------------------------------------------------------------
# Clustering: K-Means Model 
#------------------------------------------------------------------------------
def kmeans_clustering(X, n_clusters=n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

# Apply K-Means clustering on processed data

kmeans_model = kmeans_clustering(X, n_clusters=n_clusters)

#Add cluster lables to the dataframe 
df_final['Cluster'] = kmeans_model.labels_

#------------------------------------------------------------------------------
# PCA for clustering visualization 
#------------------------------------------------------------------------------
def plot_clusters_pca(X, lables, n_clusters=n_clusters):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue= lables, palette='viridis', legend='full')
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.show()

#visualize clustering using PCA
plot_clusters_pca(X, kmeans_model.labels_, n_clusters=n_clusters)

#------------------------------------------------------------------------------
# Data Analysis
# print("\n",df.columns)                      #show what columns are there 
# df.info()                                   #show columns info
#------------------------------------------------------------------------------
#this is to configure the pandas to display all the columns of the dataframe when 
#printing rather than truncating the view 
#the print(processed_df.head(10)) prints out the number of rows that is being called 
print("\n")
pd.set_option('display.max_columns', None)
print(processed_df.head(10))

#to check the top 5 most frequect values for the columns
#this is to iterate through all the columns in the original dataframe 
for col in df.columns:
    print('\n')
    print(col)
    print('-'*20)
    print(df[col].value_counts(normalize=True).head())

#------------------------------------------------------------------------------
# Histogram function
#------------------------------------------------------------------------------
def plot_histogram(df, output_dir =Output_DIagram, figsize=(12,18)):

    #create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Get the continuous
    #this finds the columns in the dataframe that have a numeric datatype
    # these are the columns for which histograms will be plotted
    continuous_columns = [col for col in df.columns if df[col].dtype != object]

    #set up the subplot grid 
    #this creates a grid of subplots with 3 rows and columns using matplotlibs subplots 
    #Flattens the axes array to make it easier to iterate over in a single dimension.
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=figsize)
    axes = axes.reshape(-1)

    #plot a histogram for each of the columns called
    #this function uses seaborn histplot to creat a histogram for each column
    #kde parameters adda kernel density estimate to the plot 
    for i , col in enumerate(continuous_columns):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')

    #adjust layout and save the figure
    #this adjust the layout of the subplots to make sure they are neatly arranged and not 
    #overlapping with a padding of 2.0
    fig.tight_layout(pad=2.0)
    plt.suptitle("Histogram of Columns", fontsize = 16, y =1.02)

    #save the figure 
    output_path = os.path.join(output_dir, 'histograms.png')
    plt.savefig(output_path, facecolor='white',dpi=100)
    plt.show()
#------------------------------------------------------------------------------
# Bar graph function function for rainy days
# this function is to help display the processed data into a bar graph 
# this function takes parameters like DF and output_dir 
# which is where the diagram will be saved to 
#------------------------------------------------------------------------------
def Bar_Graph(df, output_dir=Output_DIagram):

    #ensure the RainToday column is in binary format (1 for 'Yes', 0 for 'No')
    #this step is to ensure that the column is suitable for numerical aggregation and plotting 
    df[Bar_GraphFeature] = df[Bar_GraphFeature].apply(lambda x: 1 if x== 'Yes' else 0)

    #group by location an sum the rainToday column to get the number of rainy days per location 
    #this groups the dataframe by the location colume 
    # and it aggregates the value of the chosen 
    # feature summing them up for each location 
    df_rain_by_location = df.groupby(by='Location').sum()
    df_rain_by_location = df_rain_by_location[[Bar_GraphFeature]]

    #plotting the bar graph
    # the x axis is repensented by the feature that has been chosen to be shown on the bar graph
    # the y axis is to represent the location 
    #data=df_rain_by_location.sort_values('RainToday', ascending=False): this is used to sort the data by 
    # the number of days by descending order example number of rainy days 
    # oreint h is to specifies a horizontal orientation for th bar plot 
    # palette=' creast' is used to appl a color palette for the bars
    plt.figure(figsize=(8,12))
    sns.barplot(x=Bar_GraphFeature, y=df_rain_by_location.index,
                data=df_rain_by_location.sort_values(Bar_GraphFeature, 
                        ascending=False),orient='h', palette='crest')
    plt.xlabel(f'Number of {Bar_GraphFeature}')
    plt.title(f'{Bar_GraphFeature} by Location')
    plt.tight_layout()

    #save the figure
    #this is used to create the file with a specific name and save it to the 
    #designated location 
    output_path = os.path.join(output_dir,f'{Bar_GraphFeature}by_loc.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

#------------------------------------------------------------------------------
# Heatmap function
# this heatmap plot takes in multiple parameters
# the df is used for the datafram which will be used to plot the diagram
# the output dir is used to save the diagram to the right file path 
#figsize is the sige of the figure is set to 
#focus_cols is an optional list of column to focus on if provided only these columns will be included in the heatmap 
#------------------------------------------------------------------------------
def plot_heatmap(df, output_dir=Output_DIagram, figsize=(12,10), focus_cols=None):
    
    #this is used to provide the data fram and it is filtered to include only these 
    #columns. to allow the heatmap for a specific subset of columns to be generated
    if focus_cols:
        df = df[focus_cols]

    #Computes the correlation matrix for the DataFrame. 
    # The correlation matrix is a table showing correlation coefficients 
    # between pairs of columns. Values range from -
    # 1 (perfect negative correlation) to 1 (perfect positive correlation).
    corr_matrix = df.corr()

    #plotting the heatmap
    #Uses Seaborn’s heatmap function to plot the correlation matrix:
    #corr_matrix: The correlation matrix to be visualized.
    #annot=True: Annotates each cell with the numeric value of the correlation coefficient.
    #cmap='coolwarm': Applies the 'coolwarm' color map, which provides a gradient from blue (negative correlations) to red (positive correlations).
    #fmt='.2f': Formats the annotation text to two decimal places.
    #vmin=-1, vmax=1: Sets the color scale limits to -1 and 1 to standardize the visualization range.
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                fmt='.2f', vmin=-1, vmax=1)
    
    #save heatmap diagram to the right path
    plt.title('Correlation Heatmap')
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

#------------------------------------------------------------------------------
# scatter plot diagram
# the scatter plot diagram takes in parameters like 
# df,x_col,y_col output_dir, figsize, and hue 
#------------------------------------------------------------------------------
def scatter_plot(df,x_col,y_col ,output_dir=Output_DIagram, figsize=(8,6), hue=None):

    #create directory if it doesn't exist 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #plotting scatter plot
    # data = df is used to specifi the dataframe containing the data 
    # x=x_col is used to set the x axis to the values from  x_col
    # y=y_col is used to set the y axis to the values from y_col
    #hue is used colors the points based on categories or values in the hue column if
    #hue is none all points will be the same color 
    # the palette='viridis': applies the virdis color palette to the plot
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
# box plot diagram
# df is used to contain the data frame to be ploted 
# features is a list of column names for wich the bo plots will be created 
#output dir is the directory where the diagram will be saved
# figsize is used to specidies the dimensions of the box plot 
#------------------------------------------------------------------------------
# Box plot function
def plot_boxplot(df, features, output_dir=Output_DIagram, figsize=(12,6)):

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the box plot for each feature
    #Creates a figure with the specified size.
    #Uses the boxplot method of the DataFrame to create box plots for the columns specified in features. Each column will have its own box plot.
    #Sets the title of the box plot to "Box Plot of Features".
    #Adjusts the layout to ensure everything fits well within the figure.
    plt.figure(figsize=figsize)
    df[features].boxplot()
    plt.title("Box Plot of Features", fontsize=16)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, 'boxplot.png')
    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.show()

#------------------------------------------------------------------------------
# Calling logistic regression model
#------------------------------------------------------------------------------
evaluate(logreg,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

#------------------------------------------------------------------------------
# Calling graph functions
#------------------------------------------------------------------------------
plot_histogram(processed_df)                    # calling Histogram graph.
Bar_Graph(processed_df)                         # calling bar graph
plot_heatmap(processed_df)                      # calling heat map graph for the functiong being called

#calling scatter plot diagram 
#X_col and y_col can be changed to any funtions 
#example like x_col='Sunshine', y_col='Rainfall'

#scatter plot for min temp and max temp
scatter_plot(df=processed_df, x_col=X_COL, y_col=Y_COL)

#calling box plot function 
plot_boxplot(df=processed_df, features=FEATURES)


