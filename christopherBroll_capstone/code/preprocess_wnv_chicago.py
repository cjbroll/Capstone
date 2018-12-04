# ------------------------------------------------------------------------------------------------------------
# NOTE:  this file borrows from kaggle challengers Jenny Doyle, Joseph Brown, Mark Mummert (doyleax)
# The file can be found here: https://github.com/doyleax/West-Nile-Virus-Prediction/blob/master/Final-NB.ipynb
# Random forest code from Dr. Yuxiao Huang (Y. Huang) at GWU

# ----------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------

# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Import data
train = pd.read_csv('train.csv', encoding='utf-8')
weather = pd.read_csv('weather.csv', encoding='utf-8')
spray = pd.read_csv('spray.csv')

cols =['Trap','Latitude','Longitude','Date','Species','WnvPresent','NumMosquitos']

# ----------------------------------------------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------------------------------------------

# Preprocess training dataset
train = train[cols].groupby(cols[:len(cols)-1]).agg({'NumMosquitos':'sum'}).reset_index()
train = pd.get_dummies(train, columns=['Species'])
train.Date = pd.to_datetime(train.Date)
train.drop('NumMosquitos',axis=1,inplace=True)
train.drop_duplicates(inplace=True)
train['year'] = train['Date'].dt.year
train['month'] = train['Date'].dt.month
train['day'] = train['Date'].dt.day

# Preprocess spray
spray.Date = pd.to_datetime(spray.Date)
spray.drop_duplicates(inplace=True)

# Preprocessing weather
weather.Date = pd.to_datetime(weather.Date)
weather = weather[weather.Station == 1].drop('Station', axis=1)
cols = ['Date', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'CodeSum', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']
weather = weather[cols]

# Same as doyleax
cols = ["Tavg", "PrecipTotal", "WetBulb", "StnPressure", "SeaLevel", "AvgSpeed"]
for column in cols:
    weather[column] = weather[column].str.replace('T', '0.005') # T means trace amount
    weather[column] = weather[column].str.replace('M', '0.0') # M means missing
    weather[column] = weather[column].astype(float)

# CodeSum is a code for extreme weather events
weather.CodeSum[weather.CodeSum.str.contains('\w', regex=True)] = '1' # Make binary feature, 1 is extreme weather event
weather.CodeSum[weather.CodeSum != '1'] = '0'
weather.CodeSum = weather.CodeSum.astype(float)

# ---------------------------------------------------------------------------------
# THE FUNCTION BELOW CAN BE FOUND IN CELL 18 OF THE PYTHON NOTEBOOK FROM doyleax
# IT CREATES AVERAGES ACROSS TIMES FOR WEATHER DATA
# ----------------------------------------------------------------------------------

def weather_add(df, weather_col, func, days_range=7):
    new_list = []
    for i in df['Date']:
        mask = (weather['Date'] <= i) & (weather['Date'] >= i - pd.Timedelta(days=days_range))
        data_list = func(weather[weather_col][mask])
        new_list.append(data_list)
    print("Processing average for weather column " + weather_col)
    return new_list

train['Tmax'] = weather_add(train, weather_col='Tmax', func=np.mean)
train['Tmin'] = weather_add(train, weather_col='Tmin', func=np.mean)
train['PrecipTotal'] = weather_add(train, weather_col='PrecipTotal', func= np.sum)
train['Tmax_3'] = weather_add(train, weather_col='Tmax', func=np.mean, days_range=3)
train['Tmax_20'] = weather_add(train, weather_col='Tmax',func=np.mean, days_range=20)
train['DewPoint'] = weather_add(train, weather_col ='DewPoint', func=np.mean, days_range = 10)
train['Tmin_3'] = weather_add(train, weather_col='Tmin', func=np.mean, days_range=3)
train['Tmin_20'] = weather_add(train, weather_col='Tmin', func=np.mean, days_range=20)

# --------------------------------------------------------------------------------------
# RANDOM FOREST FOR FEATURE IMPORTANCE/COMPARISON TO doyleax
# CODE FROM Y. Huang
# --------------------------------------------------------------------------------------

# training data and targets
X = train.drop(["Trap","Date","WnvPresent"], axis=1).values
y = train.WnvPresent.values.ravel()

# Randomly choose 30% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Standization with sklearn's StandardScaler
stdsc = StandardScaler()

# Fit and transform the training set
X_train_std = stdsc.fit_transform(X_train)

# Transform the testing set
X_test_std = stdsc.transform(X_test)

# Random forest classifier
rfc = RandomForestClassifier(class_weight="balanced", random_state=0)

# Train the model
rfc.fit(X_train_std, y_train)

# Get the feature importances
importances = rfc.feature_importances_

# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, train.drop(["Trap","Date","WnvPresent"], axis=1).columns)

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)
print(f_importances)

# -------------------------------------------------------------------------------------------------
# SAME FEATURES AS doyleax FOR COMPARATIVE STUDY
# EXPORTING CLEAN DATA
# -------------------------------------------------------------------------------------------------

# Same features as doyleax
cols = ["Longitude", "DewPoint", "Tmin_20", "month", "Species_CULEX PIPIENS", "Latitude", "Tmax_20", "Tmin_3","WnvPresent"]
train = train[cols]

# Export clean data
writer = pd.ExcelWriter('clean_wnv_chicago_data.xlsx')
train.to_excel(writer)
writer.save()

