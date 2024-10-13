import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from Models.data_split import DataSplit

# Heavily inspired by user nikhilkohli @ Kaggle.com

def load_data() -> pd.DataFrame:
    # Let first column (dates) be row indices.
    data = pd.read_csv('../data/stock_data.csv', sep=",", header=0, decimal=".", index_col=0)

    df_Stock = data
    df_Stock = df_Stock.rename(columns={'Close(t)':'Close'})
    df_Stock = df_Stock.drop(columns='Date_col')

    return df_Stock


def plot_data(df_Stock: pd.DataFrame):
    df_Stock['Close'].plot(figsize=(10,7))
    plt.title("Stock price")
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()

def plot_predicted_vs_actual(df_pred: pd.DataFrame):
    df_pred[['Actual', 'Predicted']].plot(figsize=(10, 7))
    plt.title("Actual vs Predicted")
    plt.ylabel('Values')
    plt.xlabel('Date')
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()

def create_train_test_set(df_Stock: pd.DataFrame) -> DataSplit:
    features = df_Stock.drop(columns=['Close_forcast'], axis=1) # axis=1 implies the operation should be applied along the columns
    target = df_Stock['Close_forcast']

    data_len = df_Stock.shape[0] # number of rows
    training_data_size = int(data_len * 0.88)
    validation_data_size = training_data_size + int(data_len * 0.1)

    X_train, X_val, X_test = features[:training_data_size], features[training_data_size:validation_data_size], features[validation_data_size:]
    Y_train, Y_val, Y_test = target[:training_data_size], target[training_data_size:validation_data_size], target[validation_data_size:]

    return DataSplit(X_train, X_val, X_test, Y_train, Y_val, Y_test)

def get_metrics_and_plot(dataSplit: DataSplit):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataSplit.X_train, dataSplit.X_val, dataSplit.X_test, dataSplit.Y_train, dataSplit.Y_val, dataSplit.Y_test

    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    Y_train_pred = lr.predict(X_train)
    Y_val_pred = lr.predict(X_val)
    Y_test_pred = lr.predict(X_test)

    # print('Linear regression coefficients: \n', lr.coef_)
    # print('Linear regression intercept: \n', lr.intercept_)

    print("Training R-squared: ", round(metrics.r2_score(Y_train,Y_train_pred),2))
    # print('Training MAPE:', round(get_mape(Y_train,Y_train_pred), 2)) 
    print('Training Mean Squared Error:', round(metrics.root_mean_squared_error(Y_train,Y_train_pred), 2))

    print('\n')

    print("Validation R-squared: ", round(metrics.r2_score(Y_val, Y_val_pred), 2))
    print('Validation Mean Squared Error:', round(metrics.root_mean_squared_error(Y_val, Y_val_pred), 2))

    print('\n')

    print("Test R-squared: ", round(metrics.r2_score(Y_test, Y_test_pred), 2))
    print('Test Mean Squared Error:', round(metrics.root_mean_squared_error(Y_test, Y_test_pred), 2))

    ## Plotting Actual vs Predicted values
    df_pred = pd.DataFrame(Y_val.values, columns=['Actual'], index=Y_val.index)
    df_pred['Predicted'] = Y_val_pred
    df_pred = df_pred.reset_index()
    df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'],format='%Y-%m-%d')
    plot_predicted_vs_actual(df_pred)

def linear_regression_stocks():
    df_Stock = load_data()
    plot_data(df_Stock)

    dataSplit = create_train_test_set(df_Stock)
    
    get_metrics_and_plot(dataSplit)

def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Check for zero values in y_true
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, MAPE cannot be computed.")

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    linear_regression_stocks()