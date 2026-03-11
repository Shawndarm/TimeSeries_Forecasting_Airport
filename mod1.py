#####################################################################################################################
##########################################                             ##############################################
##########################################        Module 1: EDA        ##############################################
##########################################                             ##############################################
#####################################################################################################################


'''
This module stores all our used functions in the Exploratory Data Analysis part

'''



import os
import numpy as np
import pandas as pd
import scipy.stats as stats                             ## Statistical functions

from scipy.interpolate import CubicSpline               ## Cubic interpolation
import statsmodels.api as sm                            ## Statistical model estimation
from statsmodels.tsa.stattools import adfuller, kpss    ## Stationarity tests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  ## Autocorrelation and partial autocorrelation plots
from sklearn.model_selection import train_test_split    ## Data splitting
import statsmodels.formula.api as smf                   ## Statistical models fitting using formulas

from statsmodels.tsa.tsatools import lagmat             ## Lag matrix creation for time series models
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  ## Metrics
import matplotlib.pyplot as plt                         ## Data visualization
import plotly.express as px                             ## Simplified interactive charts
import plotly.graph_objects as go                       ## Custom interactive figures
import seaborn as sns                                   ## Statistical data visualization
from colorama import init, Fore, Back, Style            ## Console text color styles
from skimpy import skim                                 ## Quick overview of pandas DataFrames



# Outliers detection
def outliers(df):
    """
    Calculate the percentage of outliers in each numeric column of a DataFrame.
    Parameter: df
    Returns: outliers_perc_df
    """
    outliers_perc_list = []
    # Selection of numeric columns
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    # Outliers research
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)    # First quartile
        q3 = df[col].quantile(0.75)    # Third quartile
        iqr = q3 - q1                  # Interquartile range
        l_bound = q1 - 1.5 * iqr       # Lower bound
        u_bound = q3 + 1.5 * iqr       # Upper bound

        outliers = df[(df[col] < l_bound) | (df[col] > u_bound)]  # defining what is an outlier
        outliers_perc = round( len(outliers)/len(df)*100, 2)              # Percentage of outliers
        outliers_perc_list.append({'Column': col, 'Outliers Percentage(%)': outliers_perc})

    outliers_perc_df = pd.DataFrame(outliers_perc_list)
    return outliers_perc_df

                                          #################################

# Outliers treatment 
def replace_gas_outliers(df):

    ## Outliers identification ##
    col = 'usage_kWh'
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    l_bound = q1 - 1.5 * iqr
    u_bound = q3 + 1.5 * iqr

    ## Replace outliers ##
    outliers = (df[col] < l_bound) | (df[col] > u_bound)
    mean_value = df[col].mean()  # Mean of the column
    df[col] = np.where(outliers, mean_value, df[col])
    return

                                          #################################

# Data vizualisation
def ts_plot(time,df):
  """
  Creates a time series plot Evolution of gas consumption per year/month.
  Input : x,df
  Output : plots
  """
#Graphs creation
  for col in df:
      graph= px.line(df, x=time, y=col, title=f"Time Series Plot of {col}")
      graph.update_xaxes(
      rangeslider_visible=True,    # Option to add a slider
      rangeselector=dict(          # Option to observe specific periods
            buttons=list([
            dict(count=1, label="Month", step="month", stepmode="backward"),
            dict(count=6, label="Semester", step="month", stepmode="backward"),
            dict(count=1, label="Year-to-Date", step="year", stepmode="todate"),
            dict(count=1, label="Year", step="year", stepmode="backward"),
            dict(step="all")
                      ])))
      graph.show()

                                          #################################

# Covid-19 Identification
def covid_identification(df):
    """
    Highlights the monthly descriptive statistics of the explanatory variables to identify the covid period so that it can be eliminated
    Input: df
    Output : df of monthly descriptive statistics
    """

    monthly_stats= df.groupby(df['UsageDateTime_Zulu'].dt.to_period('M')).agg({'usage_kWh': ['mean', 'std', 'min', 'max'],'Passengers': ['mean', 'std', 'min', 'max'],'Flights': ['mean', 'std', 'min', 'max']}).reset_index()
    return monthly_stats

                                          #################################

# Correlation matrix with Kendall' Tau ##
def corr_matrix(df):

    corr_mat= df.corr(method='kendall') # Correlation matrix for non-linear relations

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, cmap="viridis", fmt=".3f", vmin=-1, vmax=1)
    plt.xticks(rotation=65)
    plt.title("Correlation matrix between all numerical variables")
    plt.show()

                                          #################################

# Identification of missing days
def date_missing(df):
    """
    Find missing dates in the DataFrame.
    Input : our df
    Output : a list containing missing dates (missing lines)
    """
   ## Creation of a list with the complete range of missing days ##
    dates= pd.date_range(start=df["days"].min(), end=df["days"].max(), freq='D')
    missing_dates= dates[~dates.isin(df["days"])].tolist()
    print(f"There are {len(missing_dates)} days missing where data was not entered: \n")
    return missing_dates

                                          #################################

# Imputation  of missing rows
def add_rows(df, missing_dates):
    """
    This function adds rows for missing dates to a DataFrame df, filling all other columns with NaN.
    Input : df and range of missing dates
    Output : our updated df
    """
    # Convert 'days' column to Timestamp objects if not already
    if not pd.api.types.is_datetime64_any_dtype(df['days']):
        df['days'] = pd.to_datetime(df['days'])

    # Convert missing_dates to Timestamp objects
    missing_dates = pd.to_datetime(missing_dates)

    # Create a DataFrame for missing dates with NaN values in other columns
    missing_data = pd.DataFrame({'days': missing_dates})
    for col in df.columns:
        if col != 'days':
            missing_data[col] = np.nan

    # Concatenate the original df and the missing_data DataFrame
    df = pd.concat([df, missing_data], ignore_index=True)

    # Sort values by 'days' and reset index
    df = df.sort_values(by='days').reset_index(drop=True)
    return df

                                          #################################

# Imputation of missing dates
def fill_missing(df):
    # Constrained cubic spline interpolation for 'usage_kWh', 'SunlightDurationMinutes', and 'Temperature_Celsius'
    for col in ['usage_kWh', 'SunlightDurationMinutes', 'Temperature_Celsius']:
        non_na_indices = df.index[~df[col].isna()]
        non_na_days = df.loc[non_na_indices, 'days'].map(pd.Timestamp.timestamp).values
        non_na_values = df.loc[non_na_indices, col].values

        # Specific constraints over the values
        if col == 'usage_kWh':
            # Values between 24,000 and 26,000 kWh for the given period
            non_na_values = np.clip(non_na_values, 24000, 26000)
        elif col == 'Temperature_Celsius':
            # Values between 11 and 12 degrees Celsius for temperature
            non_na_values = np.clip(non_na_values, 11, 12)

        spline = CubicSpline(non_na_days, non_na_values, bc_type='natural')
        missing_indices = df.index[df[col].isna()]
        missing_days = df.loc[missing_indices, 'days'].map(pd.Timestamp.timestamp).values
        interpolated_values = spline(missing_days)

        # Constraint to avoid negative values
        if col != 'Temperature_Celsius':
            interpolated_values[interpolated_values < 0] = 0

        df.loc[missing_indices, col] = interpolated_values

    return df

                                          #################################

# Imputation of special values
def replace_values(df, dates_replace):
    """
    Replace existing values in the df with missing information for specified dates by the mean of the values of the 3 previous days.
    Input: df & List of dates to replace
    Ouput: df màj
    """
    dates_replace= pd.to_datetime(dates_replace)

    for date in dates_replace:

        date_id= df.index[df['days']== date]

        # Finding the index over the 3 last days
        last_dates_id= df.index[(df['days']<date) & (df['days']>= date-pd.Timedelta(days=2))]
        mean_values= df.loc[last_dates_id, df.columns != 'days'].mean()
        df.loc[date_id, df.columns != 'days']= mean_values.values

    # Printing the modified lines
    modif_rows= df[df['days'].isin(dates_replace)]
    print("Modified rows are:")
    print(modif_rows)

                                          #################################

# Stationarity test : Augmented Dickey-Fuller test
def test_ADF(df):
    """
    Apply the Augmented Dickey-Fuller test to test the stationarity of each variable
    H0: The series has a unit root (non-stationary).
    H1: The series has no unit root (stationary).

    Input: df
    Output: DataFrame containing the results of the ADF test for each variable.
    """
    res= []

    for col in df.columns:
        adf_result = adfuller(df[col])
        res.append({'Variable': col,'ADF Statistic': adf_result[0],'p-value': adf_result[1],'Critical Value (5%)': adf_result[4]['5%'],'Stationarity': 'Yes' if adf_result[1] < 0.05 else 'No'})

    return pd.DataFrame(res)

                                          #################################

# Stationarity test : KPSS test
def kpss_test(df):
    """
    Apply the KPSS test to assess the stationarity of each variable in the DataFrame.
    H0: The series is stationary.
    H1: The series has a unit root (non-stationary).

    Input: df
    Output: DataFrame containing the results of the ADF test for each variable.
    """
    res = []

    for col in df.columns:
        kpss_result = kpss(df[col])
        stationarity = 'No' if kpss_result[1] < 0.05 else 'Yes'

        res.append({
            'Variable': col,
            'KPSS Statistic': kpss_result[0],
            'p-value': kpss_result[1],
            'Critical Values': kpss_result[3],
            'Stationarity': stationarity
        })

    return pd.DataFrame(res)

                                          #################################

# ACF/PACF graphs
def acf_pacf(df):

    df= df.loc[:, df.columns != 'days']
    num_col= len(df.columns)
    fig, axes= plt.subplots(num_col, 2, figsize=(15, 7 * num_col))

    for i, col in enumerate(df.columns):
        plot_acf(df[col], lags=40, ax=axes[i, 0], title=f'ACF for {col}')
        plot_pacf(df[col], lags=40, ax=axes[i, 1], title=f'PACF for {col}')

    plt.tight_layout()
    plt.show()

                                          #################################

# Time series decomposition
def decomposition(df):
    """
    Decomposition of gas usage data from df and plot the multiplicative decomposition: usage_kwh(t) = Trend * Seasonality * Noise

    input: df
    output: None
    """
    plt.rcParams['figure.figsize'] = 18, 8
    resu= seasonal_decompose(df['usage_kWh'], model='additive', period=365) #i.e. below
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 12))

    axes[0].plot(df.index, df['usage_kWh'], label='Original series') #Original
    axes[0].set_title('Original Series')
    axes[1].plot(resu.trend.index, resu.trend, label='Trend') #Trend
    axes[1].set_title('Trend Component')
    axes[2].plot(resu.seasonal.index, resu.seasonal, label='Seasonal') #Seasonality
    axes[2].set_title('Seasonal Component')
    axes[3].plot(resu.resid.index, resu.resid, label='Residual') #Noise
    axes[3].set_title('Noise Component')

    plt.tight_layout()
    plt.show()

                                          #################################