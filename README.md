# TimeSeries_Forecasting_Airport


# Gas Consumption Forecasting – Time Series Analysis

## Project Overview

This project focuses on forecasting gas consumption using time series econometric models. The analysis is motivated by the sharp increase in gas prices affecting infrastructure such as Edinburgh Airport and the need to anticipate future energy demand for operational planning and cost management.

The objective is to analyze historical gas consumption data, preprocess and structure the time series, and build forecasting models capable of predicting future consumption. Several statistical and econometric techniques are applied, including time series decomposition, stationarity analysis, and dynamic regression models.

The workflow follows three main stages:

1. Exploratory Data Analysis (EDA)
2. Time Series Processing and Transformation
3. Forecasting Model Development

All analysis and modeling are implemented in Python within a Jupyter Notebook.

---

## Methodology

### 1. Exploratory Data Analysis

The first stage focuses on understanding and preparing the dataset:

- Data structure inspection
- Variable interpretation
- Detection of missing values and duplicates
- Outlier detection and treatment
- Data formatting and type conversion
- Univariate and bivariate analysis

Special attention is given to exceptional events such as the COVID-19 period, which may distort normal consumption patterns.

---

### 2. Time Series Processing

To prepare the data for modeling, several time series techniques are applied:

**Handling missing observations**
- Cubic spline interpolation with constraints to estimate missing points
- Replacement with mean values in specific cases

**Stationarity analysis**
- Identification of trends and seasonality
- Statistical checks for stationarity

**Time series decomposition**
- Trend extraction
- Seasonal component identification
- Residual analysis

These steps ensure that the series satisfies the assumptions required by econometric forecasting models.

---

### 3. Forecasting Models

Several forecasting approaches are implemented and evaluated:

**ARX (AutoRegressive with Exogenous Variables)**
- Captures autoregressive behavior with external explanatory variables.

**ADL (Autoregressive Distributed Lag)**
- Models dynamic relationships between the dependent variable and lagged predictors.

**Kalman Filter**
- State-space modeling approach allowing recursive estimation and smoothing of time series dynamics.

**Out-of-sample forecasting**
- Model performance is evaluated on unseen data to assess predictive accuracy.

---

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- Scikit-learn

---

## Project Structure
