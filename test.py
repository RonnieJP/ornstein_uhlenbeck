from ornstein_uhlenbeck import ou, utils, estimation
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import numpy as np

prices = yf.download(
    ["SHEL", "BP"],
    period="5y",
    auto_adjust=True,
)["Close"]

prices = prices[["SHEL", "BP"]].dropna()

shel = prices["SHEL"].to_numpy()
bp = prices["BP"].to_numpy()
t = [i for i in range(len(bp))]

result = sm.OLS(shel, bp).fit()

# print(result.summary())

b = 1.9653

spread = shel - b*bp
logspread = np.log(shel) - b*np.log(bp)
import plotly.graph_objects as go

def plot_prices():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=bp,
            mode="lines",
            name="BP",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=shel,
            mode="lines",
            name="SHEL",
        )
    )

    fig.update_layout(
        title="Stock Prices",
        xaxis_title="Trading Days from 21 Aug 2021",
        yaxis_title="Prices",
    )

    utils.apply_theme(fig)
    fig.show()

def plot_spread():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = t,
            y = spread,
            mode = "lines",
            name = "Spread",
        )
    )

    fig.update_layout(
        title="Spread",
        xaxis_title="Trading Days from 21 Aug 2021",
        yaxis_title="Spread",
    )

    utils.apply_theme(fig)
    fig.show()

def plot_logspread():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = t,
            y = logspread,
            mode = "lines",
            name = "Log Spread",
        )
    )

    fig.update_layout(
        title="Log Spread",
        xaxis_title="Trading Days from 21 Aug 2021",
        yaxis_title="Spread",
    )

    utils.apply_theme(fig)
    fig.show()

def plot_correlation():
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = bp,
            y = shel,
            mode = "markers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x = bp,
            y = b*bp,
            mode = "lines",
        )
    )

    fig.update_layout(
        title="Correlation",
        xaxis_title="BP Prices",
        yaxis_title="SHEL Prices",
    )

    utils.apply_theme(fig)
    fig.show()

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                      y   R-squared (uncentered):                   0.990
# Model:                            OLS   Adj. R-squared (uncentered):              0.990
# Method:                 Least Squares   F-statistic:                          1.274e+05
# Date:                Fri, 21 Aug 2026   Prob (F-statistic):                        0.00
# Time:                        14:52:11   Log-Likelihood:                         -4025.3
# No. Observations:                1255   AIC:                                      8053.
# Df Residuals:                    1254   BIC:                                      8058.
# Df Model:                           1                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1             1.9653      0.006    356.877      0.000       1.955       1.976
# ==============================================================================
# Omnibus:                      263.334   Durbin-Watson:                   0.010
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               52.667
# Skew:                           0.073   Prob(JB):                     3.66e-12
# Kurtosis:                       2.007   Cond. No.                         1.00
# ==============================================================================

# Notes:
# [1] R² is computed without centering (uncentered) since the model does not contain a constant.
# [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.

m = logspread.mean()
initial_params = [30,m,1000]
result = estimation.estimate_parameters(
    logspread,
    t,
    initial_params,
)

print(result)

# fitted parmas: [ 3.498e+01 -1.969e-01  5.000e+01]
# theta = 34.98
# mu = -0.1969
# sigma = 50
