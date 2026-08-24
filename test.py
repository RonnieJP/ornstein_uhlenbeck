import numpy as np
from scipy.optimize import minimize
import ornstein_uhlenbeck.ou as ou
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

DEFAULT_BOUNDS = [
    (0, None),
    (None, None),
    (0, None),
]


def neg_log_likelihood(
    x: np.ndarray,
    t: np.ndarray,
    params: ou.OUParams,
) -> float:
    """Return the negative log likelihood of an OU path."""
    dt = np.diff(t)

    log_likelihood = 0.0

    for i in range(len(x) - 1):
        mean = ou.conditional_mean(x[i], params, dt[i])
        variance = ou.conditional_variance(params, dt[i])

        log_likelihood += -0.5 * (
            np.log(2 * np.pi * variance) + (x[i + 1] - mean) ** 2 / variance
        )

    return -log_likelihood


def estimate_parameters(
    x: np.ndarray,
    t: np.ndarray,
    initial_params: list[float],
    bounds: list[tuple[float | None, float | None]] = DEFAULT_BOUNDS,
):
    """Estimate OU parameters using maximum likelihood estimation."""

    def objective(params):
        ou_params = ou.OUParams(
            theta=params[0],
            mu=params[1],
            sigma=params[2],
        )

        return neg_log_likelihood(x, t, ou_params)

    return minimize(
        objective,
        initial_params,
        method="Powell",
        bounds=bounds,
    )


prices = yf.download(
    ["SHEL", "BP"],
    period="5y",
    auto_adjust=True,
)["Close"]

# dates = yf.download(
#     ["SHEL", "BP"],
#     period="5y",
#     auto_adjust=True,
# )["Date"]

shel = prices["SHEL"].to_numpy()
bp = prices["BP"].to_numpy()

t = [i for i in range(len(bp))]


# ------------------- AR1 ------------------#
def intial_estimators(x: np.ndarray, t: np.ndarray) -> list:
    est_mu = x.mean()
    mod = AutoReg(x, 1)
    res = mod.fit()
    delta_t = 1
    alpha, beta = res.params[0], res.params[1]
    est_theta = -np.log(beta) / delta_t
    epsilon = [x[i + 1] - alpha - beta * x[i] for i in range(len(x) - 1)]
    est_sigma = np.std(epsilon)
    return [est_theta, est_mu, est_sigma]


result = sm.OLS(shel, bp).fit()

b = result.params[0]

spread = shel - b * bp
logspread = np.log(shel) - b * np.log(bp)
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
            x=t,
            y=spread,
            mode="lines",
            name="Spread",
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
            x=t,
            y=logspread,
            mode="lines",
            name="Log Spread",
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
            x=bp,
            y=shel,
            mode="markers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bp,
            y=b * bp,
            mode="lines",
        )
    )

    fig.update_layout(
        title="Correlation",
        xaxis_title="BP Prices",
        yaxis_title="SHEL Prices",
    )

    utils.apply_theme(fig)
    fig.show()


m = spread.mean()
initial_params = intial_estimators(spread, t)
print(initial_params)

est = estimate_parameters(
    spread,
    t,
)

# plot_spread()

print(est)

# Sharpe Ratio calculation
# find return series
sharpe = pnl.mean() / pnl.std(ddof = 1) * np.sqrt(252)
time_in_position = sum(pos != 0 for pos in positions)/len(positions)