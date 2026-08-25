import ornstein_uhlenbeck.strategy as strategy
import ornstein_uhlenbeck.estimation as estimation
import yfinance as yf
import plotly.graph_objects as go
import ornstein_uhlenbeck.ou as ou
import numpy as np
import ornstein_uhlenbeck.utils as utils
import datetime

prices = yf.download(
    ["SHEL", "BP"],
    period="5y",
    auto_adjust=True,
)["Close"]

shel = prices["SHEL"].dropna().to_numpy()
bp = prices["BP"].dropna().to_numpy()

alpha, beta = strategy.estimate_hedge_parameters(shel,bp)

spread = shel - alpha - beta * bp

t = [i for i in range(len(bp))]

split_index = int(0.75 * len(spread))

train = spread[:split_index]
test = spread[split_index - 1:]

t_train = t[:split_index]
t_test = t[split_index - 1:]

print(test.shape)

#test data

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=t_train,
        y=train,
        mode="lines",
        name="Training Data"
    )
)

fig.add_trace(
    go.Scatter(
        x=t_test,
        y=test,
        mode="lines",
        name="Test Data",
    )
)


# Incremental PnL
# Cumulative PnL
# Final PnL
# Average PnL
# Completed trades
# Win rate
# Sharpe
# Drawdown
# Maximum Drawdown

# fig.add_vline(x=t[split_index - 1],line_dash = "dash")

# fig.update_layout(
#     title="Training and Test Data on Spread",
#     xaxis_title=f"Trading Days from {five_years_ago}",
#     yaxis_title="Spread",
# )

utils.apply_theme(fig)
# fig.show()

estimated_params = estimation.estimate_parameters(test, t_test)

stationary_std = ou.stationary_std(estimated_params)

upper_threshold = strategy.upper_threshold(estimated_params)
lower_threshold = strategy.lower_threshold(estimated_params)

positions = strategy.gen_positions(test, estimated_params)
pnl = strategy.gen_pnl(test, estimated_params)
cumulative_pnl = strategy.gen_cum_pnl(test, estimated_params)
final_pnl = strategy.gen_final_pnl(test, estimated_params)
drawdown = strategy.drawdown(test, estimated_params)
max_drawdown = strategy.max_drawdown(test, estimated_params)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=t_test,
        y=cumulative_pnl,
        mode="lines",
    )
)

fig.update_layout(
    title="Spread Data PnL Accumulation",
    xaxis_title="Time",
    yaxis_title="Cumulative PnL",
)

utils.apply_theme(fig)
# fig.show()

# print(f"Estimated stationary standard deviation: {stationary_std:.4f}")
# print(f"Lower threshold: {lower_threshold:.4f}")
# print(f"Upper threshold: {upper_threshold:.4f}")
