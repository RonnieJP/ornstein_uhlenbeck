import numpy as np
from ornstein_uhlenbeck.estimation import estimate_parameters
import ornstein_uhlenbeck.ou as ou
import statsmodels.api as sm


def OLS_factor(p1: np.ndarray, p2: np.ndarray):
    result = sm.OLS(p1, p2).fit()
    return result.params[0]

def pnl(
    x: np.ndarray, 
    t: np.ndarray
):
    split_index = int(0.75 * len(x))

    train = x[:split_index]
    test = x[split_index - 1:]

    t_train = t[:split_index]
    t_test = t[split_index - 1:]

    result = estimate_parameters(train, t_train)

    estimated_params = ou.OUParams(
        theta=result.x[0],
        mu=result.x[1],
        sigma=result.x[2],
    )

    print(estimated_params)

    stationary_std = ou.stationary_std(estimated_params)
    upper_threshold = estimated_params.mu + 2 * stationary_std
    lower_threshold = estimated_params.mu - 2 * stationary_std

    positions = np.zeros(len(test))
    position = 0

    for i, spread in enumerate(test):
        if position == 0:
            if spread < lower_threshold:
                position = 1
            elif spread > upper_threshold:
                position = -1

        elif position == 1:
            if spread >= estimated_params.mu:
                position = 0

        elif position == -1:
            if spread <= estimated_params.mu:
                position = 0

        positions[i] = position

    spread = test

    pnl = positions[:-1] * np.diff(spread)
    return pnl, positions

def final_pnl(pnl):
    cumulative_pnl = np.cumsum(pnl)
    fin_pnl = cumulative_pnl[-1]
    return fin_pnl

def trade_pnls(pnl, positions):
    trade_pnls = []
    in_trade = False
    trade_pnl = 0.0

    for i, position in enumerate(positions[:-1]):

        if not in_trade and position != 0:
            # Enter trade
            in_trade = True
            trade_pnl = pnl[i]

        elif in_trade and position != 0:
            # Continue holding
            trade_pnl += pnl[i]

        elif in_trade and position == 0:
            # Exit trade
            trade_pnls.append(trade_pnl)
            in_trade = False
            trade_pnl = 0.0

    return trade_pnls

def number_of_trades(trade_pnls):
    return len(trade_pnls)

def average_trade_pnl(trade_pnls):
    return np.mean(trade_pnls)

def win_rate(trade_pnls,number_of_trades):
    winning_trades = sum(trade_pnl > 0 for trade_pnl in trade_pnls)
    if number_of_trades > 0:
        win_rate = winning_trades / number_of_trades
    else:
        win_rate = 0.0
    return win_rate

def sharpe(trade_pnls, average_trade_pnl):
    std_trade_pnl = np.std(trade_pnls)
    sharpe = average_trade_pnl / std_trade_pnl
    return sharpe