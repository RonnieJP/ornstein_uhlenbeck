import numpy as np
import statsmodels.api as sm

import ornstein_uhlenbeck.ou as ou


def upper_threshold(params):
    return params.mu + 2 * ou.stationary_std(params)


def lower_threshold(params):
    return params.mu - 2 * ou.stationary_std(params)


def gen_positions(x, params):
    positions = np.zeros(len(x))
    position = 0

    if (x.ndim != 1):
        x = x[:, 0]

    for i, spread in enumerate(x):
        if position == 0:
            if spread < lower_threshold(params):
                position = 1
            elif spread > upper_threshold(params):
                position = -1

        elif position == 1:
            if spread >= params.mu:
                position = 0

        elif position == -1:
            if spread <= params.mu:
                position = 0

        positions[i] = position

    return positions


def gen_pnl(x, params):
    if (x.ndim != 1):
            x = x[:, 0]
    spread = x
    positions = gen_positions(x, params)
    return positions[:-1] * np.diff(spread)


def gen_cum_pnl(x, params):
    pnl = gen_pnl(x, params)
    return np.cumsum(pnl)


def gen_final_pnl(x, params):
    return gen_cum_pnl(x, params)[-1]


def gen_trade_pnls(x, params):
    trade_pnls = []
    in_trade = False
    trade_pnl = 0.0

    positions = gen_positions(x, params)
    pnl = gen_pnl(x, params)

    for i, position in enumerate(positions[:-1]):
        if not in_trade and position != 0:
            in_trade = True
            trade_pnl = pnl[i]

        elif in_trade and position != 0:
            trade_pnl += pnl[i]

        elif in_trade and position == 0:
            trade_pnls.append(trade_pnl)
            in_trade = False
            trade_pnl = 0.0

    return trade_pnls


def num_trades(x, params):
    trade_pnls = gen_trade_pnls(x, params)
    return len(trade_pnls)


def avg_trade_pnl(x, params):
    trade_pnls = gen_trade_pnls(x, params)

    if len(trade_pnls) > 0:
        return np.mean(trade_pnls)
    else:
        return 0.0


def winrate(x, params):
    trade_pnls = gen_trade_pnls(x, params)
    n_trades = num_trades(x, params)
    winning_trades = sum(trade_pnl > 0 for trade_pnl in trade_pnls)

    if n_trades > 0:
        win_rate = winning_trades / n_trades
    else:
        win_rate = 0

    return win_rate


def sharpe_ratio(x, params):
    pnl = gen_pnl(x, params)
    return np.mean(pnl) / np.std(pnl, ddof=1)


def drawdown(x, params):
    cumulative_pnl = gen_cum_pnl(x, params)
    running_max = np.maximum.accumulate(cumulative_pnl)
    return cumulative_pnl - running_max


def max_drawdown(x, params):
    return drawdown(x, params).min()


def estimate_hedge_parameters(p1: np.ndarray, p2: np.ndarray):
    X = sm.add_constant(p2)
    result = sm.OLS(p1, X).fit()

    return result.params[0], result.params[1]
