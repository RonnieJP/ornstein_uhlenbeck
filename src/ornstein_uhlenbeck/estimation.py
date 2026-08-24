import numpy as np
from scipy.optimize import minimize
import ornstein_uhlenbeck.ou as ou
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


def AR1_initial_estimators(x: np.ndarray, t: np.ndarray) -> list[float]:
    AR1_model = AutoReg(x, 1)
    AR1_result = AR1_model.fit()

    delta_t = t[1] - t[0]

    alpha, beta = AR1_result.params[0], AR1_result.params[1]

    epsilon = [x[i + 1] - alpha - beta * x[i] for i in range(len(x) - 1)]

    est_mu = x.mean()
    est_theta = -np.log(beta) / delta_t
    est_sigma = np.std(epsilon)

    return [est_theta, est_mu, est_sigma]


def estimate_parameters(
    x: np.ndarray,
    t: np.ndarray,
    bounds: list[tuple[float | None, float | None]] = DEFAULT_BOUNDS,
) -> ou.OUParams:
    """Estimate OU parameters using maximum likelihood estimation."""

    def objective(params):
        ou_params = ou.OUParams(
            theta=params[0],
            mu=params[1],
            sigma=params[2],
        )

        return neg_log_likelihood(x, t, ou_params)

    initial_params = AR1_initial_estimators(x, t)

    result = minimize(
        objective,
        initial_params,
        method="Powell",
        bounds=bounds,
    )

    return ou.OUParams(
        theta=result.x[0],
        mu=result.x[1],
        sigma=result.x[2],
    )
