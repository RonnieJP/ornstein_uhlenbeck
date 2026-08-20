import numpy as np
from scipy.optimize import minimize
import ornstein_uhlenbeck.ou as ou

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
