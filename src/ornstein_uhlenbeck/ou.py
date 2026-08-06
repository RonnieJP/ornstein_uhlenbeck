from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OUParams:
    theta: float
    mu: float
    sigma: float


def validate_params(params: OUParams, dt) -> None:
    """Validate the OU process parameters."""
    theta = params.theta
    sigma = params.sigma

    if theta <= 0:
        raise ValueError("theta must be positive")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    if dt <= 0:
        raise ValueError("dt must be positive")


def conditional_mean(
    x: np.ndarray | float, params: OUParams, dt: float
) -> np.ndarray | float:
    """Return the conditional mean of the OU process."""
    # validate_params(params, dt)
    theta = params.theta
    mu = params.mu
    return mu + (x - mu) * np.exp(-theta * dt)


def conditional_variance(params: OUParams, dt: float) -> float:
    """Return the conditional variance of the OU process."""
    # validate_params(params, dt)
    theta = params.theta
    sigma = params.sigma
    return sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)


def exact_step(
    x: np.ndarray | float, params: OUParams, dt: float, rng: np.random.Generator
) -> np.ndarray | float:
    """Simulate a single exact OU transition."""
    validate_params(params, dt)

    mean = conditional_mean(x, params, dt)
    var = conditional_variance(params, dt)
    noise = rng.normal(size=np.shape(x))

    return mean + np.sqrt(var) * noise


def simulate_paths(
    x0: float,
    params: OUParams,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate multiple OU sample paths using the exact transition law."""
    validate_params(params, dt)

    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")

    if not isinstance(n_paths, int):
        raise TypeError("n_paths must be an integer")

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    x = np.empty((n_steps + 1, n_paths))
    x[0] = x0

    for i in range(n_steps):
        x[i + 1] = exact_step(x[i], params, dt, rng)

    return x


def time_grid(dt: float, n_steps: int) -> np.ndarray:
    """Return the time points corresponding to a simulation."""
    if dt <= 0:
        raise ValueError("dt must be positive")

    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")

    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    # note that here n_steps can take value 0 unlike simulate_paths.
    # simulation needs atleast one update.

    return dt * np.arange(n_steps + 1)


def theoretical_mean(
    t: np.ndarray | float, x0: np.ndarray | float, params: OUParams
) -> np.ndarray | float:
    """Return the theoretical mean of the OU process at time t."""
    return params.mu + (x0 - params.mu) * np.exp(-params.theta * t)


def theoretical_variance(t: np.ndarray | float, params: OUParams) -> np.ndarray | float:
    """Return the theoretical variance of the OU process at time t."""
    return params.sigma**2 * (1 - np.exp(-2 * params.theta * t)) / (2 * params.theta)


def stationary_variance(params: OUParams) -> float:
    """Return the stationary variance of the OU process."""
    return params.sigma**2 / (2 * params.theta)


def half_life(params: OUParams) -> float:
    """Return the half-life of an OU process."""
    return np.log(2) / params.theta
