# functions and classes to develop
#
# time_grid(dt, n_steps)
# theoretical_mean(t, x0, params)
# theoretical_variance(t, params)
# stationary_variance(params)
# half_life(params)


from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OUParams:
  theta: float
  mu: float
  sigma: float


def validate_params(params, dt):
  theta = params.theta
  sigma = params.sigma

  if theta <= 0:
    raise ValueError("theta must be positive")
  
  if sigma < 0: 
    raise ValueError("sigma must be non-negative")
  
  if dt <= 0:
    raise ValueError("dt must be positive")


def conditional_mean(x, params, dt):
  validate_params(params, dt)
  theta = params.theta
  mu = params.mu
  return mu + (x - mu) * np.exp(-theta * dt)
  

def conditional_variance(params, dt):
  validate_params(params, dt)
  theta = params.theta
  sigma = params.sigma
  return sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)


def exact_step(x, params: OUParams, dt, rng):
  ## Calculates what the next value of x is under an OU process
  validate_params(params, dt)

  mean = conditional_mean(x, params, dt)
  var = conditional_variance(params, dt)
  noise = rng.normal(size=np.shape(x))

  return mean + np.sqrt(var) * noise


def simulate_paths(x0, params, dt, n_steps, n_paths, rng):
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

