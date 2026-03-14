from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OUParams:
  theta: float
  mu: float
  sigma: float

def exact_step(x, params: OUParams, dt, rng):
  ## Calculates what the next value of x is under an OU process
  theta = params.theta
  mu = params.mu
  sigma = params.sigma

  mean = mu + (x - mu) * np.exp(-2 * theta * dt)
  var = sigma**2 * (1-np.exp(-2 * theta * dt)) / (2 * theta)

  return mean + np.sqrt(var) * rng.normal()

def simulate_path(x0, params, dt, n_steps, rng):
  ## Calculates a series of points which follow an OU process
  x = np.empty(n_steps + 1)
  x[0] = x0

  for i in range(n_steps):
    x[i+1] = exact_step(x[i], params, dt, rng)

  return x

