from copy import deepcopy
from math import tan, sin, cos, sqrt, atan2

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

from scipy.linalg import cholesky

# This calculates sigma points for the unscented transformation
# Adapted from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/sigma_points.py#L24
def merwe_scaled_sigma_points(x, P, subtract=None, alpha=.00001, beta=2, kappa=0):  
  if subtract is None:
    subtract = np.subtract

  # The dimension of the state
  n = len(x)

  # Some magical scaling parameter  
  lambda_ = alpha**2 * (n + kappa) - n

  # This is the "square root" of a matrix. There are many ways to do this
  # For more information, read Implementation of the UKF: Sigma Points from
  # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
  U = cholesky((lambda_ + n)*P)

  # Setting up the sigma points
  sigmas = np.zeros((2*n+1, n))
  sigmas[0] = x
  for k in range(n):
    # The subtract is here in case the state can't be subtracted in a straightforward manner
    sigmas[k+1] = subtract(x, -U[k])
    sigmas[n+k+1] = subtract(x, U[k])

  return sigmas

# This calculates weights for the unscented transformation
# Adapted from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/sigma_points.py#L24
def merwe_scaled_weights(x, alpha=.00001, beta=2, kappa=0):  
  # The dimension of the state
  n = len(x)

  # Some magical scaling parameter  
  lambda_ = alpha**2 * (n + kappa) - n

  # Nominally the weights of the means and covariances for all of the sigma points
  c = .5 / (n + lambda_)

  # Make the arrays of weights of the appropriate size for the means and covariances
  # For more information on why there are 2n + 1 points, read Choosing Sigma Points from 
  # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
  w_m = np.full(2*n + 1, c)
  w_c = np.full(2*n + 1, c)

  # The weights for the first sigma point are a bit different
  w_m[0] = lambda_ / (n + lambda_)  
  w_c[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

  return w_m, w_c

# The unscented transformation takes sigma points that were passed through some arbitrary function
# and then calculates the mean and the covariance of the propagated points
def unscented_transform(sigmas, Wm, Wc, mean_func, residual_func, noise_cov=None):
  # Calculate the mean
  x = mean_func(sigmas, Wm)

  kmax, n = sigmas.shape

  # Calculate the covariance
  P = np.zeros((n, n))

  for k in range(kmax):
    y = residual_func(sigmas[k], x)
    P += Wc[k] * np.outer(y, y)

  if noise_cov is not None:
    if P.shape == noise_cov.shape:
      P += noise_cov
    else:
      # This is to handle a different number of measurements
      n_noise_cov = noise_cov.shape[0]
      n_P = P.shape[0]

      n = int(n_P/n_noise_cov)

      noise_cov_new = np.kron(np.eye(n), noise_cov)

      P += noise_cov_new

  return x, P

class UKF:
  def __init__(self, dt, sigma_points_func, weights_func, state_transition_func, hx, state_avg_func, measurement_avg_func, state_residual_func, measurement_residual_func):
    self.dt = dt

    # The function that calculates the sigma points
    self.sigma_points_func = sigma_points_func

    # The function that calculates the weights associated with the sigma points
    self.weights_func = weights_func

    # The function that propagates the state
    self.state_transition_func = state_transition_func

    # The function which maps the current state into the measurement space
    self.hx = hx

    # The function that takes the average of the state
    self.state_avg_func = state_avg_func

    # The function that takes the average of the measurements
    self.measurement_avg_func = measurement_avg_func

    # The function which returns the residual of the state given two states
    self.state_residual_func = state_residual_func

    # The function which returns the residual of the measurement 
    self.measurement_residual_func = measurement_residual_func

    # The process noise
    self.Q = None
    # The measurement noise
    self.R = None

    # The posterior state and covariance
    self.x = None
    self.P = None

    # The prior state and covariance
    self.x_prior = None
    self.P_prior = None

    # The sigma point weights for the mean
    self.w_m = None
    # The sigma point weights for the covariances
    self.w_c = None

    # The command
    self.u = None

    # The sigmas after they were propagated
    self.sigmas_propagated = None

    # Has the filter been initialized?
    self.init_flag = False

    # Was the update function run since the last time predict was run?
    self.update_run_flag = False

  def initialize_filter(self, x0, P0, Q, R):
    self.x = x0
    self.P = P0
    # Setting the priors because it makes logic checking easier later lol
    self.x_prior = x0
    self.P_prior = P0
    self.Q = Q
    self.R = R

    self.w_m, self.w_c = self.weights_func(self.x)
    self.init_flag = True

  def predict(self, u):
    if not self.init_flag:
      print("Cannot predict, UKF is uninitialized")
      return

    # Save the command
    self.u = u

    # Calculate the sigma points for the initial state
    sigmas_init = self.sigma_points_func(self.x, self.P, self.state_residual_func)

    self.sigmas_propagated = np.zeros(sigmas_init.shape)
    # Propagate the sigma points through the state transition function
    for i, s in enumerate(sigmas_init):
      self.sigmas_propagated[i] = self.state_transition_func(s, self.u, self.dt)

    # Calculate the prior state and covariance via the unscented transform
    self.x_prior, self.P_prior = unscented_transform(self.sigmas_propagated, self.w_m, self.w_c, self.state_avg_func, self.state_residual_func, noise_cov = self.Q)

    self.update_run_flag = False

  def update(self, z, **hx_args):
    if not self.init_flag:
      print("Cannot update, UKF is uninitialized")
      return

    if len(z) == 0:
      self.x = deepcopy(self.x_prior)
      self.P = deepcopy(self.P_prior)
      return

    if self.sigmas_propagated is None:
      print("sigmas_propagated is none, skipping update")
      return

    # pass prior sigmas through h(x) to get measurement sigmas
    # the shape of sigmas_h will vary if the shape of z varies, so
    # recreate each time
    sigmas_h = []
    for s in self.sigmas_propagated:
      temp, z = self.hx(s, z, **hx_args)
      sigmas_h.append(temp)

    sigmas_h = np.atleast_2d(sigmas_h)

    # Calculate the "average" of the transformed state into the measurement space and the system uncertainty
    zp, S = unscented_transform(sigmas_h, self.w_m, self.w_c, self.measurement_avg_func, self.measurement_residual_func, noise_cov = self.R)
    SI = np.linalg.inv(S)

    # compute cross variance of the state and the measurements
    Pxz = np.zeros((self.sigmas_propagated.shape[1], sigmas_h.shape[1]))
    N = self.sigmas_propagated.shape[0]
    for i in range(N):
      dx = self.state_residual_func(self.sigmas_propagated[i], self.x_prior)
      dz = self.measurement_residual_func(sigmas_h[i], zp)
      Pxz += self.w_c[i]*np.outer(dx, dz)

    # Calculate the Kalman Gain
    K = np.dot(Pxz, SI)

    # Calculate the residual
    y = self.measurement_residual_func(z, zp)

    # Calculate the posterior state and covariances (Sometimes a special addition function may be needed)
    self.x = deepcopy(self.x_prior + np.dot(K, y))
    self.P = deepcopy(self.P_prior - np.dot(K, np.dot(S, K.T)))

    self.update_run_flag = True

