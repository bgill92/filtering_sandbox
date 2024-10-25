from copy import deepcopy
from math import tan, sin, cos, sqrt, atan2

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

from scipy.linalg import cholesky

from utils import plot_covariance

# All of this code is the implementation of the UKF for the Robot Localization example here:
# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
# I took all of the code and wrote it in one file for easier understanding

# This is the state transition function
def move(x, dt, u, wheelbase):
  hdg = x[2]
  vel = u[0]
  steering_angle = u[1]
  dist = vel * dt

  if abs(steering_angle) > 0.001: # is robot turning?
    beta = (dist / wheelbase) * tan(steering_angle)
    r = wheelbase / tan(steering_angle) # radius

    sinh, sinhb = sin(hdg), sin(hdg + beta)
    cosh, coshb = cos(hdg), cos(hdg + beta)
    return x + np.array([-r*sinh + r*sinhb, 
                          r*cosh - r*coshb, beta])
  else: # moving in straight line
    return x + np.array([dist*cos(hdg), dist*sin(hdg), 0])

# This function is needed to handle cases where the difference in angle
# is like between 1-359 degrees. That should be 2 degrees, not -358 degrees
def normalize_angle(x):
  x = x % (2 * np.pi)    # force in range [0, 2 pi)
  if x > np.pi:          # move to [-pi, pi)
    x -= 2 * np.pi
  return x

# Calculates the residual of the measurement
def residual_h(a, b):
  y = a - b
  # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
  for i in range(0, len(y), 2):
    y[i + 1] = normalize_angle(y[i + 1])
  return y

# Calculates the residual of the state
def residual_x(a, b):
  y = a - b
  y[2] = normalize_angle(y[2])
  return y

# Given the state and the location of the landmarks, this gives back the
# measurement to each landmark
def Hx(x, landmarks):
  """ takes a state variable and returns the measurement
  that would correspond to that state. """
  hx = []
  for lmark in landmarks:
    px, py = lmark
    dist = sqrt((px - x[0])**2 + (py - x[1])**2)
    angle = atan2(py - x[1], px - x[0])
    hx.extend([dist, normalize_angle(angle - x[2])])
  return np.array(hx)

# Calculates the mean of the states
def state_mean(sigmas, Wm):
  x = np.zeros(3)

  sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
  sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
  x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
  x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
  x[2] = atan2(sum_sin, sum_cos)
  return x

# Calculates the means for the measurements
def z_mean(sigmas, Wm):
  z_count = sigmas.shape[1]
  x = np.zeros(z_count)

  for z in range(0, z_count, 2):
    sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

    x[z] = np.sum(np.dot(sigmas[:,z], Wm))
    x[z+1] = atan2(sum_sin, sum_cos)
  return x

def plot_stuff(landmarks, x_true, x_prior, x, P_prior, P_posterior):
  if x_prior.shape[0] != P_prior.shape[2]:
    print("x_prior and P_prior must be the same length")
    return
  if x.shape[0] != P_posterior.shape[2]:
    print("x and P_posterior must be the same length")  
    return

  plt.figure()

  # Plot landmarks
  if len(landmarks) > 0:
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)

  # Plot true track
  plt.plot(x_true[:,0], x_true[:,1], color='k', lw=2)

  # Plot prior covariances
  for idx, elem in enumerate(x_prior):
    plot_covariance((x_prior[idx, 0], x_prior[idx, 1]), P_prior[0:2, 0:2, idx], std=6, facecolor='k', alpha=0.3)

  # Plot posterior covariances
  for idx, elem in enumerate(x):
    plot_covariance((x[idx, 0], x[idx, 1]), P_posterior[0:2, 0:2, idx], std=6, facecolor='g', alpha=0.8)  

  plt.axis('equal')
  plt.title("UKF Robot localization")  

  plt.show()

# This calculates sigma points and weights for the unscented transformation
# Adapted from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/sigma_points.py#L24
# def MerweScaledSigmaPointsAndWeights(x, P, subtract, alpha=.00001, beta=2, kappa=0):
def merwe_scaled_sigma_points_and_weights(x, P, subtract, alpha=.00001, beta=2, kappa=0):  
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

  return sigmas, w_m, w_c

def unscented_transform(sigmas, Wm, Wc, noise_cov=None, mean_func=None, residual_func=None):
  # Calculate the mean
  x = mean_func(sigmas, Wm)

  kmax, n = sigmas.shape

  # Calculate the covariance
  P = np.zeros((n, n))

  for k in range(kmax):
    y = residual_func(sigmas[k], x)
    P += Wc[k] * np.outer(y, y)

  if noise_cov is not None:
    P += noise_cov

  return x, P

def measurement(x, landmarks, sigma_range, sigma_bearing):
  x_pos, y_pos = x[0], x[1]
  z = []
  for lmark in landmarks:
    dx, dy = lmark[0] - x_pos, lmark[1] - y_pos
    d = sqrt(dx**2 + dy**2) + randn()*sigma_range
    bearing = atan2(lmark[1] - y_pos, lmark[0] - x_pos)
    a = (normalize_angle(bearing - x[2] + randn()*sigma_bearing))
    z.extend([d, a])

  return z

def print_P(P_prior_all, P_all):
  N = P_prior_all.shape[2]

  for idx in range(N):
    print("idx: " + str(idx))
    print("P_prior: ")
    print(P_prior_all[:,:, idx])
    print("P_posterior: ")
    print(P_all[:,:, idx])    

def turn(v, t0, t1, steps):
  return [[v, a] for a in np.linspace(np.radians(t0), np.radians(t1), steps)] 

if __name__ == '__main__':
  # Set the random seed
  np.random.seed(22)

  # Parameter initialization
  dt = 0.1
  timesteps = 200

  landmarks = np.array([[5, 10], [10, 5], [15, 15]])

  # landmarks = np.array([[5, 10], [10, 5], [15, 15], [20, 5],[0, 30], [50, 30], [40, 10]])

  commands = [np.array([1.1, .01])] * timesteps

  # # accelerate from a stop
  # commands = [[v, .0] for v in np.linspace(0.001, 1.1, 30)]
  # commands.extend([commands[-1]]*50)

  # # turn left
  # v = commands[-1][0]
  # commands.extend(turn(v, 0, 2, 15))
  # commands.extend([commands[-1]]*100)

  # #turn right
  # commands.extend(turn(v, 2, -2, 15))
  # commands.extend([commands[-1]]*200)

  # commands.extend(turn(v, -2, 0, 15))
  # commands.extend([commands[-1]]*150)

  # commands.extend(turn(v, 0, 1, 25))
  # commands.extend([commands[-1]]*100)

  step = 10
  wheelbase = 0.5

  # Sigmas
  sigma_vel = 0.1
  sigma_steer = np.radians(1)
  sigma_range = 0.3
  sigma_bearing = 0.1

  # The initial true state
  x_init_true = np.array([2, 6, .3])
  P_init = np.diag([.1, .1, .05])
  # Measurement noise to each landmark
  R = np.diag([sigma_range**2, sigma_bearing**2]*len(landmarks))
  # Process noise
  Q = np.eye(3)*0.0001

  # Holds the true robot state from simulating
  x_true_all = deepcopy(x_init_true)

  # Holds the covariance matrix history
  P_all = deepcopy(P_init)

  # True robot state
  x = x_init_true

  # Set the initial state of the filter
  x_UKF = x_init_true

  # Initial covariance matrix
  P = P_init

  x_prior_all = None
  x_all = None
  P_prior_all = None
  P_all = None

  for idx, u in enumerate(commands):

    # Simulate the robot
    x = move(x, dt, u, wheelbase)

    # Run UKF every 10 timesteps
    if idx % step == 0:

      # Take a measurement
      z = measurement(x, landmarks, sigma_range, sigma_bearing)

      # Actual UKF part

      # Prediction

      # Calculate the sigma points and weights based on the current state
      sigmas_init, w_m, w_c = merwe_scaled_sigma_points_and_weights(x = x_UKF, P = P, subtract=residual_x)

      sigmas_f = np.zeros(sigmas_init.shape)
      # Propagate the sigma points through the state transition functions
      for i, s in enumerate(sigmas_init):
        sigmas_f[i] = move(s, dt*step , u, wheelbase)

      # Calculate the prior state and covariance via the unscented transform
      x_prior, P_prior = unscented_transform(sigmas_f, w_m, w_c, noise_cov = Q, mean_func=state_mean, residual_func=residual_x)

      if x_prior_all is None:
        x_prior_all = deepcopy(x_prior)
        P_prior_all = deepcopy(P_prior)
      else:
        x_prior_all = np.vstack([x_prior_all, x_prior])
        P_prior_all = np.dstack([P_prior_all, P_prior])

      # Update the sigma points at the predicted x 
      sigmas_f = merwe_scaled_sigma_points_and_weights(x = x_prior, P = P_prior, subtract=residual_x)[0]

      # Update

      # Convert the sigmas into measurement space
      # I.e. take the state, and convert it into what the expected measurement would be given that state
      sigmas_h = []
      for s in sigmas_f:
        sigmas_h.append(Hx(s, landmarks))

      sigmas_h = np.atleast_2d(sigmas_h)

      # mean and covariance of prediction passed through unscented transform
      # S is the system uncertainty
      print("sigmas_h:", sigmas_h)
      print("R:", R)
      zp, S = unscented_transform(sigmas_h, w_m, w_c, noise_cov = R, mean_func=z_mean, residual_func=residual_h)
      SI = np.linalg.inv(S)

      # compute cross variance of the state and the measurements
      Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
      N = sigmas_f.shape[0]
      for i in range(N):
        dx = residual_x(sigmas_f[i], x_prior)
        dz = residual_h(sigmas_h[i], zp)
        Pxz += w_c[i]*np.outer(dx, dz)
      
      # Calculate the Kalman Gain
      K = np.dot(Pxz, SI)

      # Calculate the residual
      y = residual_h(z, zp)

      # Calculate the posterior state and covariances (Sometimes a special addition function may be needed)
      x_UKF = deepcopy(x_prior + np.dot(K, y))
      P = deepcopy(P_prior - np.dot(K, np.dot(S, K.T))) 

      if x_all is None:
        x_all = deepcopy(x_UKF)
        P_all = deepcopy(P)
      else:
        x_all = np.vstack([x_all, x_UKF])
        P_all = np.dstack([P_all, P])

      # exit()

    x_true_all = np.vstack([x_true_all, x])

  # print_P(P_prior_all, P_all)

  print(x_true_all[:][:,2])

  plot_stuff(landmarks, x_true_all, x_prior_all, x_all, P_prior_all, P_all)

