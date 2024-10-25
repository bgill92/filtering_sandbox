import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.linalg import inv

from simulate import Simulate1DParticle
from utils import plot_filter, Q_discrete_white_noise
from mkf_internal import plot_track

if __name__ == "__main__":
	# Set the random seed
	np.random.seed(3)

	# Time vector, with an initial value of 0.0
	t_vec = np.array([0.0])
	# Timestep
	dt = 1.0
	# Number of timesteps to take
	N = 50

	# Process variance
	Q_var = 0.01
	Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)

	# True particle values
	x_0_true = 0.0
	vel_true = 1.0
	particle = Simulate1DParticle(x_0_true, vel_true, dt, Q_var)

	# Initial state (Initialize the state of the filter)
	# We want to track the particles position and velocity,
	# so the state vector is equal to [x, x_dot]
	x_0 = np.array([[1.0],[4.5]])
	# Initial variance (Initialize our belief in the state)
	P_0 = np.diag([500, 49])

	# Process model matrix
	F = np.array([[1, dt],
								[0,  1]])

	# Measurement vector
	H = np.array([[1, 0]])
	# Measurement variance
	R_var = 10
	R = np.array([[R_var]])

	# Accumulation vectors
	prediction_vec = []
	measurement_vec = []
	pos_true = []
	K_vec = []
	estimate_vec = []
	P_vec = []

	# Set the state and covariance
	x = x_0
	P = P_0

	for i in range(N):
		# Simulate the particle
		particle.simulate();
		pos_true.append(particle.x)

		# Update the time vector
		t_vec = np.append(t_vec,[t_vec[-1] + dt])

		# Prediction step 
		# Calculate the prior (Use the system behavior to predict state at the next time step)
		x_prior = F @ x
		prediction_vec.append(x_prior)

		# Update the variance (Adjust belief to account for the uncertainty in prediction)
		P = F @ P @ F.T + Q

		# Update step
		# Calculate the system uncertainty
		# (This maps the state covariance matrix to the measurement space)
		S = H @ P @ H.T + R

		# Calculate the Kalman Gain
		K = P @ H.T @ inv(S)
		print(K)
		K_vec.append(K)

		# Get a measurement and the associated belief about its accuracy
		z = particle.measure(R_var)
		measurement_vec.append(z)

		# Compute residual between estimated state and measurement
		y = z - H @ x_prior

		# Set state between the prediction and measurement based on Kalman Gain
		x = x_prior + K @ y
		estimate_vec.append(x)

		# Update belief in the state based on how certain we are in the measurement
		P = P - K @ H @ P
		P_vec.append(P)

	# print(P_vec)
	estimate_vec, P_vec = np.array(estimate_vec), np.array(P_vec)
	# print(pos_true[1:])
	# print(estimate_vec[:,0,0])
	# print(measurement_vec)
	plot_track(estimate_vec[:,0,0], pos_true, measurement_vec, P_vec)
	# plt.plot(t_vec, measurement_vec, 'rv')
	# plt.plot(t_vec, prediction_vec, 'c^')
	plt.show()
