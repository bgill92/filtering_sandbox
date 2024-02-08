import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from simulate import Simulate1DParticle
from plotting_utils import plot_filter

if __name__ == "__main__":
	# Set the random seed
	np.random.seed(22)

	# Time vector, with an initial value of 0.0
	t_vec = np.array([0.0])
	# Timestep
	dt = 0.1
	# Number of timesteps to take
	N = 100

	# True particle values
	x_0_true = 0.0
	vel_true = 1.0
	particle = Simulate1DParticle(x_0_true, vel_true, dt)

	# Initial state (Initialize the state of the filter)
	x_0 = 0.0
	# Initial variance (Initialize our belief in the state)
	P_0 = 20**2

	# Process model parameters
	vel = 1.0

	# Process variance
	Q = 0.1
	# Measurement variance
	R = 0.5

	# Accumulation vectors
	estimate_vec = np.array([x_0])
	P_vec = np.array([P_0])
	prediction_vec = np.array([x_0])
	measurement_vec = np.array([x_0])

	# Set the state and covariance
	x = x_0
	P = P_0

	for i in range(N):
		# Simulate the particle
		particle.simulate();

		# Update the time vector
		t_vec = np.append(t_vec,[t_vec[-1] + dt])

		# Prediction step 
		# Calculate the prior (Use the system behavior to predict state at the next time step)
		x_prior = x + vel*dt
		prediction_vec = np.append(prediction_vec, x_prior)

		# Update the variance (Adjust belief to account for the uncertainty in prediction)
		P = P + Q

		# Update step
		# Get a measurement and the associated belief about its accuracy
		z = particle.measure(R)
		measurement_vec = np.append(measurement_vec, z)

		# Compute residual between estimated state and measurement
		y = z - x_prior

		# Compute scaling factor based on whether the measurement or prediction is more accurate
		# Otherwise known as the Kalman Gain
		K = P / (P + R)

		# Set state between the prediction and measurement based on Kalman Gain
		x = x_prior + K*y
		estimate_vec = np.append(estimate_vec, x)

		# Update belief in the state based on how certain we are in the measurement
		P = (1 - K) * P
		P_vec = np.append(P_vec, P)

	plot_filter(t_vec, estimate_vec, P_vec)
	plt.plot(t_vec, measurement_vec, 'rv')
	plt.plot(t_vec, prediction_vec, 'c^')
	plt.show()
