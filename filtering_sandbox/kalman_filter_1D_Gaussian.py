import numpy as np
from copy import deepcopy
from typing import List

import matplotlib
import matplotlib.pyplot as plt

class Gaussian:
	def __init__(self, mean, variance):
		self.mean = mean
		self.variance = variance

	def __repr__(self):
		return f"mean: {self.mean}, variance: {self.variance}"

def g_add(one: Gaussian, two: Gaussian):
	temp = Gaussian(0.0, 0.0)
	temp.mean = one.mean + two.mean
	temp.variance = one.variance + two.variance
	return temp

def g_mult(one: Gaussian, two: Gaussian):
	temp = Gaussian(0.0, 0.0)
	temp.mean = (one.variance * two.mean + two.variance * one.mean) / (one.variance + two.variance)
	temp.variance = (one.variance * two.variance) / (one.variance + two.variance)
	return temp	

def plot_filter(y: List[Gaussian]):
	mean = [x.mean for x in y]

	plt.plot(mean, 'b')

	var = [x.variance for x in y]

	std = [np.sqrt(x) for x in var]

	std_top = [x[0] + x[1] for x in zip(mean,std)]
	std_btm = [x[0] - x[1] for x in zip(mean,std)]

	plt.plot(std_top, linestyle=':', color='k')
	plt.plot(std_btm, linestyle=':', color='k')
	plt.fill_between([x for x in range(len(y))], std_btm, std_top,facecolor='yellow', alpha=0.2)

class Simulate1DParticle:
	def __init__(self, x_0, true_vel, dt, process_var = 0):
		self.x = x_0 
		self.vel = true_vel
		self.dt = dt
		self.process_var = process_var

	def simulate(self):
		self.vel += np.random.randn() * self.process_var * dt
		self.x += self.vel * self.dt

	def measure(self, measurement_variance):
		return self.x + np.random.randn() * np.sqrt(measurement_variance)

if __name__ == '__main__':
	# Set random seed	
	np.random.seed(13)

	# Number of timesteps to simulate/filter
	N = 25

	# Timestep
	dt = 1.0

	# Initial position
	x_0 = 0.0
	# Initial position standard deviation
	std_dev_0 = 20.0

	# Expected particle velocity
	vel = 1.0
	# Variance in particle velocity
	process_var = 2.0

	# Variance in the position sensor measurements
	sensor_var = 4.5

	# The Gaussian defining the initial particle position and uncertainty
	x = Gaussian(x_0, std_dev_0**2)

	# The Gaussian defining the particles movement and the associated uncertainty
	process_model = Gaussian(vel*dt, process_var)

	# True particle parameters
	x_true_0 = 0.0
	vel_true = 1.0
	particle = Simulate1DParticle(x_true_0, vel_true, dt)

	# Lists to accumulate values
	estimates = []
	truth = []
	measurements = []
	predictions = []

	# Get initial prediction
	estimates.append(deepcopy(x))

	# Time vector and initial time
	t = []
	t_0 = 0.0

	for i in range(N):

		t_0 += dt
		t.append(t_0)

		particle.simulate()

		# The predicted new particle position given the current estimate and process model
		predict = g_add(x, process_model)

		# The measured particle position
		measurement = particle.measure(sensor_var)
		likelihood = Gaussian(measurement, sensor_var)

		# Update the estimate by multiplying the prediction with the measurement
		x = g_mult(predict, likelihood)

		estimates.append(deepcopy(x))
		truth.append(deepcopy(particle.x))
		measurements.append(likelihood)
		predictions.append(deepcopy(predict))

	for x in estimates:
		print(x)
	print(truth)
	print(measurements)
	print(predictions)

	plot_filter(estimates)
	plt.plot(t, truth, 'k')
	plt.plot(t, [x.mean for x in measurements], 'rv')
	plt.plot(t, [x.mean for x in predictions], 'c^')
	plt.show()

# if __name__ == '__main__':
# 	# A Kalman Filter is composed of equations that do two things, predict and update
# 	# Base Kalman Filters are based off of the assumption that the state and noises are Gaussian

# 	# Set the seed for the random noise
# 	np.random.seed(13)

# 	# We want to track a particle moving with a constant velocity
# 	x_truth_0 = 0.0;
# 	x_truth = x_truth_0;

# 	x_true_velocity = 0.5;

# 	# The time step
# 	dt = 0.1

# 	# The process variance for specifically the velocity
# 	process_var = 0.1;

# 	# The measurement variance for the position measurement
# 	measurement_var = 0.1;

# 	# The function which gives a position given a starting position, velocity, and timestep
# 	new_position = lambda x_0, vel, dt, process_var : vel*dt + x_0

# 	# Initialize the state of the filter
# 	x = Gaussian(0.0, 1.0);

# 	# Initialize our belief in the state


# 	# Predict
# 	# Use the system behavior to predict state at the next time step

# 	# Adjust belief to account for the uncertainty in prediction

# 	# Update
# 	# Get a measurement and associated beief about its accuracy

# 	# Compute residual between estimated state and measurement

# 	# Compute scaling factor based on whether the measurement or prediction is more accurate

# 	# Set state between the prediction and measurement based on scaling factor

# 	# Update belief in the state based on how certain we are in the measurement
