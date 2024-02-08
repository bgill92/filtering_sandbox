import numpy as np

class Simulate1DParticle:
	"""
	This class simulates a 1 dimensional particle.
	"""
	def __init__(self, x_0, true_vel, dt, process_var = 0):
		self.x = x_0 
		self.vel = true_vel
		self.dt = dt
		self.process_var = process_var

	def simulate(self):
		self.vel += np.random.randn() * self.process_var * self.dt
		self.x += self.vel * self.dt

	def measure(self, measurement_variance):
		return self.x + np.random.randn() * np.sqrt(measurement_variance)