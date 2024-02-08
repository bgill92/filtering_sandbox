import numpy as np
import matplotlib.pyplot as plt

def plot_filter(x, y, var):
	"""
	Plot a filtered output and shade the area that encompasses 1 standard deviation around the filtered value
	"""
	plt.plot(x, y, 'b')

	std = np.sqrt(var)

	std_top = y + std
	std_btm = y - std

	plt.plot(x, std_top, linestyle=':', color='k')
	plt.plot(x, std_btm, linestyle=':', color='k')
	plt.fill_between(x, std_btm, std_top,facecolor='yellow', alpha=0.2)