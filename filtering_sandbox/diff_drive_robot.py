import numpy as np
from numpy.random import randn
from math import tan, sin, cos, sqrt, atan2

from UKF import UKF, merwe_scaled_sigma_points, merwe_scaled_weights

# Function which calculates next state of robot
def diff_drive_state_update(x, v, dt):
    x += np.array([v[0]*dt*np.cos(x[2]), v[0]*dt*np.sin(x[2]), v[1]*dt])
    return x

# Normalizes an angle to be between [-pi, pi]
def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

# Sense the landmarks with some amount of error
def sense_landmarks(x, landmarks, sigma_range, sigma_bearing):
    x_pos, y_pos = x[0], x[1]
    z = []
    for lmark in landmarks:
        dx, dy = lmark[0] - x_pos, lmark[1] - y_pos
        d = sqrt(dx**2 + dy**2) + randn()*sigma_range
        bearing = atan2(lmark[1] - y_pos, lmark[0] - x_pos)
        a = (normalize_angle(bearing - x[2] + randn()*sigma_bearing))
        z.extend([d, a])
    return z

# Sense the landmarks with some amount of error and upto a maximum range
def sense_landmarks_limited_range(x, landmarks, sigma_range, sigma_bearing, max_range):
    x_pos, y_pos = x[0], x[1]
    z = []
    for lmark in landmarks:
        dx, dy = lmark[0] - x_pos, lmark[1] - y_pos
        d_true = sqrt(dx**2 + dy**2)
        if d_true > max_range:
            continue
        d = d_true + randn()*sigma_range
        bearing = atan2(lmark[1] - y_pos, lmark[0] - x_pos)
        a = (normalize_angle(bearing - x[2] + randn()*sigma_bearing))
        z.extend([d, a])
    return z

# Calculates the residual of the measurement
def residual_measurement(a, b):
    y = a - b
    # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
    for i in range(0, len(y), 2):
        y[i + 1] = normalize_angle(y[i + 1])
    return y

# Calculates the residual of the state
def residual_state(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y

# Converts the robot state into measurement space
def convert_state_to_measurement(x, z, landmarks, max_range):
    """ takes a state variable and returns the measurement
    that would correspond to that state. """
    hx = []
    for lmark in landmarks:
        px, py = lmark
        dist = sqrt((px - x[0])**2 + (py - x[1])**2)
        if dist > (max_range):
            continue
        angle = atan2(py - x[1], px - x[0])
        hx.extend([dist, normalize_angle(angle - x[2])])

    hx = np.array(hx)

    # TODO: This is sorta hacky and should be in a different function
    # If our actual measurement and predicted measurements are not the same length
    if len(z) != len(hx):
        # More measurements than predicted measurements
        if len(z) > len(hx):
            # Remove measurements with the greatest range until our measurements and predicted measurements match in length
            while (len(z) > len(hx)):
                measured_ranges = np.array([z[idx] for idx in np.arange(0,len(z), 2)])
                idx = np.argmax(measured_ranges)
                z = np.delete(z, [idx, idx + 1])
        else:
            # Remove predicted measurements with the greatest range until our measurements and predicted measurements match in length
            while (len(hx) > len(z)):
                estimated_ranges = np.array([hx[idx] for idx in np.arange(0,len(hx), 2)])
                idx = np.argmax(estimated_ranges)
                hx = np.delete(hx, [idx, idx + 1])

    return hx, z

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
def measurement_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = atan2(sum_sin, sum_cos)
    return x

class DiffDriveRobot:
    def __init__(self, x0, dt, radius, UKF_step):
        self.x = x0
        self.v = np.array([0.0, 0.0])
        self.dt = dt
        self.radius = radius
        self.state_update = diff_drive_state_update

        # Used to track one what count to run UKF
        self.count = 0

        # Run the UKF every x timestep
        self.UKF_step = UKF_step

        self.UKF = UKF(self.dt*self.UKF_step, merwe_scaled_sigma_points, merwe_scaled_weights, diff_drive_state_update, convert_state_to_measurement, state_mean, measurement_mean, residual_state, residual_measurement)


    def state_update_and_collision_check(self, xlim, ylim):
        # State update and Collision check
        self.x = self.state_update(self.x, self.v, self.dt)
        if (self.x[0] - self.radius) < xlim[0]:
            self.x[0] = xlim[0] + self.radius
        if (self.x[0] + self.radius) > xlim[1]:
            self.x[0] = xlim[1] - self.radius
        if (self.x[1] - self.radius) < ylim[0]:
            self.x[1] = ylim[0] + self.radius
        if (self.x[1] + self.radius) > ylim[1]:
            self.x[1] = ylim[1] - self.radius

        self.count += 1

    def run_UKF(self, z, **hx_args):
        if ((self.count % self.UKF_step) == 0):
            if not np.all(self.v == np.array([0, 0])):
                self.UKF.predict(self.v)

            self.UKF.update(z, **hx_args)

            count = 0

            print('P:', self.UKF.P.diagonal())

