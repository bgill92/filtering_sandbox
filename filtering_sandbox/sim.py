import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import numpy as np

from gi.repository import Gio

from diff_drive_robot import DiffDriveRobot, sense_landmarks, sense_landmarks_limited_range
from utils import update_ellipse_plot_arguments

# This is used to control setting the repeat keys functionality
# Essentially debounces a key press
schema = 'org.gnome.desktop.peripherals.keyboard'
key = 'repeat'
settings = Gio.Settings.new(schema)

# Global parameters
LANDMARKS = np.array([[0, 3], [-1.5, 0], [1.5, 0]])
XLIM = (-10, 10)
YLIM = (-10, 10)

# UKF paramaters
SIGMA_RANGE = 0.3
SIGMA_BEARING = 0.1
P = np.diag([.1, .1, .05])
Q = np.eye(3)*0.0001
R = np.diag([SIGMA_RANGE**2, SIGMA_BEARING**2])
UKF_PROCESS_STEP = 1 # The UKF cycle is run every x number of timesteps. If 1, the UKF is run every timestep

# Simulation parameters
MAX_MEASUREMENT_RANGE = 5 # Should the robot have a max sensing range? Set to None if no
dt = 0.1

# Robot parameters
x0 = np.array([0.0, 0.0, -1.57])
radius = 0.5

class SimAnimation:
    def __init__(self, x0, radius, dt):
        self.t = 0.0
        self.dt = dt

        self.robot = DiffDriveRobot(x0, dt, radius, UKF_PROCESS_STEP)
        self.robot.UKF.initialize_filter(x0, P, Q, R)

    def update(self, *args):
        # Update robot state appropriately
        self.robot.state_update_and_collision_check(XLIM, YLIM)

        # Take a measurement
        if MAX_MEASUREMENT_RANGE is not None:
            z = sense_landmarks_limited_range(self.robot.x, LANDMARKS, SIGMA_RANGE, SIGMA_BEARING, MAX_MEASUREMENT_RANGE)
        else:
            z = sense_landmarks(self.robot.x, LANDMARKS, SIGMA_RANGE, SIGMA_BEARING)

        # Run the UKF
        self.robot.run_UKF(z, landmarks=LANDMARKS, max_range=MAX_MEASUREMENT_RANGE)

        # Update graphics
        robot_plot.set(center=(self.robot.x[0], self.robot.x[1]), radius=self.robot.radius)
        heading_plot.set_data(np.array([[self.robot.x[0]], [self.robot.x[0] + self.robot.radius*np.cos(self.robot.x[2])]]), np.array([[self.robot.x[1]], [self.robot.x[1] + self.robot.radius*np.sin(self.robot.x[2])]]))
        if self.robot.UKF.init_flag:
            pred_cov_params = update_ellipse_plot_arguments((self.robot.UKF.x_prior[0], self.robot.UKF.x_prior[1]), self.robot.UKF.P_prior[0:2, 0:2], std=6, facecolor='k', alpha=0.3)
            predict_covariance_plot.set(center=pred_cov_params[0], width=pred_cov_params[1], height=pred_cov_params[2], angle=pred_cov_params[3],
                                        facecolor=pred_cov_params[4], edgecolor=pred_cov_params[5], alpha=pred_cov_params[6], lw=pred_cov_params[7], ls=pred_cov_params[8])
            update_alpha=0.0
            if self.robot.UKF.update_run_flag:
                update_alpha = 0.8
            update_cov_params = update_ellipse_plot_arguments((self.robot.UKF.x[0], self.robot.UKF.x[1]), self.robot.UKF.P[0:2, 0:2], std=6, facecolor='g', alpha=update_alpha)
            update_covariance_plot.set(center=update_cov_params[0], width=update_cov_params[1], height=update_cov_params[2], angle=update_cov_params[3],
                                        facecolor=update_cov_params[4], edgecolor=update_cov_params[5], alpha=update_cov_params[6], lw=update_cov_params[7], ls=update_cov_params[8])

        if MAX_MEASUREMENT_RANGE is not None:
            max_measurement_range_plot.set(center=(self.robot.x[0], self.robot.x[1]), radius=MAX_MEASUREMENT_RANGE, fill=False, edgecolor='b')

        # Update time and other counters
        self.t += self.dt

        return predict_covariance_plot, update_covariance_plot, robot_plot, heading_plot, max_measurement_range_plot

fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect("equal", "box")
ax.grid(True)

ax.set_xlim(XLIM)
ax.set_ylim(YLIM)

plt.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker='s', s=60)

(heading_plot, ) = ax.plot([], [], color="k")
robot_plot = ax.add_patch(plt.Circle((0.0, 0.0), 1.0))
predict_covariance_plot = ax.add_patch(Ellipse((0.0, 0.0), 1.0, 1.0))
update_covariance_plot = ax.add_patch(Ellipse((0.0, 0.0), 1.0, 1.0))
max_measurement_range_plot = ax.add_patch(plt.Circle((0.0, 0.0), 20.0, fill=False, edgecolor='b'))

sim = SimAnimation(x0, radius, dt)

# Functions that control the robot velocity depending on input
def on_key_press(ev):
    if ev.key == "up":
        sim.robot.v = np.array([1, 0.0])
    if ev.key == "left":
        sim.robot.v = np.array([0.0, 1.0])
    if ev.key == "right":
        sim.robot.v = np.array([0.0, -1.0])

def on_key_release(ev):
    sim.robot.v = np.array([0.0, 0.0])

fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)

ani = FuncAnimation(fig, sim.update, None, interval=100, blit=True)

settings.set_boolean(key, False)

plt.show()

settings.set_boolean(key, True)