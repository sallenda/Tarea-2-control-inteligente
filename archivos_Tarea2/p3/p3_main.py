import time
from robot_model import RobotModel
import matplotlib.pyplot as plt
from EKF import EKF
import numpy as np
from robot_env import HuskyTrajectoryEnv
#from robot_env_maze_narrow import HuskyTrajectoryEnv

# Environment
show_gui = False
env = HuskyTrajectoryEnv(gui=show_gui,
                         pose_rate_hz=1/5, # How often we receive position measurements
                         seed=1,
                         consider_obstacles=False)
reference = env.reference # Trajectory Reference
R_pose = env.R_pose
R_vel = env.R_vel
truth = env.reset()


# Model and EKF
Ts = 1/30
model = RobotModel(Ts=Ts)
Q = np.diag([
    0.005**2,          # x
    0.005**2,          # y
    np.deg2rad(0.2)**2, # theta
    0.02**2,           # v
    np.deg2rad(0.5)**2  # omega
])
P = np.eye(5)*1
x_init = np.array(truth).reshape(-1, 1)
ekf = EKF(model, Q, R_vel, R_pose, P, x_init, Ts=Ts)


state_hist = []
est_state_hist = []
P_hist = []
for k in range(1000):
    print(k)

    u = np.array([
        1,     # v
        0.1      # omega
    ])

    # Truth is just for evaluation, you have to use measurements for your controller and filter
    truth, collision = env.step(u)
    state_hist.append(np.array(truth).reshape(1, -1))

    measurements = env.get_measurements()

    # Kalman Filter
    ekf.predict(u)
    ekf.update(measurements['velocity'],  measurement_type='vel') # Always update encoder and IMU

    if measurements["pose_available"]:
        ekf.update(measurements['pose'],  measurement_type='pos') # Whenever a GPS data comes, update

    state_est, P = ekf.get_state_and_cov() # state_est should go to your controller
    P_hist.append(P.copy())
    est_state_hist.append(np.array(state_est).reshape(1, -1))

    if collision:
        print("Collision!")
        #break

    if show_gui:
        time.sleep(env.dt)

env.close()


# Plotting
state_hist = np.concatenate(state_hist, axis=0)
est_state_hist = np.concatenate(est_state_hist, axis=0)
P_hist = np.stack(P_hist, axis=0)
stds = np.sqrt(np.diagonal(P_hist, axis1=1, axis2=2))
time_vec = np.arange(len(est_state_hist)) * Ts

state_names = ['x', 'y', 'theta', 'v', 'omega']

plt.figure(figsize=[21, 15])
for i in range(state_hist.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.plot(time_vec, state_hist[:, i], label=state_names[i])
    plt.plot(time_vec, est_state_hist[:, i], label=state_names[i] + '_est')
    plt.fill_between(time_vec, est_state_hist[:, i] - 2 * stds[:, i], est_state_hist[:, i] + 2 * stds[:, i],
                     alpha=0.25, label=r"$\pm 2\sigma$")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()