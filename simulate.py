import numpy as np
from MPPI import model, MPPI
import matplotlib.pyplot as plt
from LKF import LKF
from PID import Controller
from scipy.spatial.transform import Rotation

physics = model(1, 1000, 16, 0.01, 0.9, 0.6, 0.4, 40, 36)
optimizer = MPPI(5, 100, physics, 0.1, 0.01)

init_pos = np.array([0,0,0])
setpoint = np.array([4,7,2])

def get_dvl_data():
    dvl_data = [0,0,0]
    return dvl_data

x_state = []
y_state = []
z_state = []

for _ in range(300):
    # print(init)
    mu = optimizer.optimize(init_pos, setpoint)
    # print(mu)
    mov, vel = physics.movement(mu, optimizer.time_step, get_dvl_data())
    init_pos = init_pos + mov
    x_state.append(init_pos[0])
    y_state.append(init_pos[1])
    z_state.append(init_pos[2])

#---------------------------------Orientation with PID and Kalman Filter-----------------------------------

gyro_noise_std = 0.03  # std deviation for gyro noise
acc_noise_std = 0.01   # std deviation for accelerometer noise

# Simulation setup
dt = 0.01  # time step
drag_coefficient = 0.7  # drag coefficient

# Create Kalman Filter object with adjusted process noise
lkf = LKF(dt, drag_coefficient)

# initialize PID controller
cntrl_yaw = Controller(10, 0, 2)
cntrl_pitch = Controller(10, 0, 2)
cntrl_roll = Controller(10, 0, 2)

# initial and setpoint orientations (yaw, pitch, roll)
orient = np.array([0.0, 0.0, 0.0])
setpoint_orient = np.array([37.0, -3.0, 4.0])

# initial sensor data
gyro_data = np.array([0, 0, 0])
acc_data = np.array([0, 0, 0])

# orientation
pos_yaw = []
pos_pitch = []
pos_roll = []

# time steps
numTimeSteps = 4000

for n in range(numTimeSteps):

    # simulate drag torques
    Dt_yaw = 500 * drag_coefficient * gyro_data[0]**2 * 0.24
    Dt_pitch = 500 * drag_coefficient * gyro_data[1]**2 * 0.4
    Dt_roll = 500 * drag_coefficient * gyro_data[2]**2 * 0.4

    # calculate the torques with PID
    yaw_torque = cntrl_yaw.output(orient[0], setpoint_orient[0])
    pitch_torque = cntrl_pitch.output(orient[1], setpoint_orient[1])
    roll_torque = cntrl_roll.output(orient[2], setpoint_orient[2])

    # net torques after subtracting drag
    netYaw = yaw_torque - Dt_yaw
    netPitch = pitch_torque - Dt_pitch
    netRoll = roll_torque - Dt_roll

    # calculate angular acceleration with noise
    acc_data = np.array([
        (netYaw / physics.I) + np.random.normal(0, acc_noise_std),
        (netPitch / physics.I) + np.random.normal(0, acc_noise_std),
        (netRoll / physics.I) + np.random.normal(0, acc_noise_std)
    ])

    # calculate angular velocities with angular acceleration and noise
    gyro_data = np.array([
        (acc_data[0] * dt) + np.random.normal(0, gyro_noise_std),
        (acc_data[1] * dt) + np.random.normal(0, gyro_noise_std),
        (acc_data[2] * dt) + np.random.normal(0, gyro_noise_std)
    ])

    # update current orientation and convert to degrees
    orient += (180 * gyro_data * dt/np.pi)

    # kalman filter prediction and update step
    lkf.predict(gyro_data)
    lkf.update(orient)

    # get estimated orientation from the kalman filter
    orient = lkf.get_orientation()

    # store the estimated orientation values
    pos_yaw.append(orient[0])
    pos_pitch.append(orient[1])
    pos_roll.append(orient[2])

figure, axis = plt.subplots(2,3)
axis[0,0].plot(np.array(range(len(x_state))), x_state)
axis[0,0].set_title("X State")
axis[0,1].plot(np.array(range(len(y_state))), y_state)
axis[0,1].set_title("Y State")
axis[0,2].plot(np.array(range(len(z_state))), z_state)
axis[0,2].set_title("Z State")

axis[1,0].plot(list(range(len(pos_yaw))), pos_yaw)
axis[1,0].set_title("Yaw State")
axis[1,1].plot(list(range(len(pos_pitch))), pos_pitch)
axis[1,1].set_title("Pitch State")
axis[1,2].plot(list(range(len(pos_roll))), pos_roll)
axis[1,2].set_title("Roll State")
plt.show()

    # thruster_array = np.array([[0.00, 0.00, 1.00, -1.0, -1.0, 0.00],
    #                             [0.00, 0.00, -1.0, -1.0, 1.00, 0.00],
    #                             [0.00, 0.00, -1.0, 1.00, -1.0, 0.00],
    #                             [0.00, 0.00, 1.00, 1.00, 1.00, 0.00],
    #                             [1.00, 1.00, 0.00, 0.00, 0.00, 1.00],
    #                             [-1.0, 1.00, 0.00, 0.00, 0.00, 1.00],
    #                             [-1.0, 1.00, 0.00, 0.00, 0.00, -1.0],
    #                             [1.00, 1.00, 0.00, 0.00, 0.00, -1.0]
    #                             ])