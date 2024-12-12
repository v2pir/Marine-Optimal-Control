import numpy as np
from MPPI import model, MPPI
import matplotlib.pyplot as plt
from ExtendedKF import EKF
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

# seconds
dt = 0.01

# drag coefficient specific to water
drag_coefficient = 0.7

# Create Kalman Filter object
ekf = EKF(dt, drag_coefficient)

# Controller for yaw, pitch, and roll
cntrl_yaw = Controller(10, 0, 0)
cntrl_pitch = Controller(10, 0, 0)
cntrl_roll = Controller(10, 0, 0)

# initial position
init_orient = np.array([0.0, 0.0, 0.0])
# setpoint
setpoint_orient = np.array([37.0, -3.0, 4.0])

# initial gyroscope and accelerometer data (from AHRS)
gyro_data = [0, 0, 0]
acc_data = [0, 0, 0]

# position --> for later graphing
pos_yaw = []
pos_pitch = []
pos_roll = []

# simulating over this many timesteps
numTimeSteps = 4000

for n in range(numTimeSteps):

    Dt_yaw = 500 * drag_coefficient * gyro_data[0]**2 * 0.24 # drag torque
    Dt_pitch = 500 * drag_coefficient * gyro_data[0]**2 * 0.4 # drag torque
    Dt_roll = 500 * drag_coefficient * gyro_data[0]**2 * 0.4 # drag torque

    yaw_torque = cntrl_yaw.output(init_orient[0], setpoint_orient[0])
    pitch_torque = cntrl_pitch.output(init_orient[1], setpoint_orient[1])
    roll_torque = cntrl_roll.output(init_orient[2], setpoint_orient[2])

    netYaw = yaw_torque - Dt_yaw
    netPitch = pitch_torque - Dt_pitch
    netRoll = roll_torque - Dt_roll

    acc_data[0] = netYaw/physics.I
    acc_data[1] = netPitch/physics.I
    acc_data[2] = netRoll/physics.I

    gyro_data[0] = acc_data[0] * dt
    gyro_data[1] = acc_data[1] * dt
    gyro_data[2] = acc_data[2] * dt

    # Prediction and update
    ekf.predict(gyro_data)
    ekf.update(acc_data)

    yaw = 180 * (gyro_data[0] * dt)/np.pi
    pitch = 180 * (gyro_data[1] * dt)/np.pi
    roll = 180 * (gyro_data[2] * dt)/np.pi

    # # Convert to quaternions
    # rot = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True)
    # ekf.x = rot.as_quat()

    # # Get orientation after filter is applied
    # quaternion = ekf.get_orientation()

    # # Get euler angles again
    # euler_angles = ekf.quat_to_eul(quaternion[0], quaternion[1], quaternion[2], quaternion[3])

    euler_angles = np.array([yaw, pitch, roll])

    # Update state
    init_orient += euler_angles

    pos_yaw.append(init_orient[0])
    pos_pitch.append(init_orient[1])
    pos_roll.append(init_orient[2])

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