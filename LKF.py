import numpy as np

# linear kalman filter
class LKF:
    def __init__(self, dt, drag_coefficient):
        self.dt = dt
        self.drag_coefficient = drag_coefficient
        self.x = np.zeros(6)  # state vector [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.P = np.eye(6) * 1.0  # covariance
        self.Q = np.eye(6) * 0.5  # process noise
        self.R = np.eye(3) * 1.0  # measurement noise
        self.F = np.eye(6)  # state transition matrix
        self.F[:3, 3:] = np.eye(3) * self.dt  # linearized state transition for angular velocity
        self.H = np.zeros((3, 6))  # measurement matrix
        self.H[0, 0] = 1  # roll
        self.H[1, 1] = 1  # pitch
        self.H[2, 2] = 1  # yaw
    
    # predict state with past data
    def predict(self, gyro):
        gyro = np.array(gyro) * (1 - self.drag_coefficient)  # Apply drag
        self.x[:3] += self.x[3:] * self.dt  # Integrate angular rates
        self.x[3:] = gyro  # Update angular velocities
        self.P = self.F @ self.P @ self.F.T + self.Q  # Predict covariance
    
    # update state with new data
    def update(self, angles):
        z = np.array(angles)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    # return yaw, pitch, roll
    def get_orientation(self):
        return self.x[:3]