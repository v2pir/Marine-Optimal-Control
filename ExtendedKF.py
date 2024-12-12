import numpy as np

class EKF:
    def __init__(self, dt, drag_coefficient):
        self.dt = dt  # Time step
        self.drag_coefficient = drag_coefficient  # Drag coefficient in water

        # state vector [q0, q1, q2, q3]
        self.x = np.array([1, 0, 0, 0])

        # state covariance matrix
        self.P = np.eye(4) * 0.01

        # process noise covariance
        self.Q = np.eye(4) * 0.001

        # measurement noise covariance
        self.R = np.eye(3) * 0.05

    def predict(self, gyro):
        q = self.x
        wx, wy, wz = gyro

        # water drag to gyro data
        wx = wx * (1 - self.drag_coefficient)
        wy = wy * (1 - self.drag_coefficient)
        wz = wz * (1 - self.drag_coefficient)

        # quaternion derivative (using gyro data)
        F = 0.5 * np.array([
            [0, -wx, -wy, -wz],
            [wx,  0,  wz, -wy],
            [wy, -wz,  0,  wx],
            [wz,  wy, -wx,  0]
        ])

        dq = F @ q * self.dt
        self.x = self.x + dq
        self.x = self.x / np.linalg.norm(self.x)  # Normalize quaternion

        # jacobian of state transition
        Phi = np.eye(4) + F * self.dt

        # predict covariance
        self.P = Phi @ self.P @ Phi.T + self.Q

    def update(self, acc):
        q = self.x

        # predicted gravity in body frame
        H_acc = 2 * np.array([
            [-q[2], q[3], -q[0], q[1]],
            [ q[1], q[0], q[3], q[2]],
            [ q[0], -q[1], -q[2], q[3]]
        ])

        # predict measurement
        z_pred = H_acc @ q
        z = acc / np.linalg.norm(acc)

        # kalman gain
        S = H_acc @ self.P @ H_acc.T + self.R
        K = self.P @ H_acc.T @ np.linalg.inv(S)

        # update the state
        y = z - z_pred
        self.x = self.x + K @ y
        self.x = self.x / np.linalg.norm(self.x)  # Normalize quaternion

        # update the covariance
        self.P = (np.eye(4) - K @ H_acc) @ self.P

    def get_orientation(self):
        return self.x
    
    #https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
    def quat_to_eul(self, w, x, y, z):

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + (y * y))
        x_eul = np.degrees(np.arctan2(t0, t1)) # x angle

        t2 = 2.0 * (w * y - z * x)
        t2 = np.where(t2 > 1.0, 1.0, t2)

        t2 = np.where(t2 < -1.0, -1.0, t2)
        y_eul = np.degrees(np.arcsin(t2)) # y angle

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * ((y * y) + z * z)
        z_eul = np.degrees(np.arctan2(t3, t4)) # z angle

        return x_eul, y_eul, z_eul