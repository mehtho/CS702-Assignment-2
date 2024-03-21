import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.x = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # self.B = np.array([[(self.dt**2)/2, 0],
        #                     [0,(self.dt**2)/2],
        #                     [self.dt,0],
        #                     [0,self.dt]])
        self.C = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        self.R = np.array([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.x = self.A @ self.x
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.C @ self.x)
        I = np.eye(len(self.x))
        self.P = (I - (K @ self.C)) @ self.P
        return self.x[0:2]
