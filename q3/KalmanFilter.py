import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, std):
        self.dt = dt
        self.x = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.C = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.array([[(self.dt)/4, 0, (self.dt)/4, 0],
                            [0, (self.dt)/4, 0, (self.dt)/4],
                            [.5, 0, .5, 0],
                            [0, .5, 0, .5]])
        self.R = np.array([[std, 0],
                           [0, std]])
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
