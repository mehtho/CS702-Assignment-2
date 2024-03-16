import numpy as np
from scipy import linalg as la


class KalmanFilter:
    def __init__(self):
        dt = 1.
        R_var = 1
        Q_var = 0.1
        self.A = np.array([[1., dt, 0, 0], 
                           [0, 1., 0, 0],
                           [0, 0, 1., dt],
                           [0, 0, 0, 1.]])
        self.C = np.array([[1., 0, 0, 0],
                           [0, 0, 1., 0]])
        self.P = np.diag([0.5, 0.1] * 2)  # covariance matrix for the state estimate
        self.Q = np.diag([Q_var] * 4)  # covariance matrix for the process disturbance
        self.R = np.array([[R_var]])  # covariance matrix for the measurement noise

    def kalman_gain(self, P: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Calculate the Kalman gain."""
        S = C @ P @ C.T + R
        K = P @ C.T @ la.inv(S)
        return K

    def predict(self, x_hat: np.ndarray) -> np.ndarray:
        """Predict the next state."""
        x_hat_minus = self.A @ x_hat
        P_minus = self.A @ self.P @ self.A.T + self.Q
        return x_hat_minus, P_minus

    def update(self, x_hat_minus: np.ndarray, P_minus: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Update the state estimate based on the measurement."""
        K = self.kalman_gain(P_minus, self.C, self.R)
        x_hat = x_hat_minus + K @ (z - self.C @ x_hat_minus)
        P = (np.eye(len(x_hat)) - K @ self.C) @ P_minus
        return x_hat, P
