import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from plot import plot_trajectory

# Load fingertip tracking data
data = pd.read_csv("./q3/txys_missingdata_padded.csv")

# Create a KalmanFilter object
kalman_filter = KalmanFilter()

# Initialize state and covariance matrices
x_hat = np.array([data['x_px'][0], 0, data['y_px'][0], 0])
P = np.eye(4)

# Kalman filter
filtered_trajectory = []
for i in range(len(data)):
    # Prediction
    x_hat = kalman_filter.predict(x_hat)
    
    # Update if there is a measurement
    if not np.isnan(data['x_px'][i]):
        z = np.array([data['x_px'][i], data['y_px'][i]])
        x_hat = kalman_filter.update(x_hat, z)
    
    filtered_trajectory.append((x_hat[0], x_hat[2]))

# Save filtered trajectory points to CSV file
filtered_trajectory_df = pd.DataFrame(filtered_trajectory, columns=['x', 'y'])
filtered_trajectory_df.to_csv('./q3/filtered_trajectory.csv', index=False)

# Plot both original and filtered trajectories
plt.figure(figsize=(12, 6))

# Original trajectory
plt.subplot(1, 2, 1)
plot_trajectory(data['x_px'], data['y_px'], 'Original Fingertip Trajectory')

# Trajectory after Kalman filter
plt.subplot(1, 2, 2)
x_coordinates = [point[0] for point in filtered_trajectory]
y_coordinates = [point[1] for point in filtered_trajectory]
plot_trajectory(x_coordinates, y_coordinates, 'Fingertip Trajectory After Kalman Filter')

plt.tight_layout()
plt.show()
