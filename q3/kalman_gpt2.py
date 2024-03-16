import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter  # Importing the KalmanFilter class from kalman_filter.py

# Load fingertip tracking data
data = pd.read_csv("./q3/txys_missingdata.csv")

# Create a KalmanFilter object
kalman_filter = KalmanFilter()

# Initialize state and covariance matrices
x_hat = np.array([data['x_px'][0], 0, data['y_px'][0], 0])
P = np.eye(4)

# Kalman filter
filtered_trajectory = []
for i in range(len(data)):
    # Prediction
    x_hat_minus, P_minus = kalman_filter.predict(x_hat)
    
    # Update
    z = np.array([data['x_px'][i], data['y_px'][i]])
    x_hat, P = kalman_filter.update(x_hat_minus, P_minus, z)
    
    filtered_trajectory.append((x_hat[0], x_hat[2]))

# # Save filtered trajectory points to CSV file
# filtered_trajectory_df = pd.DataFrame(filtered_trajectory, columns=['x', 'y'])
# filtered_trajectory_df.to_csv('./q3/filtered_trajectory.csv', index=False)

# Plot both original and filtered trajectories
plt.figure(figsize=(12, 6))

# Original trajectory
plt.subplot(1, 2, 1)
plt.scatter(data['x_px'], data['y_px'], color='red')
plt.title('Original Fingertip Trajectory')
plt.xlabel('X coordinate (pixels)')
plt.ylabel('Y coordinate (pixels)')
plt.gca().invert_yaxis()  # Invert y-axis to match the desired coordinate system
plt.grid(True)

# Trajectory after Kalman filter
plt.subplot(1, 2, 2)
x_coordinates = [point[0] for point in filtered_trajectory]
y_coordinates = [point[1] for point in filtered_trajectory]
plt.scatter(x_coordinates, y_coordinates, color='blue')
plt.title('Fingertip Trajectory after Kalman Filter')
plt.xlabel('X coordinate (pixels)')
plt.ylabel('Y coordinate (pixels)')
plt.gca().invert_yaxis()  # Invert y-axis to match the desired coordinate system
plt.grid(True)

plt.tight_layout()
plt.show()
