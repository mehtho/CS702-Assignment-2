import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

data = pd.read_csv("./q3/txys_missingdata_padded.csv")

kalman_filter = KalmanFilter()

# Kalman filter
x_meas = kalman_filter.filter(data)
filtered_trajectory = x_meas[:, :, [0, 2]]  # Pick (x, y)

# Save file
np.savez('./q3/filtered_trajectory.npz', filtered_trajectory=filtered_trajectory)

# Plot
plt.style.use("seaborn-v0_8")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original trajectory
ax[0].scatter(data['x_px'], data['y_px'], color='blue')
ax[0].set_title('Original Fingertip Trajectory')
ax[0].set_xlabel('X coordinate (pixels)')
ax[0].set_ylabel('Y coordinate (pixels)')
ax[0].set_xlim(right=1920) 
ax[0].set_ylim(top=1080)
ax[0].invert_yaxis()  # The origin is at the top-left corner
ax[0].grid(True)

# Trajectory after Kalman filter
for i in range(len(x_meas)):
    ax[1].scatter(x_meas[i, :, 0], x_meas[i, :, 1], s=5, alpha=0.3)

ax[1].set_title('Fingertip Trajectory After Kalman Filter')
ax[1].set_xlabel('X coordinate (pixels)')
ax[1].set_ylabel('Y coordinate (pixels)')
ax[1].set_xlim(right=1920) 
ax[1].set_ylim(top=1080)
ax[1].invert_yaxis()  # The origin is at the top-left corner
ax[1].grid(True)

plt.tight_layout()
plt.show()
