import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_trajectory(df):
    # Extract x, y coordinates and timestamps
    x = df['x_px']
    y = df['y_px']
    timestamps = df['t_ms']

    # Create colormap: blue shades from light to dark
    color_map = cm.Blues_r

    # Create figure and axis
    plt.figure(figsize=(8, 8))  # Set the size of the plot

    # Plot the trajectory
    trajectory_plot = plt.scatter(x, y, c=timestamps, cmap=color_map, marker='o')
    plt.title('Trajectory Plot')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    # Add colorbar to indicate time
    cbar = plt.colorbar(trajectory_plot)
    cbar.set_label('Time (ms)')

    # # Set the aspect ratio to be equal for both axes
    # plt.gca().set_aspect('equal')

    # Set the origin to the top-left corner
    plt.gca().invert_yaxis()

    # Show the plot
    plt.show()
