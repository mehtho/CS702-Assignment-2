import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load fingertip tracking data
data = pd.read_csv("./q3/txys_missingdata.csv")

# Define particle filter parameters
num_particles = 1000  # Number of particles
process_noise_std = 0.1  # Standard deviation of process noise
measurement_noise_std = 0.1  # Standard deviation of measurement noise

# Initialize particles
particles = np.random.rand(num_particles, 2) * np.array([1920, 1080])  # Initialize particles uniformly in the state space
weights = np.ones(num_particles) / num_particles  # Initialize weights uniformly

# Main loop
for i in range(len(data)):
    # Prediction
    # Add process noise to particles
    particles += np.random.normal(scale=process_noise_std, size=particles.shape)
    
    # Measurement update
    # Compute likelihood of each particle given the measurement
    # Here we assume a simple measurement likelihood based on Euclidean distance
    # You may need to adapt this depending on the characteristics of your measurement model
    measurements = np.array([data['x_px'][i], data['y_px'][i]])
    particle_measurements = particles - measurements
    likelihoods = np.exp(-0.5 * np.sum((particle_measurements / measurement_noise_std) ** 2, axis=1))
    
    # Normalize weights and handle NaN values
    likelihood_sum = np.sum(likelihoods)
    if likelihood_sum != 0 and not np.isnan(likelihood_sum):
        weights *= likelihoods / likelihood_sum
    else:
        weights.fill(1 / num_particles)  # Reset weights to uniform distribution
    
    # Resampling
    # Resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
    particles = particles[indices]
    weights = np.ones(num_particles) / num_particles
    
    # Visualization
    # Plot particles to visualize estimated trajectory
    plt.scatter(particles[:, 0], particles[:, 1], c='blue', marker='.', alpha=0.5)
    plt.scatter(measurements[0], measurements[1], c='red', marker='o')  # Plot current measurement
    plt.title('Particle Filter Trajectory Estimation')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.gca().invert_yaxis()  # Invert y-axis if necessary
    plt.grid(True)
    plt.pause(0.01)  # Pause to update the plot

plt.show()
