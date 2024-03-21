import cv2
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from dataclasses import dataclass

FRAME_RATE = 10

def scale_coordinates(x, y, ori_size=(1920, 1080), down_size=(1280, 720)):
    scale_x = down_size[0] / ori_size[0]
    scale_y = down_size[1] / ori_size[1]
    new_x = int(x * scale_x)
    new_y = int(y * scale_y)
    return new_x, new_y

@dataclass
class Particle:
    x: np.ndarray
    weight: float


def calc_input() -> np.ndarray:
    return np.array([[-15, 5]]).T

def motion_model(x, u):
    x[0, 0]  = x[0, 0] + u[0, 0]
    x[1, 0]  = x[1, 0] + u[1, 0]
    return x

def noised_input(u: np.ndarray, fps=FRAME_RATE) -> np.ndarray:
    d_vv, d_vw, d_wv, d_ww, = 5.0, 0.3, 3, 2.0
    cov = np.diag([d_vv ** 2, d_vw ** 2, d_wv ** 2, d_ww ** 2])
    pdf = multivariate_normal(cov=cov)
    noise = pdf.rvs()

    uv = u[0, 0] + noise[0] * np.sqrt(np.abs(u[0, 0]) / fps) + noise[1] * np.sqrt(np.abs(u[1, 0]) / fps)
    uw = u[1, 0] + noise[2] * np.sqrt(np.abs(u[0, 0]) / fps) + noise[3] * np.sqrt(np.abs(u[1, 0]) / fps)
    return np.array([[uv, uw]]).T


def observe(x: np.ndarray, l: np.ndarray):
    dx = l[0, 0] - x[0, 0]
    dy = l[1, 0] - x[1, 0]
    l = np.linalg.norm([dx, dy])
    phi = np.arctan2(dy, dx)
    return l, phi


def likelihood(observation: np.ndarray, estimation: np.ndarray, std_dev=1.0):
    distance = np.sqrt((estimation[0, 0] - observation[0, 0]) ** 2 + (estimation[1, 0] - observation[1, 0]) ** 2)
    return multivariate_normal.pdf(distance, mean=0, cov=std_dev**2)


def resample(particles: [Particle]) -> [Particle]:
    _particles = list(filter(lambda p: p.weight > 0.01, particles))
    _particles = sorted(_particles, key=lambda p: p.weight, reverse=True)

    while len(_particles) < NP:
        _particles = sorted(_particles, key=lambda p: p.weight, reverse=True)
        _particles[0].weight /= 2
        new_particle = Particle(_particles[0].x, _particles[0].weight)
        _particles.append(new_particle)

    return _particles


def main():
    x = np.array([[1650*2/3, 657*2/3]]).T
    particles = [Particle(np.copy(x), 1 / NP) for i in range(NP)]

    VideoCap = cv2.VideoCapture('./q3/hand_tracking_output.MOV')
    ControlSpeedVar = 50  #Lowest: 1 - Highest:100
    HiSpeed = 100    
    
    df = pd.read_csv("./q3/txys_missingdata.csv")

    old_centers = []
    while(True):
        centers = []
        # Read frame
        ret, frame = VideoCap.read()
     
        current_frame_index = (int(VideoCap.get(cv2.CAP_PROP_POS_FRAMES)) - 1) * 100
        if(current_frame_index<2000):
            continue
        filtered_rows = df[df["t_ms"] == current_frame_index]

        if not filtered_rows.empty:
            centers.append(np.array([[int(filtered_rows.x_px)], [int(filtered_rows.y_px)]]))            
            old_centers = centers

        if (len(centers) > 0):
            # Draw the detected circle
            scaled_x, scaled_y = scale_coordinates(int(centers[0][0]),int(centers[0][1]))

            cv2.circle(frame, (scaled_x, scaled_y), 10, (0, 191, 255), 2)
            x = np.array([[scaled_x, scaled_y]]).T
            x_new = np.array([[scaled_x, scaled_y]]).T
            x = x_new

        # Move the particles
        particles_new = []
        for particle in particles:
            px = particle.x
            pu = calc_input()
            pu = noised_input(pu)
            px_new = motion_model(px, pu)
            particles_new.append(Particle(px_new, particle.weight))
        particles = particles_new

        # Observation
        if (len(centers) > 0):
            observation = x

            for particle in particles:
                estimation = particle.x
                weight = likelihood(observation, estimation)
                particle.weight *= weight + 1e-10

            # Normalize weight
            total_weight = sum([p.weight for p in particles])
            for particle in particles:
                particle.weight /= total_weight

        # Resampling
        particles = resample(particles)

        for particle in particles:
            cv2.circle(frame, (int(particle.x[0, 0]), int(particle.x[1, 0])), 10, (229, 208, 12), 2)

        cx, cy = x[0, 0], x[1, 0]
        cv2.circle(frame, (int(cx), int(cy)), 10, (240, 240, 240), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    NP = 25
    main()
