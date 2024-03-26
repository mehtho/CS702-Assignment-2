import pygame
import random
import copy
from dataclasses import dataclass
import cv2
import numpy as np

pygame.init()
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
WHITE = (240, 240, 240)
GREEN = (0, 200, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
transform = lambda x, y: (x, SCREEN_HEIGHT - y)
@dataclass
class Bird:
    x: float
    y: float
    vx: float
    vy: float
    w: float = 20
    h: float = 20

@dataclass
class Pipe:
    x: float
    h: float
    w: float = 70
    gap: float = 200


@dataclass
class PIDController:
    Kp: float = 0.1                 # proportional gain
    Ki: float = 0.01                # integral gain
    Kd: float = 0.1                 # derivative gain
    error_accumulator: float = 0    # error accumulator
    prev_error: float = 0           # previous error
    prev_u: float = 0               # previous control signal
    dt: float = 1 / 50              # time step
    gravity: float = -50            # gravity constant
    alpha: float = 50               # constant for velocity calculation

    def calc_input(self, sp: float, pv: float, umin: float = -100, umax: float = 100) -> float:
        e = sp - pv
        P = self.Kp * e
        self.error_accumulator += e
        I = self.Ki * self.error_accumulator
        D = self.Kd * (e - self.prev_error)
        self.prev_error = e
        u = np.clip(P + I + D, umin, umax)
        v = self.alpha * (u - self.prev_u) / self.dt - self.gravity
        self.prev_u = u
        return v
    

def bird_motion(bird: Bird, u: float, dt: float, gravity: float = -50) -> Bird:
    """Updates the bird's y position and velocity."""
    new_bird = copy.deepcopy(bird)
    new_bird.y = bird.y + bird.vy * dt
    new_bird.vy = bird.vy + (u + gravity) * dt
    return new_bird


def pipe_motion(pipe: Pipe, vx: float, dt: float) -> (Pipe, int):
    """Updates the pipe"""
    new_pipe = copy.deepcopy(pipe)
    new_pipe.x -= vx * dt
    d_score = 0
    if new_pipe.x < -pipe.w:
        new_pipe.x = SCREEN_WIDTH
        new_pipe.h = random.randint(200, 300)
        d_score = 1
    return new_pipe, d_score


def calculate_the_control_signal(bird: Bird, pipe: Pipe, pid_controller: PIDController) -> int:
    """Calculate the control signal for the bird."""
    sp = pipe.h + pipe.gap / 2
    pv = bird.y + bird.h / 2  # Adjusted to consider the center of the bird
    u_jump = pid_controller.calc_input(sp, pv)
    return u_jump


pid_controller = PIDController()


def main():

    bird = Bird(50, 300, 30, 0)
    x, y = transform(bird.x, bird.y)
    bird_rect = pygame.Rect(x, y, bird.w, bird.h)

    pipe_height = random.randint(50, 100)
    pipe = Pipe(SCREEN_WIDTH - 50, pipe_height)

    x, h = transform(pipe.x, pipe.h)
    bottom_pipe_rect = pygame.Rect(x, 0, pipe.w, h)

    x, y = transform(pipe.x, pipe.h + pipe.gap)
    top_pipe_rect = pygame.Rect(x, y, pipe.w, SCREEN_HEIGHT - y)

    clock = pygame.time.Clock()
    running = True
    fps = 30
    dt = 1 / fps

    score = 0
    k = 0  

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        u_jump = calculate_the_control_signal(bird, pipe, pid_controller)
        bird = bird_motion(bird, u_jump, 1 / 30)  # Assuming 30 fps
        pipe, d_score = pipe_motion(pipe, bird.vx, 1 / 30)
        score += d_score
        bird.vx += d_score * 10

        pygame.draw.rect(screen, GREEN, bird_rect)
        pygame.draw.rect(screen, GREEN, bottom_pipe_rect)
        pygame.draw.rect(screen, GREEN, top_pipe_rect)
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        # Check for collision with pipes or out of bounds
        if bird.y > SCREEN_HEIGHT or bird.y < 0 or \
           (bird.x + bird.w > pipe.x and bird.x < pipe.x + pipe.w and \
           (bird.y < pipe.h or bird.y + bird.h > pipe.h + pipe.gap)):
            print("Game over")
            running = False  

        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(bird.x, bird.y, bird.w, bird.h))
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pipe.x, 0, pipe.w, pipe.h))
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pipe.x, pipe.h + pipe.gap, pipe.w, SCREEN_HEIGHT - (pipe.h + pipe.gap)))  # Draw top pipe

        pygame.display.flip()

        frame = pygame.surfarray.array3d(screen)
        frame = np.flip(frame, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pygame.display.update()

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
