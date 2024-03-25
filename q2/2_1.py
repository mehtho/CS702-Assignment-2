import pygame
import random
import copy
from dataclasses import dataclass
import cv2
import numpy as np


pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


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
    Kp: float = 0.1
    Ki: float = 0.01
    Kd: float = 0.1
    error_accumulator: float = 0
    prev_error: float = 0

    def calc_input(self, sp: float, pv: float, threshold: float = 0) -> float:
        e = sp - pv
        P = self.Kp * e
        self.error_accumulator += e
        I = self.Ki * self.error_accumulator
        D = self.Kd * (e - self.prev_error)
        self.prev_error = e

        pid = P + I + D
        return 1 if pid > threshold else 0


def calculate_the_control_signal(bird: Bird, pipe: Pipe, pid_controller: PIDController) -> int:
    sp = pipe.h + pipe.gap / 2
    pv = bird.y
    u_jump = pid_controller.calc_input(sp, pv)
    return u_jump


def main():
    clock = pygame.time.Clock()
    bird = Bird(50, 300, 0, 0)
    pipe = Pipe(SCREEN_WIDTH - 50, random.randint(200, 300))
    pid_controller = PIDController()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('gameplay.mp4', fourcc, 30.0, (SCREEN_WIDTH, SCREEN_HEIGHT))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        u_jump = calculate_the_control_signal(bird, pipe, pid_controller)
        bird.vy += (u_jump * 15 - 9.8)  
        bird.y += bird.vy

        pipe.x -= 3
        if pipe.x < -pipe.w:
            pipe.x = SCREEN_WIDTH
            pipe.h = random.randint(200, 300)

        if bird.y > SCREEN_HEIGHT or bird.y < 0:
            print("Game over")
            running = False

        # Drawing
        screen.fill((255, 255, 255)) 
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(bird.x, bird.y, bird.w, bird.h)) 
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pipe.x, 0, pipe.w, pipe.h)) 
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pipe.x, pipe.h + pipe.gap, pipe.w, SCREEN_HEIGHT - (pipe.h + pipe.gap)))  # Draw top pipe

        pygame.display.flip()  #
        
        frame = pygame.surfarray.array3d(screen)
        frame = np.flip(frame, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

        clock.tick(30)  
    out.release()
    pygame.quit()

if __name__ == "__main__":
    main()
