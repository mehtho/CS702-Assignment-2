import pygame
import random
import copy
import numpy as np
from scipy.optimize import minimize


pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


FLAP_DURATION = 10  
PREDICTION_HORIZON = 10  


class Bird:
    def __init__(self, x, y, vy):
        self.x = x
        self.y = y
        self.vy = vy
        self.w = 20
        self.h = 20


class Pipe:
    def __init__(self, x, h):
        self.x = x
        self.h = h
        self.w = 70
        self.gap = 200


def cost_function(action, bird, pipe):
    future_bird = copy.deepcopy(bird)
    for _ in range(PREDICTION_HORIZON):
        future_bird.vy += action * 15 - 9.8
        future_bird.y += future_bird.vy
        if future_bird.y > SCREEN_HEIGHT or future_bird.y < 0:
            return float('inf')
    return abs(pipe.h + pipe.gap / 2 - future_bird.y) 


def model_predictive_control(bird, pipe):
    initial_guess = np.zeros(PREDICTION_HORIZON)
    result = minimize(cost_function, initial_guess, args=(bird, pipe))
    return result.x[0]


# Main game loop
def main():
    clock = pygame.time.Clock()
    bird = Bird(50, random.randint(100, SCREEN_HEIGHT - 100), 0)  # Initialize bird at a random position
    pipe = Pipe(SCREEN_WIDTH - 50, random.randint(200, 300))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Perform MPC to get control signal
        control_signal = model_predictive_control(bird, pipe)

        # Update bird position
        bird.vy += control_signal * 15 - 9.8
        bird.y += bird.vy

        # Drawing
        screen.fill((255, 255, 255)) 
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(bird.x, bird.y, bird.w, bird.h))  # Draw bird
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pipe.x, 0, pipe.w, pipe.h))  # Draw bottom pipe
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pipe.x, pipe.h + pipe.gap, pipe.w, SCREEN_HEIGHT - (pipe.h + pipe.gap)))  # Draw top pipe

        pygame.display.flip()  
        clock.tick(30)  

    pygame.quit()

if __name__ == "__main__":
    main()
