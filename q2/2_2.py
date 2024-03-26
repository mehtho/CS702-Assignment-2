import numpy as np
import pygame
import random
import copy
from dataclasses import dataclass
from pyomo.environ import *
from pyomo.dae import *

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


def bird_motion(bird: Bird, u: float, dt: float, gravity: float = -50) -> Bird:
    """Updates the bird's y position and velocity."""
    new_bird = copy.deepcopy(bird)
    new_bird.y = bird.y + bird.vy * dt
    new_bird.vy = bird.vy + (u + gravity) * dt
    return new_bird

@dataclass
class Pipe:
    x: float
    h: float
    w: float = 70
    gap: float = 200


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


def calculate_the_control_signal(bird: Bird, pipe: Pipe, k: int):
    sp = pipe.h + pipe.gap / 2
    pv = bird.y - bird.h / 2
    if bird.x > pipe.x + pipe.w:    
        sp = SCREEN_HEIGHT / 2    
    u_jump = pid.calc_input(sp, pv)
    return u_jump


@dataclass
class MPCController:
    dt = 1 / 80              # time step
    gravity = -50            # gravity constant
    horizon = 15
    bird_h = 20
    nfe = 100

    def __init__(self):
        self.prev_v = 0
        self.prev_y = 300

    def get_reference_trajectory(self):
        pv_target = self.pv + np.arange(0, 1, 1 / self.horizon) * (self.sp - self.pv)
        return pv_target

    def create_model(self):
        ref = self.get_reference_trajectory()
        m = ConcreteModel()
        DT = self.horizon * self.dt
        m.t = ContinuousSet(bounds=(0, DT))
        m.u = Var(m.t)
        m.v = Var(m.t)
        m.y = Var(m.t)
        m.dvdt = DerivativeVar(m.v, wrt=m.t)
        m.dydt = DerivativeVar(m.y, wrt=m.t)
        m.c1 = Constraint(m.t, rule=lambda m, t: m.dvdt[t] == m.u[t] + self.gravity)
        m.c2 = Constraint(m.t, rule=lambda m, t: m.dydt[t] == m.v[t])

        m.v[m.t.first()].fix(self.prev_v)
        m.y[m.t.first()].fix(self.prev_y)

        def integral(m, t):
            indices = np.arange(self.horizon)
            pv_target = np.interp(t / self.dt, indices, ref)
            pv_current = m.y[t] - self.bird_h / 2
            obj = (pv_current - pv_target) ** 2
            return obj
        
        m.integral = Integral(m.t, wrt=m.t, rule=integral)    
        m.obj = Objective(expr=m.integral, sense=minimize)
        return m

    def predict_value(self, m, data):
        t_bar = np.array([value(t) for t in m.t])
        u_bar = np.array([value(data[t]) for t in m.t])
        u_params = np.polyfit(t_bar, u_bar, 2)
        u = np.polyval(u_params, dt)
        return u

    def calc_input(self, sp, pv):
        self.sp = sp
        self.pv = pv
        m = self.create_model()
        discretizer = TransformationFactory("dae.finite_difference")
        discretizer.apply_to(m, nfe=self.nfe, wrt=m.t, scheme="BACKWARD")
        solver = SolverFactory("ipopt")
        solver.solve(m, tee=False)
        u = self.predict_value(m, m.u)
        v = self.predict_value(m, m.v)
        y = self.predict_value(m, m.y)
        self.prev_v = v
        self.prev_y = y
        return u

pid = MPCController()


if __name__ == "__main__":

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
    while running:
        screen.fill(WHITE)

        u_jump = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    u_jump = 500

        u_jump = calculate_the_control_signal(bird, pipe, k)

        bird = bird_motion(bird, u_jump, dt)
        x, y = transform(bird.x, bird.y)
        bird_rect.y = y

        pipe, d_score = pipe_motion(pipe, bird.vx, dt)
        x, y = transform(pipe.x, pipe.h)
        bottom_pipe_rect = pygame.Rect(x, y, pipe.w, pipe.h)
        top_pipe_rect = pygame.Rect(x, 0, pipe.w, SCREEN_HEIGHT - pipe.h - pipe.gap)

        score += d_score
        bird.vx += d_score * 10

        pygame.draw.rect(screen, GREEN, bird_rect)
        pygame.draw.rect(screen, GREEN, bottom_pipe_rect)
        pygame.draw.rect(screen, GREEN, top_pipe_rect)

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        if bird_rect.colliderect(bottom_pipe_rect) or \
                bird_rect.colliderect(top_pipe_rect) or \
                bird.y + bird.h > 1.5 * SCREEN_HEIGHT or \
                bird.y < -0.5 * SCREEN_HEIGHT:
            running = False

        pygame.display.update()
        clock.tick(fps)

        k += 1  

    pygame.quit()

