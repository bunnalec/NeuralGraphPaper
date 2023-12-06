import pygame
import sys
from batchedCartpole import CartPole
import torch

# Initialize Pygame
pygame.init()

# Constants
w, h = (600, 400)

# Pygame screen setup
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption('CartPole Environment')

clock = pygame.time.Clock()

env = CartPole()
state = env.initialize(1)

regret = 0

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Player input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = torch.Tensor([0])
    elif keys[pygame.K_RIGHT]:
        action = torch.Tensor([1])
    else:
        action = torch.Tensor([2])

    if keys[pygame.K_RETURN]:
        env.reset(torch.Tensor([0]).long())
    
    reward, done = env.step(action)
    reward = reward.item()
    done = done.item()

    regret -= reward

    state = env.get_state()

    x, v, theta, omega = state[0]
    x = x.item()
    theta = theta.item()
    scale = 300

    x *= scale

    print(f"Reward:{reward} | Regret:{regret} | done:{done} | x:{x} | theta:{theta}", end="\r")

    screen.fill((255, 255, 255))

    # Cart
    pygame.draw.rect(screen, (0, 0, 255), (x + w//2 - 50, h - 100, 100, 30))

    # Pole
    pole_center = (x + w//2, h - 100)
    pole_end = (pole_center[0] + scale//2 * pygame.math.Vector2(0, -1).rotate_rad(theta).x,
                pole_center[1] + scale//2 * pygame.math.Vector2(0, -1).rotate_rad(theta).y)
    pygame.draw.line(screen, (255, 0, 0), pole_center, (round(pole_end[0]), round(pole_end[1])), 5)

    # Display the updated screen
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(20)