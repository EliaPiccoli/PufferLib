import hackatari
from hackatari import HackAtari

import pygame

"""Modifications
Breakout:
    - strengthX, X parameter
    - driftX, X in {r,l}
    - inverse_gravity
    - color_pX, X in {0,4}
    - color_bX, X in {0,4}
    - color_rXY, X, Y in {0,4}
    
MsPacman:
    - caged_ghosts
    - disable_orange
    - disable_red
    - disble_cyan
    - powerX, X in {0,4}
    - inverted
    - change_levelX, X in {0,3}
    
Pong:
    - lazy_enemy
    - up_driftX, X in {1,5}
    - down_driftX, X in {1,5}
    - left_driftX, X in {1,5}
    - right_driftX, X in {1,5}
    
Seaquest:
    - unlimited_oxygen
    - gravity
    - disable_enemies
    - random_color_enemies

SpaceInvaders:
    - disable_shield_left
    - disable_shield_right
    - disable_shield_middle
    - disable_shields
    - curved
    - relocateXX, X in {35,53}
"""

env = HackAtari("Breakout", ["strength5",], render_mode="rgb_array") # human, rgb_array

# define policy
policy = None

pygame.init()

for i in range(10):
    obss = []
    obs, _ = env.reset()
    done = False
    nstep = 1

    tr = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        action = policy(torch.Tensor(obs).unsqueeze(0))[0] if policy else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        tr += reward

        if reward:
            print(reward)

        if terminated or truncated:
            print(info, tr)
            tr = 0
            env.reset()

        nstep += 1
        env.render(env._state_buffer_rgb[-1])

env.close()