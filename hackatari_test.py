import cv2
import hackatari
from hackatari import HackAtari
from gymnasium.wrappers import GrayScaleObservation
import pufferlib
from pufferlib.environments.atari.environment import AtariPostprocessor
import pygame
import torch
import numpy as np

from tqdm import tqdm

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
def color_variations(num_comb=25):
    val1 = range(5)
    val2 = range(5)
    
    combinations = []
    for v1 in val1:
        for v2 in val2:
            combinations.append(("color_p"+str(v1), "color_b"+str(v2)))
    return combinations[:num_comb]

def env_creator(*args, **kwargs):
    modification = []
    name = "Breakout"
    
    if "modification" in kwargs:
        modification = kwargs["modification"]
    if "env_name" in kwargs:
        name = kwargs["env_name"]
    env = HackAtari(name, modification, obs_mode="ori", buffer_window_size=1, render_mode="rgb_array") # human, rgb_array
    env = GrayScaleObservation(env)
    env = pufferlib.postprocess.ResizeObservation(env, downscale=2)
    env = AtariPostprocessor(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return env

# define policy
model = torch.load('/home/malio/PufferLib/experiments/breakout-663eb907/model_000382.pt', map_location='cuda:3', weights_only=False)

model.eval()
print("Model loaded")
policy = model

# here define the environment name
env_name = "Breakout"
device = torch.device("cuda:3")

with open(f"stats_{env_name}.txt", "w") as f:
    episodes = 10
    
    # here define the modification combinations 
    # using  a function to generate them or
    # manually write them 
    combs = color_variations()
    # combs = [("lazy_enemy","up_driftX1", "down_drift2"), 
    #          ("lazy_enemy","up_driftX1", "down_drift2"),]
    
    print("total combinations: ", len(combs))
    
    for combination in combs:
        episodic_returns = []
        
        name = f"{env_name}_"
        for c in combination:
            name += c + "_"
        
        f.write(f"Combination: {combination}\n")
        
        print("*"  * 50)
        print(f"Combination: {combination}")
        print("*"  * 50)
        modification = combination
        
        env = env_creator( env_name=env_name)
        driver = env

        pygame.init()

        # make video 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'./videos/{name}.mp4',fourcc, 30.0, (160,210))
        print(f"Starting {name}...")

        f.write("*" * 50 + "\n")
        with tqdm(total=episodes) as pbar:
            for i in range(episodes):
                obss = []
                obs, _ = env.reset()
                done = False

                episodic_return = 0
                while not done:
                    render = driver.render()
                    out.write(render)
                    if policy:
                        action = policy(torch.as_tensor(obs, device=device).unsqueeze(0))[0].cpu().numpy()
                    else: # random action
                        action = driver.action_space.sample()

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episodic_return += reward

                    if terminated or truncated:
                        f.write(f"info: {info}, total reward: {episodic_return}\n")
                        episodic_returns.append(episodic_return)
                        episodic_return = 0
                        pbar.update(1)
                        done = True
                        env.reset()
        returns = np.array(episodic_returns)
        
        f.write(f"Total reward: {np.sum(returns)}\n")
        f.write(f"mean: {np.mean(returns)}\n")
        f.write(f"std: {np.std(returns)}\n")
        f.write(f"max: {np.max(returns)}\n")
        f.write(f"min: {np.min(returns)}\n")
        f.write("*" * 50 + "\n")
        out.release()
        cv2.destroyAllWindows()
        env.close()