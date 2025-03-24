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
    - color_pX, X in {0,4}
    - color_bX, X in {0,4}
    
Pong:
    - lazy_enemy
"""
def color_variations(num_comb=25):
    val1 = range(5)
    val2 = range(5)
    
    combinations = []
    for v1 in val1:
        for v2 in val2:
            combinations.append(("color_p"+str(v1), "color_b"+str(v2)))
    return combinations[:num_comb]

def env_creator(env_name, modification):
    env = HackAtari(env_name, modification, obs_mode="ori", buffer_window_size=1, render_mode="rgb_array") # human, rgb_array
    env = GrayScaleObservation(env)
    env = pufferlib.postprocess.ResizeObservation(env, downscale=2)
    env = AtariPostprocessor(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return env

def pong_hacktari_eval(model, device):
    policy = model

    # here define the environment name
    env_name = "Pong"

    import time
    t = time.time()
    print(env_name, " hackatari time:", t)
    with open(f"stats_{env_name}_{t}.txt", "w") as f:
        episodes = 50
        
        # here define the modification combinations 
        # using  a function to generate them or
        # manually write them 
        combs = [("lazy_enemy")]
        # combs = [("lazy_enemy","up_driftX1", "down_drift2"), 
        #          ("lazy_enemy","up_driftX1", "down_drift2"),]
        
        print("total combinations: ", len(combs))
        
        for combination in combs:
            episodic_returns = []
            
            name = f"{env_name}_{combination}"
            
            f.write(f"Combination: {combination}\n")
            
            print("*"  * 50)
            print(f"Combination: {combination}")
            print("*"  * 50)
            
            env = env_creator(env_name=env_name, modification=combination)
            driver = env

            pygame.init()

            # make video 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'./videos/{name}_{t}.mp4',fourcc, 30.0, (160,210))
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
                            with torch.no_grad():
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

def breakout_hacktari_eval(model, device):
    policy = model

    # here define the environment name
    env_name = "Breakout"

    import time
    t = time.time()
    print(env_name, " hackatari time:", t)
    with open(f"stats_{env_name}_{t}.txt", "w") as f:
        episodes = 50
        
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
            
            env = env_creator(env_name=env_name, modification=combination)
            driver = env

            pygame.init()

            # make video 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'./videos/{name}_{t}.mp4',fourcc, 30.0, (160,210))
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
                            with torch.no_grad():
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

if __name__ == "__main__":
    breakout_hacktari_eval(None, 'cpu')
    # pong_hacktari_eval(None, 'cpu')