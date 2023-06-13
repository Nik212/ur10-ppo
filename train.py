import argparse
import torch
from torch import nn
import pybullet
import numpy as np
from ur10_env import UR10
from gym.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for training loop",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", default=str, help="model to use")
    parser.add_argument("-n", "--num_envs", default=2, type=int, help="Number of enviroments to run")
    parser.add_argument("-l", "--length", default=1000, type=int, help="episode length")
    parser.add_argument("-e", "--max_episodes", default=1000, type=int, help="max episodes")
    parser.add_argument("-ci", "--ckpt_interval", default=1, type=int, help="Save checkopoint after each *-th episode")
    parser.add_argument("-v", "--visualize", default=False, type=bool, help="use wandb visualization")
    parser.add_argument("-r", "--render", default=False, type=bool, help="allow rendering")
    parser.add_argument("-rs", "--restore", default=False, type=bool, help="allow rendering")
    args = parser.parse_args()


    if args.render:
        pybullet.connect(pybullet.GUI)
        pybullet.resetSimulation()

    if args.visualize:
        run = wandb.init(project="sber-robotics-test")
        run.name = 'ppo-ur10-vectorized'
    

    N = args.num_envs
    max_episodes = args.max_episodes
    max_timesteps = args.length


    env = DummyVecEnv([lambda: FlattenObservation(UR10(is_train=True, is_dense=True))] * N)
    

    # learning parameters

    action_dim = env.action_space.shape[0]
    state_dim = state_dim = env.observation_space.shape[0]
    
    if args.model == 'ppo':
        from agents.ppo_agent import PPO
        from agents.ppo_agent import Memory

        from config import ppo_config
        if not args.restore:
            ckpt_path = f'checkpoints/ppo_agent/last_episode.pth' # add here you checkpoint path
            ckpt = None
        else:
            ckpt = f'checkpoints/ppo_agent/last_episode.pth'

        agent = PPO(state_dim, action_dim, ppo_config, restore=args.restore, ckpt=ckpt, use_wandb=args.visualize)
        memory = Memory()

    if args.model == 'ddpg':
        raise NotImplementedError

    running_reward = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1

            # Run old policy
            action = agent.select_action(state, memory).reshape((N, action_dim))
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward.mean() # calculates mean across N envs

        agent.update(memory)
        memory.clear_memory()
        time_step = 0
        if args.visualize:
            wandb.log({'total_episode_rewards': running_reward, 'done_ratio': sum(done)/len(done)}, step = i_episode)
        if i_episode % args.ckpt_interval == 0:
            if not args.restore:
                print(f'Episode {i_episode} :: saving current policy to {ckpt_path}')
                torch.save(agent.policy.state_dict(), ckpt_path)