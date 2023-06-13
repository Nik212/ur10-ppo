import argparse
import pybullet
from ur10_env import UR10
from gym.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for training loop",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", default=1, help="model to use")
    parser.add_argument("-l", "--length", default=1000, type=int, help="episode length")
    parser.add_argument("-r", "--render", default=False, type=bool, help="allow rendering")
    args = parser.parse_args()

    if args.render:
        pybullet.connect(pybullet.GUI)
        pybullet.resetSimulation()

    max_timesteps = args.length


    env = DummyVecEnv([lambda: FlattenObservation(UR10(is_train=True, is_dense=True))] * 1)
    

    # learning parameters

    action_dim = env.action_space.shape[0]
    state_dim = state_dim = env.observation_space.shape[0]
    
    if args.model == 'ppo':
        from agents.ppo_agent import PPO
        from agents.ppo_agent import Memory

        from config import ppo_config
        ckpt = f'checkpoints/ppo_agent/last_episode.pth' # add here you checkpoint path
        agent = PPO(state_dim, action_dim, ppo_config, restore=True, ckpt=ckpt, use_wandb=False)
        memory = Memory()


    if args.model == 'ddpg':
        raise NotImplementedError


    running_reward = 0
    time_step = 0
    
    # training loop
    state = env.reset()
    for t in range(max_timesteps):
        time_step += 1

        # Run old policy
        action = agent.select_action(state, memory).reshape((1, action_dim))
        state, reward, done, _ = env.step(action)

        if done:
            break

        running_reward += reward.mean() # calculates mean across N envs