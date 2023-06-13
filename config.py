from dataclasses import dataclass

@dataclass
class ppo_config:
    action_std = 0.5
    lr = 0.0003
    betas = [0.9, 0.990]
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    update_timestep = 64
    print_interval = 10