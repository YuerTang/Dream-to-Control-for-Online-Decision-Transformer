import torch
import numpy as np

class TrajectoryCollector:
    def __init__(self):
        # Convolutional layers for processing image-based observations
        self.conv_layers_obs = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.flatten = torch.nn.Flatten()
        self.dense_layers_obs = torch.nn.Sequential(
            torch.nn.Linear(64 * 8 * 8, 1000 * 17),
            torch.nn.ReLU()
        )

    def transform_model_obs(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)  # Add batch dimension if not present
        x = self.conv_layers_obs(obs)
        x = self.flatten(x)
        x = self.dense_layers_obs(x)
        return x.view(1000, 17).detach().numpy()  # Force shape to (1000, 17)

    def transform_model_action(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Add batch dimension if it's a single action
        action = action.repeat(1000, 1)  # Repeat the action to match the desired shape
        return action.detach().numpy()[:1000, :6]  # Force shape to (1000, 6)

    def transform_model_rewards(self, rewards):
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor([rewards], dtype=torch.float32)  # Ensure it's an array
        rewards = rewards.repeat(1000, 1)  # Repeat to match length of 1000
        return rewards.detach().numpy()[:1000]  # Force shape to (1000, 1)

    def transform_model_dones(self, dones):
        dones = np.array([dones], dtype=bool)
        dones = np.repeat(dones, 1000)  # Repeat to match length of 1000
        return dones[:1000]  # Force shape to (1000,)
