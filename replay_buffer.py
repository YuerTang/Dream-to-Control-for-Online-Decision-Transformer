import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        self.trajectories = trajectories[:capacity]
        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    '''
    def add_new_trajs(self, new_trajs):
        
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
        else:
            self.trajectories[self.start_idx:self.start_idx + len(new_trajs)] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity
        self.trajectories = self.trajectories[:self.capacity]
        assert len(self.trajectories) <= self.capacity
    
    '''

    def add_new_trajs(self, new_trajs):
        # Convert numpy arrays to tensors if necessary
        converted_trajs = []
        for traj in new_trajs:
            converted_traj = {}
            for key, value in traj.items():
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if isinstance(value, np.ndarray):
                    converted_traj[key] = torch.from_numpy(value).float()  # Ensure all tensors are float for consistency
                else:
                    converted_traj[key] = value
            converted_trajs.append(converted_traj)

        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(converted_trajs)
        else:
            end_idx = self.start_idx + len(converted_trajs)
            if end_idx > self.capacity:
                overflow = end_idx - self.capacity
                self.trajectories[self.start_idx:] = converted_trajs[:-overflow]
                self.trajectories[:overflow] = converted_trajs[-overflow:]
            else:
                self.trajectories[self.start_idx:end_idx] = converted_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        self.trajectories = self.trajectories[:self.capacity]
        assert len(self.trajectories) <= self.capacity, "Replay buffer overflow error."
        print("Updated replay buffer size:", len(self.trajectories))
        
    

    def print_details(self):
        print("Replay Buffer Details:")
        if not self.trajectories:
            print("Buffer is empty.")
        else:
            for idx, traj in enumerate(self.trajectories):
                print(f"Trajectory {idx + 1}:")
                for key in ['states', 'actions', 'rewards', 'next_states', 'dones']:
                    if key in traj:
                        data = np.array(traj[key])
                        print(f"  {key.capitalize()}: {data.shape}, dtype={data.dtype}")
                    else:
                        print(f"  {key.capitalize()}: Key not found in trajectory.")

