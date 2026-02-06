try:
    from isaacgym import gymtorch
except ImportError:
    gymtorch = None

import torch
import numpy as np


class VecEnvWithStateReset:
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.state_memory = []
        self.memory_reset_prob = 0.5
        self.reset_counts = {'mu': 0, 'memory': 0}
        
        # Pass through attributes
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.gym = env.gym
        self.sim = env.sim
        self.root_states = env.root_states
        self.dof_pos = env.dof_pos
        self.dof_vel = env.dof_vel
        self.dof_state = env.dof_state
    
    def step(self, actions):
        return self.env.step(actions)
    
    def reset(self):
        return self.env.reset()
    
    def reset_idx(self, env_ids):
        return self.env.reset_idx(env_ids)
    
    def get_observations(self):
        return self.env.get_observations()
    
    def get_privileged_observations(self):
        return self.env.get_privileged_observations()
    
    def add_current_states_to_memory(self):
        sample_rate = max(1, self.num_envs // 100)
        for i in range(0, self.num_envs, sample_rate):
            state = self._extract_state(i)
            self.state_memory.append(state)
        
        max_size = 10000
        if len(self.state_memory) > max_size:
            self.state_memory = self.state_memory[-max_size:]
    
    def _extract_state(self, env_idx):
        state = {
            'root_states': self.env.root_states[env_idx].cpu().numpy().copy(),
            'dof_pos': self.env.dof_pos[env_idx].cpu().numpy().copy(),
            'dof_vel': self.env.dof_vel[env_idx].cpu().numpy().copy(),
        }
        return state
    
    def reset_with_memory_sampling(self, env_ids):
        if len(env_ids) == 0:
            return self.env.get_observations()
        
        if torch.is_tensor(env_ids):
            env_ids_list = env_ids.cpu().numpy().tolist()
        else:
            env_ids_list = list(env_ids)
        
        memory_reset_ids = []
        normal_reset_ids = []
        
        for env_id in env_ids_list:
            if (np.random.random() < self.memory_reset_prob 
                and len(self.state_memory) > 0):
                memory_reset_ids.append(env_id)
                self.reset_counts['memory'] += 1
            else:
                normal_reset_ids.append(env_id)
                self.reset_counts['mu'] += 1
        
        if len(memory_reset_ids) > 0:
            for env_id in memory_reset_ids:
                state = self._sample_random_state()
                self._set_env_state(env_id, state)
        
        if len(normal_reset_ids) > 0:
            normal_reset_tensor = torch.tensor(normal_reset_ids, 
                                              dtype=torch.long, 
                                              device=self.device)
            self.env.reset_idx(normal_reset_tensor)
        
        return self.env.get_observations()
    
    def _sample_random_state(self):
        idx = np.random.randint(len(self.state_memory))
        return self.state_memory[idx]
    
    def _set_env_state(self, env_idx, state):
        # Set root states
        self.env.root_states[env_idx] = torch.from_numpy(
            state['root_states']).to(self.device)
        
        # Set DOF positions and velocities
        self.env.dof_pos[env_idx] = torch.from_numpy(
            state['dof_pos']).to(self.device)
        self.env.dof_vel[env_idx] = torch.from_numpy(
            state['dof_vel']).to(self.device)
        
        # Update DOF state - handle different possible shapes
        if len(self.env.dof_state.shape) == 3:
            # Shape: (num_envs, num_dofs, 2) where [:, :, 0] is pos, [:, :, 1] is vel
            self.env.dof_state[env_idx, :, 0] = self.env.dof_pos[env_idx]
            self.env.dof_state[env_idx, :, 1] = self.env.dof_vel[env_idx]
        elif len(self.env.dof_state.shape) == 2:
            # Shape: (num_envs * num_dofs, 2) - flattened version
            num_dofs = self.env.dof_pos.shape[1]
            start_idx = env_idx * num_dofs
            end_idx = start_idx + num_dofs
            self.env.dof_state[start_idx:end_idx, 0] = self.env.dof_pos[env_idx]
            self.env.dof_state[start_idx:end_idx, 1] = self.env.dof_vel[env_idx]
        
        # Tell Isaac Gym to update the simulation
        env_ids_int32 = torch.tensor([env_idx], dtype=torch.int32, device=self.device)
        
        # Import gymtorch here if needed
        if gymtorch is not None:
            self.env.gym.set_actor_root_state_tensor_indexed(
                self.env.sim,
                gymtorch.unwrap_tensor(self.env.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )
            
            self.env.gym.set_dof_state_tensor_indexed(
                self.env.sim,
                gymtorch.unwrap_tensor(self.env.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )
    
    def get_reset_statistics(self):
        total = sum(self.reset_counts.values())
        if total == 0:
            return {
                'mu_frac': 0.0,
                'memory_frac': 0.0,
                'memory_size': len(self.state_memory)
            }
        
        return {
            'mu_frac': self.reset_counts['mu'] / total,
            'memory_frac': self.reset_counts['memory'] / total,
            'memory_size': len(self.state_memory)
        }

