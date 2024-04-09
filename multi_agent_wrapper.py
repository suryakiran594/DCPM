import gym
import numpy as np
from gym.spaces import Box
import pdb

class MultiAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        total_observation_space_size = self._calculate_flattened_observation_space_size()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(total_observation_space_size,), dtype=np.float32)

        # Assuming action space remains unchanged
        self.action_space = self.env.action_space
        
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        
    def _calculate_flattened_observation_space_size(self):
        funds_size = 1 * self.env.num_agents
        priority_size = 1 * self.env.num_agents
        utm_price_size = 1 * self.env.num_agents
        consecutive_failures_size = 1 * self.env.num_agents
        return funds_size + priority_size + utm_price_size + consecutive_failures_size
    
    def reset(self):
        obs = self.env.reset()
        return self._flatten_obs(obs)

    def step(self, action):
        action_dict = self._unflatten_actions(action)
        obs, rewards, dones, info = self.env.step(action_dict)
        return self._flatten_obs(obs), self._flatten_rewards(rewards), self._flatten_dones(dones), info

    # def _flatten_obs(self, obs_dict):
    #     obs_list = [obs_dict[agent_id] for agent_id in sorted(obs_dict.keys())]
    #     return np.concatenate(obs_list, axis=0)
    
    def _flatten_obs(self, obs_dict):
        # Initialize lists to collect observation components
        funds_list = []
        priority_list = []
        utm_price_list = []
        consecutive_failures_list = []
        
        # Collect and prepare observation components
        for agent_id in sorted(obs_dict.keys()):
            agent_obs = obs_dict[agent_id]
            funds_list.append(agent_obs["funds"].reshape(-1))  # Flatten to 1D
            priority_list.append(np.array([agent_obs["priority"]]))  # Convert scalar to 1D array
            utm_price_list.append(agent_obs["utm_price"].reshape(-1))  # Flatten to 1D
            consecutive_failures_list.append(agent_obs["consecutive_failures"].reshape(-1))  # Flatten to 1D
        
        # Concatenate each list of observation components
        funds = np.concatenate(funds_list)
        priorities = np.concatenate(priority_list)
        utm_prices = np.concatenate(utm_price_list)
        consecutive_failures = np.concatenate(consecutive_failures_list)
        
        # Finally, concatenate all observation components into a single array
        flat_obs = np.concatenate([funds, priorities, utm_prices, consecutive_failures])
        
        return flat_obs


    def _flatten_rewards(self, rewards_dict):
        return np.sum([reward for reward in rewards_dict.values()])

    def _flatten_dones(self, dones_dict):
        return dones_dict['__all__']

    # def _unflatten_actions(self, actions):
    #     pdb.set_trace()
    #     action_dict = {}
    #     num_agents = len(actions) // self.action_space.n
    #     for i in range(num_agents):
    #         action_dict[f'agent_{i}'] = actions[i]
    #     return action_dict
    
    def _unflatten_actions(self, action):
        # Assuming the same action is applicable for all agents, which might not be what you want.
        action_dict = {f'agent_{i}': action for i in range(self.num_agents)}
        return action_dict
