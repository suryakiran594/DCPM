import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict
import time 
import wandb
import pdb

class AirspaceMultiAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_agents = 10
        self.utm_price = 100  # Initial UTM selling price

        self.action_space = Discrete(3)  # Three bidding options: {Bid10, Bid20, Bid30}
        self.observation_space = Dict({
            "funds": Box(low=0, high= 500, shape=(1,),dtype=np.int32),
            "priority": Discrete(2),  # 0 for low, 1 for high priority
            "utm_price": Box(low=50, high=150, shape=(1,),dtype=np.int32),
            "consecutive_failures": Box(low=0, high=3, shape=(1,),dtype=np.int32)  # Track consecutive failures
        })
        self.max_utm_price = 150
        # Initialize agents with random priorities and funds
        self.agents = {f"agent_{i}": self._initialize_agent() for i in range(self.num_agents)}

    def reset(self):
        self.utm_price = np.random.uniform(50, 150)  # Randomize UTM price each episode
        self.current_step = 0
        self.start_timers = {}
        for agent_id in self.agents.keys():
            self.agents[agent_id] = self._initialize_agent()  # Reset agents' state
            self.start_timers[agent_id]=time.time()
            time.sleep(2)
        return {agent_id: self._get_observation(agent_id) for agent_id in self.agents.keys()}

    def calculate_reward(self,bid_amount,success_criteria):
        if bid_amount  <= success_criteria:
          return -1
        elif (bid_amount > success_criteria ) and (bid_amount < self.max_utm_price):
          return 5
        else:
          return 10

    def step(self, action_dict):
        rewards = {}
        obs = {}
        done = {}
        self.end_timer = {}
        for agent_id, action in action_dict.items():
            agent_info = self.agents[agent_id]
            bid_amount = self._action_to_bid_amount(action)
            success = self._process_bid(agent_id, bid_amount)
            
            if agent_info["funds"] < bid_amount:
                agent_info["funds"] = 0
            
            agent_info["consecutive_failures"] = 0 if success else agent_info["consecutive_failures"] + 1
            
            # Check for dynamic priority update
            if agent_info["consecutive_failures"] >= 3:
                agent_info["priority"] = 1  # Increase priority after 3 failed attempts
                
            rewards[agent_id] = self.calculate_reward(bid_amount,success)
            obs[agent_id] = self._get_observation(agent_id)
            done[agent_id] = agent_info["funds"] <= 0
            
            agent_int_id = int(agent_id.split('_')[-1])
            wandb.log({f"{agent_int_id}/funds":agent_info["funds"],"step":self.current_step})
            wandb.log({f"{agent_int_id}/consecutive_failures":agent_info["consecutive_failures"],"step":self.current_step})
            wandb.log({f"{agent_int_id}/rewards":rewards[agent_id],"step":self.current_step})
            wandb.log({f"{agent_int_id}/priority":agent_info["priority"],"step":self.current_step})
            wandb.log({f"{agent_int_id}/done":int(done[agent_id]),"step":self.current_step})
            
        done['__all__'] = all(done.values())
        self.current_step += 1
        return obs, rewards, done, {}

    def _initialize_agent(self):
        return {
            "funds": np.random.uniform(0, 500),
            "priority": np.random.choice([0, 1]),
            "consecutive_failures": 0
        }

    def _get_observation(self, agent_id):
        agent_info = self.agents[agent_id]
        return {
            "funds": np.array([agent_info["funds"]]),
            "priority": agent_info["priority"],
            "utm_price": np.array([self.utm_price]),
            "consecutive_failures": np.array([agent_info["consecutive_failures"]])
            }

    def _action_to_bid_amount(self, action):
        return (action + 1) * 50

    def _process_bid(self, agent_id, bid_amount):
        agent_info = self.agents[agent_id]
        success_criteria = self.utm_price * 0.8 * (0.9 if agent_info["priority"] == 1 else 1)
        return bid_amount >= success_criteria

    def render(self,render_array='human'):
      pass

    def close(self):
      pass