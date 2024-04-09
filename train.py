from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from airspace_wrapper import AirspaceMultiAgentEnv
from multi_agent_wrapper import MultiAgentWrapper
import wandb

wandb.init(
    entity="rovell",
    project="dynamic_pricing",
    monitor_gym=True,      
    save_code=True,
)

# Wrap your environment
wrapped_env = make_vec_env(lambda: MultiAgentWrapper(AirspaceMultiAgentEnv()), n_envs=1)

# Initialize the agent
model = PPO("MlpPolicy", 
            wrapped_env, 
            verbose=1,
            learning_rate = 0.0003,
            gamma = 0.995,
            n_steps = 100,
            ent_coef = 0.02,
            vf_coef = 0.5,
            max_grad_norm = 1,
            policy_kwargs = {"optimizer_kwargs":{"eps":1e-7}}
            )

# Train the agent
model.learn(total_timesteps=10000000)

# Save the model
model.save("multi_agent_airspace_model")