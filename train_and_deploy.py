# Imports
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import sagemaker
from sagemaker.pytorch import PyTorchModel
import time

# Constants
SLIPPAGE_COEFFICIENT = 0.01
PRICE_IMPACT_COEFFICIENT = 0.05

# Custom Environment Class
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.remaining_shares = 1000  # Total shares to sell over a trading day (390 minutes)

        # Action and Observation Spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.remaining_shares = 1000
        observation = self._next_observation()
        return observation, {}  # Observation and empty info dictionary for gymnasium compatibility

    def step(self, action):
        action_value = action[0]
        shares_to_trade = int(action_value * 10)  # Scale the action to represent shares

        # Trading Logic
        if shares_to_trade > 0:
            self.remaining_shares -= shares_to_trade
            reward = -shares_to_trade  # Negative reward for each trade to minimize cost
        else:
            reward = 0  # No reward adjustment for no action

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.remaining_shares <= 0
        truncated = False  # For `gymnasium` compatibility
        observation = self._next_observation()
        info = {}  # Extra info dictionary (can add metrics if needed)

        return observation, reward, done, truncated, info  # Return 5 values as expected

    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row['bid_price_1'],
            row['bid_size_1'],
            self.remaining_shares,
            390 - self.current_step  # Example of time left in the day
        ], dtype=np.float32)

# Data Loading
data = pd.read_csv("C:/Users/anil sadgire/Blackhouse/Blockhouse-Work-Trial/data/AAPL_Quotes_Data.csv")

# Initialize the custom environment and wrap in DummyVecEnv
env = DummyVecEnv([lambda: TradingEnv(data)])

# Model setup and training
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("rl_model_sagemaker.zip")

# SageMaker Inference Script
def model_fn(model_dir):
    return SAC.load(f"{model_dir}/rl_model_sagemaker.zip")

def predict_fn(input_data, model):
    action, _states = model.predict(input_data)
    return action

# SageMaker Deployment Configuration
sagemaker_session = sagemaker.Session()
role = 'arn:aws:s3:::blackhousebbucket/train_and_deploy.py'  # Replace with actual SageMaker role ARN

pytorch_model = PyTorchModel(
    model_data='s3://blackhousebbucket/train_and_deploy.py',  # Replace with your S3 path
    role=role,
    entry_point='inference.py',
    framework_version='1.8.0',
    py_version='py3'
)

# Deploy the model
predictor = pytorch_model.deploy(instance_type='ml.m5.large', initial_instance_count=1)


# Wait for the endpoint to be ready
print("Waiting for endpoint deployment...")
time.sleep(300)  # Adjust time if needed for endpoint setup
print("Endpoint is deployed and ready for inference.")
