# !pip install tensorflow==1.15.0 stable-baselines gym-anytrading gym
# !pip install mpi4py
# !pip install finta

import time

# FinTA stuff
from gym_anytrading.envs import StocksEnv

from stable_baselines import A2C
from stable_baselines import PPO2





# Create the new Environment
def add_signals(env):
  start = env.frame_bound[0] - env.window_size  # 0
  end = env.frame_bound[1]  # 150
  prices = env.df.loc[:, "High"].to_numpy()[start:end]
  signal_features = env.df.loc[:, ["High", "SMA", "RSI", "OBV"]].to_numpy()[start:end]
  return prices, signal_features

# Create the new Environment
class CustomStockEnv(StocksEnv):
  _process_data = add_signals


class Train_model():
    def __init__(self, env, filename, timesteps):
        self.env = env
        self.filename = filename
        self.timesteps = timesteps


    def A2C_model(self):
        start = time.time()
        a2c_model = A2C("MlpLstmPolicy", self.env, verbose=1)
        # Learn the model
        a2c_model.learn(total_timesteps=self.timesteps)
        end = time.time()

        # Save the model
        a2c_model.save(self.filename)
        print("Training time for A2C model: ", (end-start)/60, "minutes")

        return a2c_model

    def PPO_model(self):
        start = time.time()
        ppo_model = PPO2("MlpPolicy", self.env, verbose=1)
        # Learn the model
        ppo_model.learn(total_timesteps=self.timesteps)
        end = time.time()

        # Save the model
        ppo_model.save(self.filename)
        print("Training time for PPO model: ", (end-start)/60, "minutes")

        return ppo_model






