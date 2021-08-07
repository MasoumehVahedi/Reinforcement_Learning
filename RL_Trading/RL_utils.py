# pip install tensorflow==1.15.0 stable-baselines gym-anytrading gym

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt

import gym

from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv



############## Test the Environment ###############

def test_environment(df):
    # window_size : is a parameter for spedifying the previous time steps for trading
    # # Note: the first parameter of frame_bound should be equal the window_size
    environment = gym.make("stocks-v0", df=df, frame_bound=(5, 150), window_size=5)
    state = environment.reset()
    while True:
        action = environment.action_space.sample()  # Take some random actions
        n_state, reward, done, info = environment.step(action)  # Apply action to environment
        if done:
            print("info", info)
            break

    # Visulaise the environment
    plt.figure(figsize=(15, 12))
    plt.cla()
    environment.render_all()
    plt.show()

################# Build Environment using A2C algorithm ###############

def train_model(df):
    environment_maker = lambda: gym.make("stocks-v0", df=df, frame_bound=(5, 150), window_size=5)
    environment = DummyVecEnv([environment_maker])

    # Train the model
    model = A2C("MlpLstmPolicy", environment, verbose=1)
    # Learn the model
    model.learn(total_timesteps=100000)

    return model

################ Evaluate model ################

def eval_model(df, model):
    # Creat a new environment
    new_env = gym.make("stocks-v0", df=df, frame_bound=(80, 130), window_size=5)
    obs = new_env.reset()
    while True:
        obs = obs[np.newaxis, ...]  # Reshape observation to work on non-vectorise environment
        action, _state = model.predict(obs)  # Using prediction model instead of random action in test environment
        obs, rewards, done, info = new_env.step(action)
        if done:
            print("info", info)
            break

    # Visulaise the environment
    plt.figure(figsize=(15, 12))
    plt.cla()
    new_env.render_all()
    plt.show()
