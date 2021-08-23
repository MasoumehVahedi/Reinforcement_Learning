import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from finta import TA

from customEnvironment import CustomStockEnv



def read_file(path):
    # Read the txt file
    df = pd.read_csv(path)
    # we need to convert the Date column to datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort date from oldest to newest
    df.sort_values("Date", ascending=True, inplace=True)
    # Set date as index
    df.set_index("Date", inplace=True)

    return df


# Calculate SMA, OBV, and RSI from TA
def cal_TA_indices(df):
    # Add, three indices from TA library to the dataframe
    df["SMA"] = TA.SMA(df, 12)
    df["RSI"] = TA.RSI(df)
    df["OBV"] = TA.OBV(df)

    # Fill NaN values
    df.fillna(0, inplace=True)
    return df


def evaluate_env(df, model):
    env_test = CustomStockEnv(df=df, frame_bound=(12, 50), window_size=12)
    obs = env_test.reset()
    while True:
        obs = obs[np.newaxis, ...]  # Reshape observation to work on non-vectorise environment
        action, _state = model.predict(obs)  # Using prediction model instead of random action in test environment
        obs, rewards, done, info = env_test.step(action)
        if done:
            print("info", info)
            break
    # Visulaise the environment
    plt.figure(figsize=(15, 12))
    plt.cla()
    env_test.render_all()
    plt.show()


