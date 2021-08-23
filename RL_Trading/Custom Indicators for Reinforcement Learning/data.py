# Gym stuff
import gym
import gym_anytrading

from stable_baselines.common.vec_env import DummyVecEnv

from customEnvironment import CustomStockEnv
from customEnvironment import Train_model
from utils_for import read_file
from utils_for import cal_TA_indices
from utils_for import evaluate_env



if __name__ == "__main__":
    path = "/inputs/Data/Stocks/alex.us.txt"
    df = read_file(path)
    print(df.head())
    print(df.info())
    print(df.dtypes)
    print("Min Date : {}".format(df.index.min()))
    print("Max Date : {}".format(df.index.max()))

    # Create the Environment
    env = gym.make("stocks-v0", df=df, frame_bound=(5, 150), window_size=5)

    # Calculate SMA, OBV, and RSI from TA
    df = cal_TA_indices(df)

    # Split data to train and test dataset
    split_date = "2016-02-25"
    df_train = df[:split_date]
    df_test = df[split_date:]
    print("Shape of train data : {}".format(df_train.shape))
    print("Shape of test data : {}".format(df_test.shape))

    # Build Environment and Train
    new_env = CustomStockEnv(df=df_train, frame_bound=(12, 50), window_size=12)
    env_maker = lambda: new_env
    env = DummyVecEnv([env_maker])
    # Build
    DQN = Train_model(env, 1000000)
    A2C_model = DQN.A2C_model()
    PPO_model = DQN.PPO_model()

    # Make a prediction and evaluate models
    pred_a2c_model = evaluate_env(df=df_test, model=A2C_model)
    pred_ppo_model = evaluate_env(df=df_test, model=PPO_model)


