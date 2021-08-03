import pandas as pd
from RL_utils import test_environment
from RL_utils import train_model
from RL_utils import eval_model



def read_file(path):
    # Read the txt file
    df = pd.read_csv(path)

    # we need to convert the Date column to datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df

if __name__ == "__main__":
    path = "/inputs/Data/Stocks/alex.us.txt"
    # Read the txt file
    df = read_file(path)
    print(df.head())
    print("Min Date : {}".format(df.index.min()))
    print("Max Date : {}".format(df.index.max()))

    # Test environment
    test_environment(df)

    # Build environment and Train the model
    model = train_model(df)

    # Evaluation and visualise results
    eval_model(df, model)
