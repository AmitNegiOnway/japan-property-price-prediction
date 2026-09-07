import pandas as pd
import numpy as np
import os


def load_data(data_url: str) -> pd.DataFrame:
    # Load data
    df = pd.read_csv(data_url)
    return df


def save_data(df: pd.DataFrame, data_path: str) -> None:
    # Save the dataset
    raw_data_path = os.path.join(data_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    df.to_csv(
        os.path.join(raw_data_path, "df.csv"),
        index=False
    )


def main():

    df = load_data(
        data_url="https://raw.githubusercontent.com/AmitNegiOnway/japan-property-price-prediction/main/02.csv"
    )

    save_data(df, "./data")


if __name__ == "__main__":
    main()