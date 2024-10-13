# src/data_prep.py

import pandas as pd

def load_and_print_data():
    data = pd.read_csv('../data/matches.csv', sep=";", header=0, decimal=",")

    data.drop(columns=data.columns[0], axis=1,  inplace=True)

    print(data.head(10).to_string(index=False))

if __name__ == "__main__":
    load_and_print_data()
