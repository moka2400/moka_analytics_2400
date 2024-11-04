import pandas as pd

def load_data() -> pd.DataFrame:
    data = pd.read_csv('../data/co2_emissions.csv', sep=",", header=0, decimal=".", index_col=0)
    return data