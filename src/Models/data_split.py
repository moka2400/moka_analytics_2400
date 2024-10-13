# Models/data_split.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    Y_train: pd.DataFrame
    Y_val: pd.DataFrame
    Y_test: pd.DataFrame