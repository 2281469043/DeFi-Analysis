import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime

result = pyreadr.read_r('../../data/transactions.rds')

df = result[None]
transaction_label = df.columns.tolist()

data_type = df["type"]
borrow_type = df[df["type"] == "borrow"][["id", "timestamp"]]
borrow_type.info()

