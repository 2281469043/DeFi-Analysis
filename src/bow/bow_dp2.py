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

# Convert timestamp to date
borrow_type["timestamp"] = pd.to_datetime(borrow_type["timestamp"], unit='s')
borrow_type["date"] = borrow_type["timestamp"].dt.date
borrow_type = borrow_type.drop(columns=["timestamp"])
borrow_type.head()

plt.plot(borrow_type["date"].value_counts().sort_index().rolling(window=7).mean())
# Make the X axis wider
plt.xticks(rotation=15)
plt.show()

transactions_type = df["type"]
transactions_type.value_counts().plot(kind='bar')

# Get all borrow transactions
borrow_transactions = df[df["type"] == "borrow"]

# only keep id and timestamp
borrow_transactions = borrow_transactions[["id", "timestamp"]]
borrow_transactions.info()

# convert timestamp to datetime, with unit days
borrow_transactions["timestamp"] = pd.to_datetime(borrow_transactions["timestamp"], unit='s')
borrow_transactions.head()

# convert datetime to date
borrow_transactions["date"] = borrow_transactions["timestamp"].dt.date
borrow_transactions.head()

# drop timestamp
borrow_transactions = borrow_transactions.drop(columns=["timestamp"])
borrow_transactions.head()
