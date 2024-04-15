import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

transaction = pyreadr.read_r("../../data/transactions.rds")
df = transaction[None]
df['DateTime'] = df['timestamp'].transform(lambda x: datetime.fromtimestamp(x))
df.head()

dailyTransactionCount = df.groupby([df['DateTime'].dt.date]).count()



