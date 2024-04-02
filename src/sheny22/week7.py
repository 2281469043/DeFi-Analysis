import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
transaction_rds = pyreadr.read_r('/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds')
df = transaction_rds[None]
df.plot() # plot the data
df_pandas = pd.DataFrame(df) # set to pandas data frame
transaction_rds_data_label = df_pandas.columns.tolist()
df_pandas.head() # check the basic data, know the sample of data

data_type = df_pandas["type"]
data_type.value_counts().plot(kind='bar')

borrow_type = df_pandas[df_pandas["type"] == "borrow"]
borrow_type = borrow_type[["id", "timestamp"]]
borrow_type.info()

borrow_type["timestamp"] = pd.to_datetime(borrow_type["timestamp"], unit='s')
borrow_type.head()

borrow_type["date"] = borrow_type["timestamp"].dt.date
borrow_type.head()

borrow_type = borrow_type.drop(columns=["timestamp"])
borrow_type.head()
