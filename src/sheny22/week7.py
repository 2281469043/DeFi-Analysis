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
# print(transaction_rds_data_label) # use to check the data label
df_pandas.head() # check the basic data, know the sample of data


