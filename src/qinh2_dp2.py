import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
transaction_rds = pyreadr.read_r('/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds')
df = transaction_rds[None]
df_pandas = pd.DataFrame(df) # set to pandas data frame
transaction_rds_data_label = df_pandas.columns.tolist()
# print(transaction_rds_data_label) # use to check the data label
df_pandas.head() # check the basic data
# create a 'DateTime' column from the 'timestamp' column 
df_pandas['DateTime'] = df_pandas['timestamp'].transform(lambda x: datetime.datetime.fromtimestamp(x))
df_pandas.head() # check the updated data
# start to draw the graph set x_axis is DateTime and y_axis is amount
df_pandas['DateTime'] = pd.to_datetime(df_pandas['DateTime'])

# convert DataFrame columns to numpy arrays
x = df_pandas["DateTime"].values
y = df_pandas["amount"].values

# plot the graph
plt.plot(x, y)
plt.title("Analysis Graph")
plt.xlabel("DateTime")
plt.ylabel("Amount")
plt.show()