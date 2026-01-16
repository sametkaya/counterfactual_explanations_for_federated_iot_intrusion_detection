import pandas as pd
df = pd.read_csv('dataset/toniot/Network_dataset_1.csv', nrows=5)
print(df.dtypes)