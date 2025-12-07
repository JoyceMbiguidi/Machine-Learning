import pandas as pd
import numpy as np

path = "https://raw.githubusercontent.com/JoyceMbiguidi/data/main/decathlon.csv"
raw_df = pd.read_csv(path, sep = ";")

print(raw_df.head())
print(raw_df.dtypes)
print(raw_df.select_dtypes(exclude=np.number).columns.tolist())
