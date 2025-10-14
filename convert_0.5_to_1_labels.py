import pandas as pd
from pathlib import Path


data_dir=Path(r"C:\Users\schulung\Desktop\P_MFT\Data")
df=pd.read_csv(data_dir/"merged.csv")

print("labels_uniqe",df["label"].unique())
print("number of them")
print(df["label"].value_counts())

df["label"]=df["label"].apply(lambda x:1 if x==0.5 else x).astype(int)
df.to_csv(data_dir/"main_data_clean.csv",index=False)

print("labels_uniqe",df["label"].unique())
print("number of them")
print(df["label"].value_counts())


