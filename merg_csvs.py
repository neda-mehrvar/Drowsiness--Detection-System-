import pandas as pd
from pathlib import Path


root=Path(r"C:\Users\schulung\Desktop\P_MFT\Data")
csv_files=list(root.glob("*_new.csv"))
print("found CSVs:",len(csv_files))

dfs=[pd.read_csv(f) for f in csv_files]
merged_df=pd.concat(dfs,ignore_index=True)

out_csv=root/"merged.csv"
merged_df.to_csv(out_csv,index=False)

print("----ok")
print(merged_df.head())

