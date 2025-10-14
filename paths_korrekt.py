import pandas as pd
from pathlib import Path

#data_dir=Path(r"C:\\Users\schulung\Desktop\P_MFT\Data")
#csv_files=list(data_dir.rglob("*.csv"))
#print("Found CSVs:",len(csv_files))

root=Path(r"C:\Users\schulung\Desktop\P_MFT\Data")
csv_file=Path(root/"s016_blink_labels.csv")
subject=csv_file.stem.split("_")[0]
df=pd.read_csv(csv_file, names=["file_path","label"])
def rel_path(x):
    name=Path(str(x)).name
    return fr"{subject}_noglasses\natural\left\{name}"

df["file_path"]=df["file_path"].apply(rel_path)
out_csv=csv_file.with_name(csv_file.stem+"_new.csv")
df.to_csv(out_csv,index=False)

print("---ok")
print(df.head())

















