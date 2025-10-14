import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir=Path(r"C:\Users\schulung\Desktop\P_MFT\Data")
df=pd.read_csv(data_dir/"main_data_clean.csv")

train_df, t2_df=train_test_split(df,test_size=0.40, random_state=42, stratify=df["label"])
val_df, test_df=train_test_split(t2_df,test_size=0.50, random_state=42, stratify=t2_df["label"])

train_df.to_csv(data_dir/"train.csv",index=False)
val_df.to_csv(data_dir/"val.csv",index=False)
test_df.to_csv(data_dir/"test.csv",index=False)
print("split did")

print("train_len: ",len(train_df),"Val: ",len(val_df),"test: ",len(test_df))
print("train_labels:\n",train_df["label"].value_counts())
print("val_labels:\n",val_df["label"].value_counts())
print("test_labels:\n",test_df["label"].value_counts())





