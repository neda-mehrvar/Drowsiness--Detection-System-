import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path




counts=pd.read_csv(root/"train.csv")["label"].value_counts()
n0=counts[0]
n1=counts[1]

n=n0+n1
w0=n/(2*n0)
w1=n/(2*n1)
train_weights=torch.tensor([w0,w1],dtype=torch.float32)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_weights=train_weights.to(device)






