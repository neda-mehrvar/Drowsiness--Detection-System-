from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

from config import Config


import logging
import datetime

now=datetime.datetime.now().strftime("%d.%m.%y__%H-%M-%S")
log_filename=f"log_file_{now}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger=logging.getLogger()
logger.info("********************************")

from models_loss import compute_class_weights_and_loss, build_model,build_loss


jf = Config("config.json")
jf.show()

class EyesDataset(Dataset):
    def __init__(self,csv_file,root,img_size=224,augment=False):

        self.root=Path(root)
        df=pd.read_csv(csv_file)
        self.paths=df["file_path"].astype(str).tolist()
        self.labels=df["label"].astype(int).tolist()
        base_tf=[
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
        augment_tf=[
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                transforms.RandomResizedCrop(size=img_size,scale=(0.8,1.0)),
                transforms.GaussianBlur(kernel_size=5,sigma=(0.1,2.0))
        ] if augment else []
        self.tf=transforms.Compose(augment_tf + base_tf)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p=Path(self.paths[idx])
        if not p.is_absolute():
            p=self.root/p
        img=Image.open(p).convert("RGB")
        img=self.tf(img)
        label=torch.tensor(self.labels[idx],dtype=torch.long)
        return img,label


train_dataset=EyesDataset(jf["root"]/"train.csv",root=jf["root"],img_size=jf["img_size"], augment=True)
val_dataset=EyesDataset(jf["root"]/"val.csv",root=jf["root"],img_size=jf["img_size"], augment=False)
test_dataset=EyesDataset(jf["root"]/"test.csv",root=jf["root"],img_size=jf["img_size"], augment=False)

class_weights, criterion=compute_class_weights_and_loss(jf["root"]/jf["train.csv"])
print("---weights  and Loss are  ok----")

use_cuda=torch.cuda.is_available()
train_loader=DataLoader(train_dataset,batch_size=jf["batch_size"],shuffle=True,num_workers=jf["num_workers"],pin_memory=use_cuda,drop_last=True)
val_loader=DataLoader(val_dataset,batch_size=jf["batch_size"],shuffle=False,num_workers=jf["num_workers"],pin_memory=use_cuda)
test_loader=DataLoader(test_dataset,batch_size=jf["batch_size"],shuffle=False,num_workers=jf["num_workers"],pin_memory=use_cuda)

print("---Dataloader is ok---")

def setup_logging(jf):
    now = datetime.datetime.now().strftime("%d.%m.%y__%H-%M-%S")
    log_dir = Path(jf["root"])/ jf["log_dir"]
    log_dir.mkdir(parents=True,exist_ok=True)
    log_filename = log_dir / f"log_file_{now}.log"
    logging.basicConfig(
        filename=str(log_filename),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    return logger


def train_one_epoch(model, train_loader, criterion,optimizer,device):
    model.train()
    total_loss,total_correct,total=0.0,0,0
    for imgs,labels in train_loader:
        imgs,labels=imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits=model(imgs)
        loss=criterion(logits,labels)
        loss.backward()
        optimizer.step()

        logger.info(f"loss={loss.item()}\n")
        total_loss+=loss.item()*labels.size(0)
        total_correct+=(logits.argmax(1)==labels).sum().item()
        total+=labels.size(0)
    avg_loss=total_loss/total
    accuracy=total_correct/total
    logger.info(f"avg_loss={avg_loss:.4f},acc={accuracy:.4f},")
    return avg_loss, accuracy

@torch.no_grad()
def evaluate_one_epoch(model,val_loader,criterion,device):

    model.eval()
    total_loss, total_correct, total=0.0,0,0
    for imgs, labels in val_loader:
        imgs, labels=imgs.to(device),labels.to(device)

        logits=model(imgs)
        loss=criterion(logits,labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = total_correct / total
    return avg_loss,accuracy

def train_epochs(model, train_loader, val_loader, criterion,optimizer,device,
               num_epochs=jf["num_epochs"],save_path="best.pt"):
    best_val_acc=-1.0
    best_epoch=-1

    for epoch in range(1,num_epochs+1):
        train_loss ,train_acc=train_one_epoch(model,train_loader,criterion, optimizer, device)
        val_loss,val_acc=evaluate_one_epoch(model,val_loader,criterion, device)

        logger.info(f"[{epoch}/{num_epochs}]"
                    f"train_loss={train_loss:.4f},train_acc={train_acc:.4f},")
        logger.info(f"[{epoch}/{num_epochs}]"
                    f"val_loss={val_loss:.4f},val_acc={val_acc:.4f}\n"+"-"*100)
        if val_acc > best_val_acc:
            best_val_acc=val_acc
            torch.save({
                    "epoch":epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "val_acc": val_acc
                },save_path)
            logger.info(f"--saved best model-- : (val_acc={val_acc:.4f}) to {save_path}")
    logger.info(f"Training done. Best val_acc={best_val_acc:.4f}")

def main():

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    model=build_model(jf).to(device)
    print("model is made")
    criterion=build_loss(jf)
    optimizer=optim.Adam(model.parameters(),lr=jf["learning_rate"])

    train_epochs(model,train_loader,val_loader,criterion, optimizer, device,num_epochs=jf["num_epochs"],save_path="best.pt")



if __name__=="__main__":
       print("************Start _ training************")
       main()















