import torch.nn as nn
import torchvision.models as models
import pandas as pd
from pathlib import Path
import pandas as pd
#from PIL import Image
import torch
import torchvision
#from torch.utils.data import Dataset
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader , WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights


def build_model(jf):
    model_name=jf["model_name"]
    weights_name=jf["weights_name"]
    num_classes=jf["num_classes"]

    if model_name=="resnet18":
        weights=getattr(models,"ResNet18_Weights").DEFAULT if weights_name=="IMAGENET1K_V1" else None
        model=models.resnet18(weights=weights)
        model.fc=nn.Linear(model.fc.in_features,num_classes)
    elif model_name=="resnet34":
        weights = getattr(models, "ResNet34_Weights").DEFAULT
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name=="mobilenet_v3_small":
        weights=getattr(models,"MobileNet_V3_Small_Weights").DEFAULT if weights_name=="IMAGENET1K_V1" else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3]= nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model:{model_name}")
    return model

def compute_class_weights_and_loss(train_csv):
    counts = pd.read_csv(train_csv)["label"].value_counts().sort_index()
    n0, n1 = int(counts[0]), int(counts[1])
    n = n0 + n1
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    class_weights = class_weights / class_weights.max()
    #print("weights:",class_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights=class_weights.to(device)
    print("weights:",class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return class_weights, criterion

def build_loss(jf):
    if jf["train_weights"]:
        _, criterion=compute_class_weights_and_loss(jf["root"]/jf["train.csv"])
    else:
        criterion=nn.CrossEntropyLoss()
    return criterion









