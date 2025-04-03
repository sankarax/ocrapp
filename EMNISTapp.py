import torch
from torch import nn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import cv2
from preprocessing import prepro
from tqdm.auto import tqdm
from timeit import default_timer as timer 
from modelv0 import EMNISTV0

import matplotlib.pyplot as plt

BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=[1, 2])),  
    transforms.Lambda(lambda x: torch.flip(x, dims=[2]))  
])

train_data =  datasets.EMNIST(
    root="data",
    split="byclass",
    train=True,
    download=True,
    transform=transform,
    target_transform=None
)   
test_data = datasets.EMNIST(
    root="data",
    split="byclass",
    train=False,
    download=True,
    transform=transform,
    target_transform=None
)

train_dl = DataLoader(train_data,
                      batch_size=BATCH_SIZE,
                      shuffle=True)

test_dl = DataLoader(test_data,
                      batch_size=BATCH_SIZE,
                      shuffle=False)

train_featuresbatch, train_labelsbatch = next(iter(train_dl))
train_featuresbatch.shape, train_labelsbatch.shape
class_names = train_data.classes

def accuracy_fn(true,pred):
    correct = torch.eq(true,pred).sum().item()
    acc = (correct/len(pred)) * 100
    return acc

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device:torch.device=device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X,y = X.to(device),y.to(device)
            y_pred = model(X)
            
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(true=y, 
                                pred=y_pred.argmax(dim=1)) 
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_acc": acc}

def train_model(model:torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               lossfn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):    
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)

        loss = lossfn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(true=y,pred = y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"\nTrain loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%\n")

def test_model(model:torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               lossfn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    test_loss,test_acc = 0,0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            
            test_loss += lossfn(test_pred, y)
            test_acc += accuracy_fn(true=y, 
                                pred=test_pred.argmax(dim=1)) 
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        
    print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

torch.manual_seed(42)
model = EMNISTV0(input_shape=1,
                        hidden_units=32,
                        output_shape=len(class_names)).to(device)

from pathlib import Path
ModelPath = Path("models")
ModelPath.mkdir(parents= True,exist_ok= True)

ModelName = "EMNISTV1.pth"
ModelSavePath = ModelPath / ModelName   
model.load_state_dict(torch.load(ModelSavePath))

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(),
                        lr = 0.1)

def make_predictions(filepath:str):
    pred_probs = []
    preprocessed_img = prepro(filepath)
    image = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
    image = 255-image
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    for x in range(0,27):
        for y in range(0,27):
            if resized[x][y] < 105:
                resized[x][y] = 0
    resized = resized /255.0
    transform2 = transforms.Compose([
        transforms.ToTensor(),       
    ])
    img_tensor = transform2(resized).to(torch.float32)
    img_tensor = img_tensor.to("cuda")
    model.to(device)
    model.eval()
    with torch.inference_mode():
        sample = torch.unsqueeze(img_tensor,dim=0).to(device)
        pred_logits = model(sample)
        pred_prob = torch.softmax(pred_logits.squeeze(),dim=0)
        pred_probs.append(pred_prob.cuda())
        pred_probs = torch.stack(pred_probs)
        pred_class = pred_probs.argmax(dim=1)
    return class_names[pred_class]
