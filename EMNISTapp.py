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
import os
import matplotlib.pyplot as plt

from collections import defaultdict
import re

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
model.load_state_dict(torch.load(ModelSavePath, map_location=torch.device('cpu')))

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(),
                        lr = 0.1)

def make_predictions(filepath: str):
    final_img = prepro(filepath)
    import re
    from collections import defaultdict
    transform2 = transforms.Compose([
        transforms.ToTensor(),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = 'preprocessed/'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    word_groups = defaultdict(list)
    for filename in image_files:
        match = re.match(r"(text\d+)_(\d+)\.png", filename)
        if match:
            word_id, letter_idx = match.groups()
            word_groups[word_id].append((int(letter_idx), filename))

    full_phrase = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for word_id in sorted(word_groups.keys(), key=lambda x: int(x[4:])):  
            sorted_files = [fname for _, fname in sorted(word_groups[word_id])]
            img_tensors = []

            for filename in sorted_files:
                img_path = os.path.join(image_dir, filename)
                preprocessed_img = cv2.imread(img_path)

                if preprocessed_img is None:
                    print(f"Warning: {filename} couldn't be read.")
                    continue

                image = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
                image = 255 - image
                resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

                for x in range(28):
                    for y in range(28):
                        if resized[x][y] < 105:
                            resized[x][y] = 0

                resized = resized / 255.0
                img_tensor = transform2(resized).to(torch.float32)
                img_tensors.append(img_tensor)

            if not img_tensors:
                continue

            batch_tensor = torch.stack(img_tensors).to(device)
            pred_probs = []

            for sample in batch_tensor:
                sample = torch.unsqueeze(sample, dim=0).to(device)
                pred_logits = model(sample)
                pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
                pred_probs.append(pred_prob)

            pred_probs2 = torch.stack(pred_probs)
            pred_classes = pred_probs2.argmax(dim=1)
            predicted_letters = [class_names[i] for i in pred_classes]
            predicted_word = ''.join(predicted_letters)
            full_phrase.append(predicted_word)

    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)

    final_output = ' '.join(full_phrase)
    print(final_output)
    return final_output

    
