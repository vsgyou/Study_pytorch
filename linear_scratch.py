#%%

#import zone

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
#%%

# fix seed (참고 : https://blog.naver.com/neutrinoant/223094212615)
def seed_every(num):
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_every(42)

# device 할당
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)
#%%

# Define class
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_size, output_size))    
        self.bias = nn.Parameter(torch.zeros(output_size))
    def forward(self, input):
        return torch.mm(input, self.weight) + self.bias, self.weight, self.bias
    
# Define def
# train
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = []
    for input, true_y in data_loader:
        input = input.to(device)
        true_y = true_y.to(device)
        x_size = input.shape
        y_size = true_y.shape
        pred, weight, bias = model(input)
        loss = criterion(true_y, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    return sum(total_loss) / len(total_loss), weight, bias ,x_size, y_size

# evaluate
def eval(model, data_loader, criterion, device):
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for input, true_y in data_loader:
            input = input.to(device)
            true_y = true_y.to(device)

            pred, weight, bias = model(input)
            loss = criterion(true_y, pred)
            total_loss.append(loss)
    return sum(total_loss) / len(total_loss), weight, bias
#%%

# Data generate
def data_generate(nrow, ncol, split_size):
    x = torch.randn(nrow, ncol)
    w = torch.randint(low = 1, high = 10, size = (ncol,1)).float()
    b = torch.randn(1).float()
    y = torch.mm(x,w) + b

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_size)
    
    return x_train, x_test, y_train, y_test

# Data loader
x_train, x_test, y_train, y_test = data_generate(nrow = 100, ncol = 5 ,split_size = 0.2)
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, 
                          batch_size = 20, 
                          shuffle = True, 
                          drop_last = True)
test_loader = DataLoader(test_dataset,
                         batch_size = 20,
                         shuffle = True,
                         drop_last = True)
# %%

# main_define variables

model = LinearLayer(input_size = 5, output_size = 1).to(device)
learning_rate = 0.01
epochs = 200
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

# main_

with tqdm(range(1, epochs + 1)) as tr:
    for epoch in tr:
        train_loss, train_weight, train_bias, x_size, y_size = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_weight, valid_bias = eval(model, test_loader, criterion, device)
        
        if epoch % 50 == 0:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}, train_weight:{train_weight}, train_bais:{train_bias}')
            print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}, valid_weight:{valid_weight}, valid_bias:{valid_bias}')
# %%
