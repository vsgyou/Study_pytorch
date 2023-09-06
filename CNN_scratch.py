 #%%

# import 
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
# %%

# device 할당
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

# %%

# train, valid, eval 정의
def train(model, data_loader, optimizer, citerion, device):
    model.train()
    total_loss = []

    for images, label in data_loader:
        input = images.to(device)

# layer 정의
class Net_layer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = input_size, 
                              out_channels = output_size,
                              kernel_size = kernel_size, 
                              padding = 1)
        self.batch_norm = nn.BatchNorm2d(num_features = output_size)
            # conv의 output 결과가 bachnorm의 input으로 들어감
        self.relu = nn.ReLU(inplace = True)
        self.Pool = torch.nn.MaxPool2d(kernel_size = 2)
    
    def forward(self, x):
        output = self.conv(x)
        output = self.batch_norm(output)
        output = self.relu(output)
        output = self.Pool(output)
        return output

class full_connect_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.connect = nn.Linear(in_features = self.input_size, 
                                 out_features = self.output_size)
    
    def forward(self, input):
        input_data = input.view(input.shape[0], -1)
        output = self.connect(input_data)
        return output
     
net1 = Net_layer(input_size = 1,
                output_size = 32,
                kernel_size = 3)
net2 = Net_layer(input_size = 32,
                 output_size = 64,
                 kernel_size = 3)
net3 = full_connect_layer(input_size = 64 * 7 * 7, 
                          output_size = 10)

model = nn.Sequential(net1, net2, net3).to(device)   
# %%

# data 불러오기
train_set = datasets.MNIST(root = 'MNIST_data/', 
                           train = True, 
                           transform = transforms.ToTensor(), 
                           download = True)

test_set = datasets.MNIST(root = 'MNIST_data/', 
                          train = False,
                          transform = transforms.ToTensor(),
                          download = True)

train_set, valid_set = random_split(train_set, 
                                    [len(train_set) - len(test_set), len(test_set)], torch.Generator().manual_seed(123))

train_loader = DataLoader(train_set, 
                          batch_size = 64, 
                          shuffle = True, 
                          drop_last = True)

test_loader = DataLoader(test_set, 
                         batch_size = 64, 
                         shuffle = True,
                         drop_last = True)

valid_loader = DataLoader(valid_set,
                          batch_size = 64, 
                          shuffle = True,
                          drop_last = True)

# %%

# 인수 정의

# def loss(outputs, labels):
#     criterion = []
#     criterion = nn.CrossEntropyLoss().to(device)
#     loss = criterion(outputs, labels)
#     return loss
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 50
best_valid_loss = float('Inf')
early_stopping_count = 0
early_stopping = 5

#%%

# 모델 실행
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = []
    
    for images, label in data_loader:
        
        input = images.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(input).to(device)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    return sum(total_loss) / len(total_loss)

def eval(model, data_loader, criterion, device):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for images, label in data_loader:
            input = images.to(device)
            label =label.to(device)

            pred = model(input)
            loss = criterion(pred, label)
            total_loss.append(loss)
    return sum(total_loss) / len(total_loss)

def pred(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, label in data_loader:
            input = images.to(device)
            label = label.to(device)
            pred = model(input)
            _, pred = torch.max(pred, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    accuracy = correct / total
    return accuracy
# train(model, train_loader, optimizer, criterion, device)
# %%
from tqdm.notebook import tqdm
with tqdm(range(1, epochs + 1)) as tr :
    for epoch in tr:
        train_loss = train(model, train_loader, optimizer, criterion, device)
        valid_loss = eval(model, valid_loader, criterion, device)
        
        if epoch % 5 == 0:

            
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}. valid_loss:{valid_loss.item():5f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count >= early_stopping:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
            break

#%%
model_result = model
model_result.load_state_dict(torch.load('best_model.pth'))
accuracy = pred(model, test_loader, device)
print(f' Accuracy on test data: {100 * accuracy:.2f}%')

# %%
