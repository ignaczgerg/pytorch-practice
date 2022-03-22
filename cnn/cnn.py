import torch
from torch import nn, optim
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 
from collections import OrderedDict

# I will use the nn.Sequential, so I don't have to implement the forward pass
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 1 

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 16, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(num_classes=10)
x = torch.randn(64, 1, 28, 28)
# print(model)
# print(model(x).shape)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load train data
train_dataset = datasets.MNIST(root='/Users/ignaczg/Dropbox/Personal/Deep Learning/Pytorch/pytorch-practice/fully-connected/dataset/', 
                                train=True, 
                                transform=transforms.ToTensor(), 
                                download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Load test data
test_dataset = datasets.MNIST(root='/Users/ignaczg/Dropbox/Personal/Deep Learning/Pytorch/pytorch-practice/fully-connected/dataset/', 
                                train=False, 
                                transform=transforms.ToTensor(), 
                                download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # reshape
        # data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() #important!
        loss.backward()

        # gradient descent
        optimizer.step()


# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        print(f'accuracy: {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
