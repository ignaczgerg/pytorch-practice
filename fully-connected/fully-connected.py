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

model = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(input_size, 250)),
          ('fc2', nn.Linear(250, 100)),
          ('fc3', nn.Linear(100, 25)),
          ('fc4', nn.Linear(25, num_classes))
        ]))


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
        data = data.reshape(data.shape[0], -1)

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
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        print(f'accuracy: {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
