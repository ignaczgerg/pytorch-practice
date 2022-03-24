import torch
import torchvision
from torch import nn
from torch.nn import functional as F 
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_lenght = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 63
num_epoch = 2 

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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checkin accuracy on training data')
    else:
        print("Checking accuracy on test data")
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)

        print(f"{float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)