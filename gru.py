##installing dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ##Create fully connected network
# class NN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x=F.relu(self.fc1(x))
#         x=self.fc2(x)
#         return x
    
# model = NN(784, 10)
# x=torch.randn(64, 784)
# print(model(x).shape)



##SET DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##HYPERPARAMETERS
input_size = 28 ##the number of expected features in the input ##(Hin)
sequence_length = 28 ##L
num_layers = 2 
hidden_size = 256
learning_rate = 0.001
num_classes = 10
num_epochs = 2
batch_size = 64 ##N

##Hin: The input size. This is the number of expected features in the input (i.e., the dimensionality of the input features).
# Hout: The hidden size. This is the number of features in the hidden state (i.e., the dimensionality of the output from the RNN layer).

##CREATE A RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) ## num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1, batch_first -  If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes) ##used for flattening

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward prop
        out, _ = self.gru(x, h0)
        return self.fc(self.flatten(out))


##LOAD DATA
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

##INITIALISE NETWORK
model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)

##LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

##CHEKCING ACCURACY
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Accuracy on Training...')
    else:
        print('Accuracy on Testing...')
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).squeeze(dim=1)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'GOT {correct}/{total} correct with accuracy of {float((correct)/float(total)) * 100:.2f}%')

    model.train()

##TRAINING

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        ##GET DATA TO DEVICE
        images = images.to(device).squeeze(dim=1) ##to prevent the 
        labels = labels.to(device)

        #FORWARD PASS
        scores = model(images)
        loss = criterion(scores, labels)

        #BACKWARD PASS AND OPTIMIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

check_accuracy(loader=train_loader, model=model)

check_accuracy(loader=test_loader, model=model)
    



        