import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

class BIRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first = True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        # c0 is required for LSTM only
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        # self.rnn/gru(x,h0)
        out, _ = self.lstm(x,(h0,c0))
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out
    
train_dataset = datasets.FashionMNIST(root='',train=True,
                                      transform = transforms.ToTensor(),
                                      download = True)


test_dataset = datasets.FashionMNIST(root='',train=False,
                                      transform = transforms.ToTensor(),
                                      download = True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle = True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle = True)


model = BIRNN(input_size,hidden_size,num_layers,num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for batch,(data,targets), in enumerate(train_loader):
        data,targets = data.to(device).squeeze(1),targets.to(device)

        # data = data.reshape(data.shape[0],-1)
        pred = model(data)
        loss = loss_fn(pred,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")

    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.inference_mode():
        for x,y in loader:
            x,y = x.to(device).squeeze(1),y.to(device)
            # print(x.shape)
            # x = x.reshape(x.shape[0],-1)
            # print(x.shape)

            pred = model(x)
            _,predictions = pred.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy \
                {float(num_correct)/float(num_samples)*100:.2f}")
            
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
