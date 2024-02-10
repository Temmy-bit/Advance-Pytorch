import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
train_dataset = datasets.FashionMNIST(root='',train=True,
                                      transform = transforms.ToTensor(),
                                      download = True)


test_dataset = datasets.FashionMNIST(root='',train=False,
                                      transform = transforms.ToTensor(),
                                      download = True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle = True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle = True)

class_names = train_dataset.classes
model = NN(input_size=input_size,num_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for batch,(data,targets), in enumerate(train_loader):
        data,targets = data.to(device),targets.to(device)

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
            x,y = x.to(device),y.to(device)
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

random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_dataset), k=9):
    test_samples.append(sample)
    test_labels.append(label)
    
test_samples[0].shape

pred_probs=[]
for i in test_samples:
    
    pred_probs.append(model(i))

# pred_probs[:2]
print(pred_probs[:2])

pred_classes = []

for i in pred_probs:
   pred = i.argmax(dim=1)
   pred_classes.append(pred)
# pred_classes 
# pred_classes = pred_probs.argmax(dim=1)
print(pred_classes[:5])


plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    plt.subplot(nrows,ncols,i+1)
    
    plt.imshow(sample.squeeze(),cmap="gray")
    
    pred_label = class_names[pred_classes[i].detach()]
    print(pred_label)
    
    truth_label = class_names[test_labels[i]]
    
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    
    if pred_label == truth_label:
        plt.title(title_text,fontsize = 10,c="g")
        
    else:
        plt.title(title_text,fontsize=10,c="r")
        
    plt.axis(False);

plt.show();