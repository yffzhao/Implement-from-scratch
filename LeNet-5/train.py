import torch
from torch import nn
from net import LeNet
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# Data transform
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Load Data
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = LeNet().to(device)

# Loss
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# lr
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train
def train(dataloader, model, loss_function, optimizer):
    loss, acc, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Forward
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_function(output, y)
        _, pred = torch.max(output, 1)
        cur_acc = torch.sum(y == pred)/output.shape[0]

        # Backward
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        # Loss
        loss += cur_loss.item()
        acc += cur_acc
        n= n+1
    print("train_loss " + str(loss/n))
    print("train_acc" + str(acc/n))

def validate(dataloader, model, loss_function):
    model.eval()
    loss, acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Forward
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_function(output, y)
            _, pred = torch.max(output, 1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            # Loss
            loss += cur_loss.item()
            acc += cur_acc
            n = n + 1
        print("valid_loss " + str(loss / n))
        print("valid_acc" + str(acc / n))
    return acc/n

epoch = 50
best_acc = 0
best_epoch = epoch
for t in range(epoch):
    print(f'epoch{t+1}\n---------------')
    train(train_dataloader, model, loss_function, optimizer)
    a = validate(test_dataloader, model, loss_function)
    # Save best weight
    if a > best_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        best_acc = a
        best_epoch = t+1
        torch.save(model.state_dict(), folder+'/best_model.pth')
print(f'best epoch: {best_epoch}, acc: {best_acc}')
print("Done")



