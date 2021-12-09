"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
densenet = torchvision.models.densenet121(pretrained=False, progress=True, num_classes=10)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)

def get_loader(batch_size=32, num_workers=4) -> Tuple[Dataset, DataLoader, Dataset, DataLoader]:
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (trainset, trainloader, testset, testloader)


trainset, trainloader, testset, testloader = get_loader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

densenet.to(device)

for epoch in range(100):
    running_loss = 0.0
    pbar = tqdm(trainloader)
    densenet.train()
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000)
            )
            running_loss = 0.0

PATH = './cifar_densenet121.pth'
torch.save(densenet.state_dict(), PATH)
