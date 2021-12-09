import torch
import onnx
import cv2
import torchvision
import torchvision.transforms as transforms

onnx_model = onnx.load("./densenet201.onnx")
onnx.checker.check_model(onnx_model)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([224, 224]),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
batch_size = 4

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = dataiter.next()
densenet = torchvision.models.densenet201(pretrained=False, progress=True, num_classes=10)
densenet.load_state_dict(torch.load("./cifar_densenet201.pth", map_location=torch.device("cuda:0")))
output = densenet(images)

import pdb; pdb.set_trace()
print(output)
