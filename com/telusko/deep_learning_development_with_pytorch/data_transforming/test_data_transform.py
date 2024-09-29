from torchvision import transforms
from torchvision.datasets import CIFAR10

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_data=CIFAR10(root="./test_data",train=False,download=True,transform=test_transform)

print(test_data)