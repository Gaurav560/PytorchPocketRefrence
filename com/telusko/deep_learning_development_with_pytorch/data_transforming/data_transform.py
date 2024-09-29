# data might need to be adjusted before it is passed into the NN model for training and testing.
# ex: data values may be normalized to assist training,
# augmented to create large datasets, or converted from one type to another.
from torchvision import transforms
from torchvision.datasets import CIFAR10

train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize(
         mean=(0.4914, 0.4822, 0.4465),
         std=(0.2023, 0.1994, 0.2010)
     )])

train_data=CIFAR10(root="./data/",train=True,transform=train_transforms,download=True)

print(train_data)

print(train_data.transforms)