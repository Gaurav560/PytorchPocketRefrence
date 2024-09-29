from torchvision.datasets import CIFAR10

test_data=CIFAR10(root="./test/",train=False,download=True)
print(test_data)