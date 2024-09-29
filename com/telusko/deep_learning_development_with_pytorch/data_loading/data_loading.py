import torch  # Importing the PyTorch library for tensor operations and deep learning
import torchvision  # Importing the torchvision library which contains datasets, models, and transforms
from torchvision.datasets import CIFAR10  # Importing the CIFAR10 dataset class from torchvision

# Printing the version of PyTorch and torchvision to ensure they are installed and to check compatibility
print(torch.__version__)
print(torchvision.__version__)

# Loading the CIFAR-10 training dataset
train_data = CIFAR10(root="./train/", train=True, download=True)
# - root: Directory where the dataset will be stored
# - train: Specifies whether to load the training split (True) or the test split (False)
# - download: Downloads the dataset if it is not already present in the specified root directory

# Printing the dataset object, which includes information about the dataset
print(train_data)

# Printing the total number of training samples in the dataset
print(len(train_data))  # Should return 50000 for the training set

# Printing the shape of the image data. This will typically be (50000, 32, 32, 3) for CIFAR-10
print(train_data.data.shape)

# Printing the list of class names in the dataset
print(train_data.classes)  # e.g., ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Printing the mapping of class names to their corresponding index values
print(train_data.class_to_idx)  # e.g., {'plane': 0, 'car': 1, ..., 'truck': 9}

# Checking the type of the first element in the dataset
print(type(train_data[0]))  # Should output <class 'tuple'> because it contains (image, label)

# Checking the length of the first element (should be 2: image and label)
print(len(train_data[0]))  # Should output 2

# Checking the type of the 50,000th element (the last image)
print(type(train_data[49999]))  # Should output <class 'tuple'>

# Accessing the first image and its label
data, label = train_data[0]
# - data: The image itself
# - label: The corresponding class label for the image

# Printing the type of the image data (e.g., PIL Image or tensor depending on transformations)
print(type(data))

# Printing the image data (may output a lot of information)
print(data)

# Printing the type of the label (should be <class 'int'>)
print(type(label))

# Printing the value of the label, which indicates the class index of the image
print(label)

# Using the label to access the corresponding class name from the dataset
print(train_data.classes[label])  # Prints the human-readable class name corresponding to the label
