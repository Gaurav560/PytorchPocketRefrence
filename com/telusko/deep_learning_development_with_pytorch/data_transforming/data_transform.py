# Import necessary libraries
from torchvision import transforms  # For data transformation and augmentation
from torchvision.datasets import CIFAR10  # CIFAR10 dataset

# Define transformations to be applied to the training data
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.RandomCrop(32, padding=4),  # Randomly crop the image to 32x32 with 4 pixels of padding
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
        transforms.Normalize(  # Normalize the image tensor
            mean=(0.4914, 0.4822, 0.4465),  # Mean values for RGB channels
            std=(0.2023, 0.1994, 0.2010)  # Standard deviation values for RGB channels
        )
    ]
)

# Load the CIFAR10 training dataset with the defined transformations
train_data = CIFAR10(root="./data/", train=True, transform=train_transforms, download=True)

# Print dataset information
print(train_data)  # Output dataset summary

# Print the transformations applied to the dataset
print(train_data.transforms)

# Retrieve the first data point and its corresponding label from the dataset
data, label = train_data[0]

# Print the type of the data tensor
print(type(data))  # Should print: <class 'torch.Tensor'>

# Print the size of the data tensor
print(data.size())  # Should print: torch.Size([3, 32, 32]) for an RGB image

# Print the data tensor values
print(data)  # Displays the tensor containing pixel values of the first image
