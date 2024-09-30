import torch  # PyTorch library for building and running neural networks
import torch.nn as nn  # Sub-library for defining neural network layers
import torch.nn.functional as F  # Contains common functions like activation functions (e.g., ReLU, Softmax)


# We are designing a simple neural network called "SimpleNet"
class SimpleNet(
    nn.Module):  # We create a class that represents the neural network model, inheriting from nn.Module (a base class for all neural networks in PyTorch)
    def __init__(self):  # This function is the constructor. It defines the layers of the model.
        super(SimpleNet, self).__init__()  # Call the constructor of the parent class nn.Module to initialize it

        # Define the layers of our network (basically, what transformations we want to do on the data):

        # First layer: A fully connected (linear) layer that takes input of size 2048 and gives output of size 256
        self.fc1 = nn.Linear(2048, 256)

        # Second layer: Another fully connected layer that takes the 256 output from the previous layer and reduces it to 64
        self.fc2 = nn.Linear(256, 64)

        # Third layer: A fully connected layer that takes the 64 output from the previous layer and reduces it to 2 (because we want 2 possible outputs, like 2 classes)
        self.fc3 = nn.Linear(64, 2)

    # This function defines how data will flow through the network (from input to output)
    def forward(self, x):  # 'x' is the input to the network
        # Reshape the input into the correct size:
        # -1 means "infer this dimension automatically", 2048 is the input size we defined above.
        x = x.view(-1, 2048)

        # Apply the first layer (fc1), then use the ReLU activation function to introduce non-linearity
        x = F.relu(self.fc1(x))

        # Apply the second layer (fc2), again followed by the ReLU activation function
        x = F.relu(self.fc2(x))

        # Apply the third layer (fc3), but this time we use the Softmax function, which is useful for classification tasks
        # Softmax will turn the two output values into probabilities (which sum up to 1)
        x = F.softmax(self.fc3(x), dim=1)  # dim=1 means we apply softmax across the rows (since it's a 2D tensor)


# Now we create the model by instantiating (creating an object of) the SimpleNet class
simple_net = SimpleNet()

# Let's print the model to see its structure (i.e., layers and parameters)
print(simple_net)

# We now create some random input data to test the model
# torch.randn(2048) generates random numbers in a 1D array (tensor) of size 2048
input_set = torch.randn(2048)

# We run this random input data through the network we just created
output = simple_net(input_set)  # This will apply the forward() function and give us the model's predictions
