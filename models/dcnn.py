import torch.nn as nn
import torch.nn.functional as F

class DeepCNN(nn.Module):
    """
    A deep convolutional neural network (DCNN) for processing sequential data and make RUL-estimation.

    This model consists of multiple convolutional layers followed by fully 
    connected layers. It is designed for regression tasks like RUL-estimation.
    """
    def __init__(self):
        super(DeepCNN, self).__init__()

        # Convolutional Layers with Limited Pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,1), padding='same')      
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5,1), padding='same')
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5,1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5,1), padding='same')
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(3,1), padding='same')

        # Fully Connected Layers
        self.fc1 = nn.Linear(510, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Defines the forward pass of the DeepCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), representing the RUL prediction.
        """
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x