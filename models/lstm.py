import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) model for sequential data processing.

    This model consists of an LSTM network with multiple layers, followed 
    by a fully connected output layer for regression tasks (RUL-prediction).
    """
    def __init__(self, sequence_length=30, nb_features=17,):
        """
        Initializes the LSTMModel.

        Args:
            sequence_length (int, optional): The length of the input sequence. Defaults to 30.
            nb_features (int, optional): The number of features per time step. Defaults to 17.
        """
        super(LSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.nb_features = nb_features
        self.lstm = nn.LSTM(input_size=nb_features, hidden_size=50, num_layers=3, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)  # (hidden_size -> output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the LSTMModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, nb_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), representing the regression prediction.
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        x = self.dropout(lstm_out)
        x = self.fc(x)
        x = self.relu(x)

        return x