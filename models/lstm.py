import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, sequence_length=30, nb_features=17,):
        super(LSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.nb_features = nb_features
        self.lstm = nn.LSTM(input_size=nb_features, hidden_size=50, num_layers=3, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)  # (hidden_size -> output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        x = self.dropout(lstm_out)
        x = self.fc(x)
        x = self.relu(x)

        return x

    def summary(self):
        print(self)