import torch
from torch import nn


class ATTModel(nn.Module):
    """
    LSTM-based model with an attention mechanism and handcrafted feature processing.

    This model processes sequential input data using an LSTM, applies an attention mechanism
    to focus on important time steps, and combines the output with handcrafted features 
    before making a final prediction.
    """
    def __init__(self):

        super(ATTModel, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=17, hidden_size=50, num_layers=1)
        self.attention = Attention3dBlock()

        self.linear = nn.Sequential(
            nn.Linear(in_features=1500, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(inplace=True)
        )

        self.handcrafted = nn.Sequential(
            nn.Linear(in_features=34, out_features=10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=20, out_features=1)
        )

    def forward(self, inputs, handcrafted_feature):
        y = self.handcrafted(handcrafted_feature)
        x, (hn, cn) = self.lstm(inputs)
        x = self.attention(x)
        x = x.reshape(-1, 1500)
        x = self.linear(x)
        out = torch.concat((x, y), dim=1)
        out = self.output(out)
        return out
        

class Attention3dBlock(nn.Module):
    """
    Attention mechanism for enhancing important time steps in LSTM output.

    This block applies a linear transformation followed by a softmax operation 
    to generate attention weights, which are then used to reweight the LSTM outputs.
    """
    def __init__(self):
        super(Attention3dBlock, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=30, out_features=30),
            nn.Softmax(dim=2),
        )

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.linear(x)
        x_probs = x.permute(0, 2, 1)
        output = x_probs * inputs
        return output