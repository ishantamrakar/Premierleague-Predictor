import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super(MatchPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # We'll use CrossEntropyLoss which combines LogSoftmax and NLLLoss
        return x
