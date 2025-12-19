import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    def __init__(self, input_dim, num_teams, embedding_dim=10, hidden_dim=128, num_classes=3):
        super(MatchPredictor, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        # Adjust input_dim to include the concatenated team embeddings
        # input_dim + (2 * embedding_dim for home and away teams)
        self.fc1 = nn.Linear(input_dim + (2 * embedding_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, home_team_ids, away_team_ids):
        home_embeds = self.team_embedding(home_team_ids)
        away_embeds = self.team_embedding(away_team_ids)
        
        # Concatenate features and embeddings
        x = torch.cat((x, home_embeds, away_embeds), dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
