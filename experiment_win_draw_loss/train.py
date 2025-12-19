import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import PremierLeagueDataLoader
from models import MatchPredictor
import json

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target, home_team_ids, away_team_ids in train_loader:
        data, target, home_team_ids, away_team_ids = data.to(device), target.to(device), home_team_ids.to(device), away_team_ids.to(device)
        optimizer.zero_grad()
        output = model(data, home_team_ids, away_team_ids)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target, home_team_ids, away_team_ids in test_loader:
            data, target, home_team_ids, away_team_ids = data.to(device), target.to(device), home_team_ids.to(device), away_team_ids.to(device)
            output = model(data, home_team_ids, away_team_ids)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

if __name__ == '__main__':
    # Hyperparameters
    EMBEDDING_DIM = 10 
    INPUT_DIM_MODEL = 26 
    HIDDEN_DIM = 128
    NUM_CLASSES = 3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000

    # Load data
    data_loader = PremierLeagueDataLoader(data_path='..')
    data_loader.load_data()
    data_loader.preprocess_data()
    train_loader, test_loader = data_loader.get_dataloaders(batch_size=BATCH_SIZE)

    if train_loader and test_loader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        NUM_TEAMS = data_loader.num_teams

        # Calculate Class Weights
        full_train_labels = train_loader.dataset.labels
        class_counts = torch.bincount(full_train_labels)
        class_weights = torch.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            if class_counts[i] > 0:
                class_weights[i] = 1.0 / class_counts[i].float()
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)

        model = MatchPredictor(input_dim=INPUT_DIM_MODEL, num_teams=NUM_TEAMS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Save team_to_id mapping
        with open('saved_model/team_to_id.json', 'w') as f:
            json.dump(data_loader.team_to_id, f, indent=4)
        print("Team to ID mapping saved to saved_model/team_to_id.json")

        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = "saved_model/best_model.pth"

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}: ' 
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | ' 
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f'Validation loss improved. Saving model to {best_model_path}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}.')
                    break
        
        print("\nTraining finished. The best model and team mapping have been saved.")
