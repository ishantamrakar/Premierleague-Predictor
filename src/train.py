import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import PremierLeagueDataLoader
from models import MatchPredictor

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
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

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
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
    INPUT_DIM = 26 # Number of features from data_loader.py
    HIDDEN_DIM = 128
    NUM_CLASSES = 3 # Home, Draw, Away
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 20

    # Load data
    data_loader = PremierLeagueDataLoader(data_path='.')
    data_loader.load_data()
    data_loader.preprocess_data()
    train_loader, test_loader = data_loader.get_dataloaders(batch_size=BATCH_SIZE)

    if train_loader and test_loader:
        # Initialize model, loss, and optimizer
        model = MatchPredictor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, test_loader, criterion)
            
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
