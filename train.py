import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from model import KeyNet
from dataset import KeyDataset

# File where the best model weights will be stored
model_file_path = Path('checkpoints') / 'keynet.pt'
model_file_path.parent.mkdir(exist_ok=True)

# --- Hyperparameters ---
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 2000
PATIENCE        = 50     # Early Stopping patience (will halve lr or stop after too many epochs w/o improvement)

# 1. Load and split the dataset (edit folders respectively)
dataset_dir = Path('Dataset') / 'giantsteps-mtg-key-dataset'
preprocessed_dir = Path('Dataset') / 'mtg-preprocessed-audio'
dataset = KeyDataset(dataset_dir, preprocessed_dir, pitch_range=(-4,7))

# Use a standard 80/20 train/validation split
train_len = int(len(dataset) * 0.8)
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 2. Initialize model, criterion, optimizer
model = KeyNet(num_classes=24, in_channels=1, Nf=20).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Training/Validation loop with early stopping and learning rate scheduling
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # --- Training phase ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in train_loader:
        inputs = batch['spec'].to(DEVICE)      # Spectrogram chunks
        labels = batch['gt_id'].to(DEVICE)     # Camelot-mapped key indices

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_train_loss = running_loss / total
    train_acc = correct / total

    # --- Validation phase ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['spec'].to(DEVICE)
            labels = batch['gt_id'].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = val_loss / total
    val_acc = correct / total

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
          f"TrainLoss: {avg_train_loss:.4f} | TrainAcc: {train_acc:.4f} | "
          f"ValLoss: {avg_val_loss:.4f} | ValAcc: {val_acc:.4f}"
    )

    # --- Early stopping and learning rate adjustment ---
    # Save best model and reset patience if validation improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), model_file_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            model.load_state_dict(torch.load(model_file_path))
            patience_counter = 0
            LEARNING_RATE /= 2
            print("Learning rate halved.")
            # Update optimizer with new learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
        # Stop if learning rate is too small; this condition prevents endless fine-tuning
        if LEARNING_RATE < 1e-7:
            print("Early stopping: minimal learning rate reached.")
            break
print("Training complete!")
print(f"Best validation loss: {best_val_loss:.6f}")