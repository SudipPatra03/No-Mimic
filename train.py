"""
Training Script
===============
Trains the CNN-BiLSTM Voice Authenticity Detection model using PyTorch.
Includes validation, learning rate scheduling, early stopping, and saving
the best model checkpoint.
"""

import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path to support config/src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config
from src.dataset import create_data_loaders
from src.model import build_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Create data loaders
    try:
        train_loader, val_loader, _ = create_data_loaders()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have run the dataset preparation script first.")
        sys.exit(1)

    # 2. Build model, criterion, optimizer, and scheduler
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.NUM_EPOCHS, 
        eta_min=1e-6
    )

    # 3. Setup tracking variables
    best_val_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print(f"\nEarly stopping counter {config.EARLY_STOPPING_PATIENCE}\n")

    for epoch in range(config.NUM_EPOCHS):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # 4. Record metrics
        tl = train_loss / train_total
        ta = train_correct / train_total
        vl = val_loss / val_total
        va = val_correct / val_total

        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
              f"Train L:{tl:.4f} A:{ta:.4f} | Val L:{vl:.4f} A:{va:.4f} | "
              f"LR:{lr:.2e} | {elapsed:.1f}s")

        # 5. Checkpoint saving & Early stopping
        if vl < best_val_loss:
            best_val_loss = vl
            best_acc=va
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': va,
                'history': history,
            }, config.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (Accuracy: {va:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete! Best Accuracy: {best_acc*100}%")

    # Save training history to json
    history_path = os.path.join(config.MODELS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    train()
