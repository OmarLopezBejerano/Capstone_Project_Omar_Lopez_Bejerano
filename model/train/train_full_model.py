#!/usr/bin/env python3
"""
Full MIMIC-BP Training Script
==============================
Train ResNet152 BP prediction model on complete MIMIC-BP dataset (1524 patients).

This script can run in the background for several hours.
Progress is saved to logs and checkpoints.

Usage:
    python train_full_model.py

Output:
    - best_model_full.pth (best model checkpoint)
    - resnet152_bp_model_full.pth (final model)
    - training_log.txt (detailed training log)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import json
from datetime import datetime
import ast

print("="*80)
print("FULL MIMIC-BP TRAINING - ResNet152 BP Prediction".center(80))
print("="*80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
print("="*80 + "\n")

# Configuration
DATA_PATH = 'MIMIC-BP/'
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_PATIENCE = 15
WINDOW_SIZE = 1250  # 10 seconds at 125 Hz

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else
                     'cuda' if torch.cuda.is_available() else 'cpu')

# Load patient splits
print("Loading patient splits...")
with open(os.path.join(DATA_PATH, 'train_subjects.txt'), 'r') as f:
    train_subjects = ast.literal_eval(f.read())

with open(os.path.join(DATA_PATH, 'val_subjects.txt'), 'r') as f:
    val_subjects = ast.literal_eval(f.read())

with open(os.path.join(DATA_PATH, 'test_subjects.txt'), 'r') as f:
    test_subjects = ast.literal_eval(f.read())

print(f"Train: {len(train_subjects)} patients")
print(f"Val:   {len(val_subjects)} patients")
print(f"Test:  {len(test_subjects)} patients\n")


# Dataset class
class BPDataset(Dataset):
    def __init__(self, patient_list, data_path, window_size=1250, augment=False):
        self.patient_list = patient_list
        self.data_path = data_path
        self.window_size = window_size
        self.augment = augment
        self.samples = []

        print(f"Loading data for {len(patient_list)} patients...")
        for patient_id in tqdm(patient_list, desc="Loading"):
            ecg = np.load(os.path.join(data_path, 'ecg', f'{patient_id}_ecg.npy'))
            ppg = np.load(os.path.join(data_path, 'ppg', f'{patient_id}_ppg.npy'))
            resp = np.load(os.path.join(data_path, 'resp', f'{patient_id}_resp.npy'))
            labels = np.load(os.path.join(data_path, 'labels', f'{patient_id}_labels.npy'))

            for seg_idx in range(ecg.shape[0]):
                segment_length = ecg.shape[1]

                if self.augment:
                    stride = self.window_size // 2
                    num_windows = (segment_length - self.window_size) // stride + 1

                    for w in range(num_windows):
                        start_idx = w * stride
                        end_idx = start_idx + self.window_size

                        if end_idx <= segment_length:
                            self.samples.append({
                                'ecg': ecg[seg_idx, start_idx:end_idx],
                                'ppg': ppg[seg_idx, start_idx:end_idx],
                                'resp': resp[seg_idx, start_idx:end_idx],
                                'sbp': labels[seg_idx, 0],
                                'dbp': labels[seg_idx, 1]
                            })
                else:
                    start_idx = (segment_length - self.window_size) // 2
                    end_idx = start_idx + self.window_size

                    self.samples.append({
                        'ecg': ecg[seg_idx, start_idx:end_idx],
                        'ppg': ppg[seg_idx, start_idx:end_idx],
                        'resp': resp[seg_idx, start_idx:end_idx],
                        'sbp': labels[seg_idx, 0],
                        'dbp': labels[seg_idx, 1]
                    })

        print(f"✓ Created {len(self.samples)} samples\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        signals = np.stack([
            sample['ecg'],
            sample['ppg'],
            sample['resp']
        ], axis=0).astype(np.float32)

        signals = (signals - signals.mean(axis=1, keepdims=True)) / (signals.std(axis=1, keepdims=True) + 1e-8)
        labels = np.array([sample['sbp'], sample['dbp']], dtype=np.float32)

        return torch.from_numpy(signals), torch.from_numpy(labels)


# Model architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet152_BP(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet152_BP, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(256, 36, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Create datasets
print("\n" + "="*80)
print("CREATING DATASETS")
print("="*80 + "\n")

train_dataset = BPDataset(train_subjects, DATA_PATH, augment=True)
val_dataset = BPDataset(val_subjects, DATA_PATH, augment=False)
test_dataset = BPDataset(test_subjects, DATA_PATH, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples:   {len(val_dataset):,}")
print(f"Test samples:  {len(test_dataset):,}\n")


# Initialize model
print("="*80)
print("INITIALIZING MODEL")
print("="*80 + "\n")

model = ResNet152_BP(num_classes=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"- Model parameters: {num_params:,}")
print(f"- Optimizer: Adam (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
print(f"- Scheduler: ReduceLROnPlateau")
print(f"- Loss: MSE\n")
# Training functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for signals, labels in tqdm(train_loader, desc="Training", leave=False):
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * signals.size(0)
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    mae_sbp = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))
    mae_dbp = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))

    return epoch_loss, mae_sbp, mae_dbp


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for signals, labels in tqdm(val_loader, desc="Validating", leave=False):
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * signals.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    mae_sbp = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))
    mae_dbp = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))

    return epoch_loss, mae_sbp, mae_dbp


# Training loop
print("="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [], 'val_loss': [],
    'train_mae_sbp': [], 'train_mae_dbp': [],
    'val_mae_sbp': [], 'val_mae_dbp': [],
    'lr': []
}

start_time = time.time()

# Open log file
log_file = open('training_log.txt', 'w')
log_file.write(f"Full MIMIC-BP Training Started: {datetime.now()}\n")
log_file.write("="*80 + "\n\n")

try:
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_mae_sbp, train_mae_dbp = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mae_sbp, val_mae_dbp = validate_epoch(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae_sbp'].append(train_mae_sbp)
        history['train_mae_dbp'].append(train_mae_dbp)
        history['val_mae_sbp'].append(val_mae_sbp)
        history['val_mae_dbp'].append(val_mae_dbp)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        log_msg = f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)\n"
        log_msg += f"- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
        log_msg += f"- Train MAE: SBP={train_mae_sbp:.2f}, DBP={train_mae_dbp:.2f}\n"
        log_msg += f"- Val MAE:   SBP={val_mae_sbp:.2f}, DBP={val_mae_dbp:.2f}\n"
        log_msg += f"- LR: {current_lr:.6f}\n"

        print(log_msg)
        log_file.write(log_msg)
        log_file.flush()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, 'best_model_full.pth')
            print(f"Best model saved | (Val Loss: {val_loss:.4f})\n")
            log_file.write(f"Best model saved.\n\n")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{MAX_PATIENCE}\n")
            log_file.write(f"No improvement. Patience: {patience_counter}/{MAX_PATIENCE}\n\n")

        # Early stopping
        if patience_counter >= MAX_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs\n")
            log_file.write(f"Early stopping triggered after {epoch+1} epochs\n\n")
            break

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user\n")
    log_file.write("\n\nTraining interrupted by user\n\n")

finally:
    training_time = time.time() - start_time

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'training_time': training_time
    }, 'resnet152_bp_model_full.pth')

    # Final summary
    summary = f"\n{'='*80}\n"
    summary += "TRAINING COMPLETE\n"
    summary += f"{'='*80}\n"
    summary += f"Total Time: {training_time/3600:.2f} hours\n"
    summary += f"Best Val Loss: {best_val_loss:.4f}\n"
    summary += f"Final Epoch: {epoch+1}\n"
    summary += f"Models saved:\n"
    summary += f"- best_model_full.pth (best checkpoint)\n"
    summary += f"- resnet152_bp_model_full.pth (final model)\n"
    summary += f"{'='*80}\n"

    print(summary)
    log_file.write(summary)
    log_file.close()

print("\nTraining complete. Check training_log.txt for details.\n")
