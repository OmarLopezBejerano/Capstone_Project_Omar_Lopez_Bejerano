#!/usr/bin/env python3
"""
Test Full MIMIC-BP Model
========================
Comprehensive evaluation of the trained ResNet152 BP prediction model.

Usage:
    python test_full_model.py

Output:
    - Detailed test metrics
    - Visualizations (loss curves, Bland-Altman plots, error distributions)
    - ISO 81060-2:2018 compliance assessment
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ast

print("="*80)
print("FULL MODEL EVALUATION - ResNet152 BP Prediction".center(80))
print("="*80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# Configuration
DATA_PATH = 'MIMIC-BP/'
BATCH_SIZE = 64
WINDOW_SIZE = 1250  # 10 seconds at 125 Hz

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else
                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


### Training and Evaluation Functions

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics per ISO 81060-2:2018 standard

    Args:
        predictions: numpy array of shape (N, 2) - predicted [SBP, DBP]
        targets: numpy array of shape (N, 2) - actual [SBP, DBP]

    Returns:
        Dictionary with metrics for SBP and DBP
    """
    # Calculate errors
    errors = predictions - targets

    # Mean Error (ME) - should be ≤ ±5 mmHg
    me_sbp = np.mean(errors[:, 0])
    me_dbp = np.mean(errors[:, 1])

    # Standard Deviation (STD) - should be ≤ 8 mmHg
    std_sbp = np.std(errors[:, 0])
    std_dbp = np.std(errors[:, 1])

    # Mean Absolute Error (MAE)
    mae_sbp = np.mean(np.abs(errors[:, 0]))
    mae_dbp = np.mean(np.abs(errors[:, 1]))

    # Root Mean Squared Error (RMSE)
    rmse_sbp = np.sqrt(np.mean(errors[:, 0]**2))
    rmse_dbp = np.sqrt(np.mean(errors[:, 1]**2))

    return {
        'SBP': {
            'ME': me_sbp,
            'STD': std_sbp,
            'MAE': mae_sbp,
            'RMSE': rmse_sbp
        },
        'DBP': {
            'ME': me_dbp,
            'STD': std_dbp,
            'MAE': mae_dbp,
            'RMSE': rmse_dbp
        }
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for signals, labels in tqdm(train_loader, desc="Training"):
        signals = signals.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * signals.size(0)
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)

    return epoch_loss, metrics


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for signals, labels in tqdm(val_loader, desc="Validation"):
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * signals.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)

    return epoch_loss, metrics


def print_test_metrics(metrics):
    """Print formatted test metrics"""
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    print(f"{'Metric':<20} {'SBP':>15} {'DBP':>15}")
    print("-"*80)
    print(f"{'MAE (mmHg)':<20} {metrics['SBP']['MAE']:>15.3f} {metrics['DBP']['MAE']:>15.3f}")
    print(f"{'RMSE (mmHg)':<20} {metrics['SBP']['RMSE']:>15.3f} {metrics['DBP']['RMSE']:>15.3f}")
    print(f"{'ME (mmHg)':<20} {metrics['SBP']['ME']:>15.3f} {metrics['DBP']['ME']:>15.3f}")
    print(f"{'STD (mmHg)':<20} {metrics['SBP']['STD']:>15.3f} {metrics['DBP']['STD']:>15.3f}")
    print("-"*80)

    # ISO 81060-2:2018 compliance check
    iso_sbp = abs(metrics['SBP']['ME']) <= 5 and metrics['SBP']['STD'] <= 8
    iso_dbp = abs(metrics['DBP']['ME']) <= 5 and metrics['DBP']['STD'] <= 8

    print(f"\nISO 81060-2:2018 Standard Compliance:")
    print(f"   SBP: {'PASS' if iso_sbp else 'FAIL'} (ME ≤ 5 mmHg, STD ≤ 8 mmHg)")
    print(f"   DBP: {'PASS' if iso_dbp else 'FAIL'} (ME ≤ 5 mmHg, STD ≤ 8 mmHg)")
    print("="*80)


### Dataset Class

class BPDataset(Dataset):
    def __init__(self, patient_list, data_path, window_size=1250):
        self.patient_list = patient_list
        self.data_path = data_path
        self.window_size = window_size
        self.samples = []

        print(f"Loading data for {len(patient_list)} patients...")
        for patient_id in tqdm(patient_list, desc="Loading"):
            ecg = np.load(os.path.join(data_path, 'ecg', f'{patient_id}_ecg.npy'))
            ppg = np.load(os.path.join(data_path, 'ppg', f'{patient_id}_ppg.npy'))
            resp = np.load(os.path.join(data_path, 'resp', f'{patient_id}_resp.npy'))
            labels = np.load(os.path.join(data_path, 'labels', f'{patient_id}_labels.npy'))

            for seg_idx in range(ecg.shape[0]):
                segment_length = ecg.shape[1]
                start_idx = (segment_length - self.window_size) // 2
                end_idx = start_idx + self.window_size

                self.samples.append({
                    'ecg': ecg[seg_idx, start_idx:end_idx],
                    'ppg': ppg[seg_idx, start_idx:end_idx],
                    'resp': resp[seg_idx, start_idx:end_idx],
                    'sbp': labels[seg_idx, 0],
                    'dbp': labels[seg_idx, 1]
                })

        print(f" Created {len(self.samples)} samples\n")

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


### Model Architecture

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


### Load Test Data

print("Loading test data")
with open(os.path.join(DATA_PATH, 'test_subjects.txt'), 'r') as f:
    test_subjects = ast.literal_eval(f.read())

print(f"Test set: {len(test_subjects)} patients\n")

test_dataset = BPDataset(test_subjects, DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test samples: {len(test_dataset):,}\n")


### Load Trained Model

print("="*80)
print("LOADING TRAINED MODEL")
print("="*80 + "\n")

MODEL_PATH = 'best_model_full.pth'

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found: {MODEL_PATH}")
    print("Please ensure training is complete and best_model_full.pth exists.")
    exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = ResNet152_BP(num_classes=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: {MODEL_PATH}")
if 'epoch' in checkpoint:
    print(f"- Trained for {checkpoint['epoch']+1} epochs")
else:
    print(f"- Trained for N/A epochs")

if 'val_loss' in checkpoint:
    print(f"- Best validation loss: {checkpoint['val_loss']:.4f}")
else:
    print(f"- Best validation loss: N/A")


### Extract Training History

print("\n" + "="*80)
print("TRAINING HISTORY")
print("="*80 + "\n")

if 'history' in checkpoint:
    history = checkpoint['history']

    print(f"Training completed with {len(history['train_loss'])} epochs")
    print(f"\nFinal Training Metrics:")
    print(f"- Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"- Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"- Train MAE: SBP = {history['train_mae_sbp'][-1]:.2f}, DBP = {history['train_mae_dbp'][-1]:.2f}")
    print(f"- Val MAE: SBP = {history['val_mae_sbp'][-1]:.2f}, DBP = {history['val_mae_dbp'][-1]:.2f}")

    # Find best epoch
    best_epoch = np.argmin(history['val_loss'])
    print(f"\nBest Epoch: {best_epoch + 1}")
    print(f"Val Loss: {history['val_loss'][best_epoch]:.4f}")
    print(f"Val MAE: SBP = {history['val_mae_sbp'][best_epoch]:.2f}, DBP = {history['val_mae_dbp'][best_epoch]:.2f}")
else:
    history = None
    print("No training history found in checkpoint")


### Test Model

print("\n" + "="*80)
print("TESTING MODEL ON TEST SET")
print("="*80 + "\n")

criterion = nn.MSELoss()
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for signals, labels in tqdm(test_loader, desc="Testing"):
        signals = signals.to(device)
        labels = labels.to(device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * signals.size(0)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
all_predictions = np.vstack(all_predictions)
all_targets = np.vstack(all_targets)

print(f"\nTesting complete!")
print(f"Test Loss: {test_loss:.4f}")
print(f"Samples evaluated: {len(all_predictions):,}\n")


### Calculate Test Metrics

test_metrics = calculate_metrics(all_predictions, all_targets)
print_test_metrics(test_metrics)


### Visualization 1: Training Curves

if history is not None:
    print("\nGenerating visualizations.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7, label='Best Epoch')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # SBP MAE
    axes[0, 1].plot(epochs, history['train_mae_sbp'], 'b-', label='Train SBP', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae_sbp'], 'r-', label='Val SBP', linewidth=2)
    axes[0, 1].axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7, label='Best Epoch')
    axes[0, 1].axhline(y=15, color='orange', linestyle=':', alpha=0.7, label='Clinical Target')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE (mmHg)', fontsize=12)
    axes[0, 1].set_title('SBP Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # DBP MAE
    axes[1, 0].plot(epochs, history['train_mae_dbp'], 'b-', label='Train DBP', linewidth=2)
    axes[1, 0].plot(epochs, history['val_mae_dbp'], 'r-', label='Val DBP', linewidth=2)
    axes[1, 0].axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7, label='Best Epoch')
    axes[1, 0].axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='Clinical Target')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MAE (mmHg)', fontsize=12)
    axes[1, 0].set_title('DBP Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: training_curves.png")


### Visualization 2: Bland-Altman Plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# SBP Bland-Altman
sbp_mean = (all_predictions[:, 0] + all_targets[:, 0]) / 2
sbp_diff = all_predictions[:, 0] - all_targets[:, 0]
sbp_mean_diff = np.mean(sbp_diff)
sbp_std_diff = np.std(sbp_diff)

axes[0].scatter(sbp_mean, sbp_diff, alpha=0.3, s=10)
axes[0].axhline(y=sbp_mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {sbp_mean_diff:.2f}')
axes[0].axhline(y=sbp_mean_diff + 1.96*sbp_std_diff, color='red', linestyle='--', linewidth=1.5,label=f'+1.96 SD: {sbp_mean_diff + 1.96*sbp_std_diff:.2f}')
axes[0].axhline(y=sbp_mean_diff - 1.96*sbp_std_diff, color='red', linestyle='--', linewidth=1.5,label=f'-1.96 SD: {sbp_mean_diff - 1.96*sbp_std_diff:.2f}')
axes[0].set_xlabel('Mean SBP (mmHg)', fontsize=12)
axes[0].set_ylabel('Difference (Predicted - Actual) mmHg', fontsize=12)
axes[0].set_title('Bland-Altman Plot: SBP', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# DBP Bland-Altman
dbp_mean = (all_predictions[:, 1] + all_targets[:, 1]) / 2
dbp_diff = all_predictions[:, 1] - all_targets[:, 1]
dbp_mean_diff = np.mean(dbp_diff)
dbp_std_diff = np.std(dbp_diff)

axes[1].scatter(dbp_mean, dbp_diff, alpha=0.3, s=10)
axes[1].axhline(y=dbp_mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {dbp_mean_diff:.2f}')
axes[1].axhline(y=dbp_mean_diff + 1.96*dbp_std_diff, color='red', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {dbp_mean_diff + 1.96*dbp_std_diff:.2f}')
axes[1].axhline(y=dbp_mean_diff - 1.96*dbp_std_diff, color='red', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {dbp_mean_diff - 1.96*dbp_std_diff:.2f}')
axes[1].set_xlabel('Mean DBP (mmHg)', fontsize=12)
axes[1].set_ylabel('Difference (Predicted - Actual) mmHg', fontsize=12)
axes[1].set_title('Bland-Altman Plot: DBP', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bland_altman_plots.png', dpi=150, bbox_inches='tight')
print("Saved: bland_altman_plots.png")


### Visualization 3: Error Distributions

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# SBP error histogram
axes[0, 0].hist(sbp_diff, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0, 0].axvline(x=sbp_mean_diff, color='green', linestyle='-', linewidth=2, label=f'Mean: {sbp_mean_diff:.2f}')
axes[0, 0].set_xlabel('Error (mmHg)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('SBP Error Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# DBP error histogram
axes[0, 1].hist(dbp_diff, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0, 1].axvline(x=dbp_mean_diff, color='green', linestyle='-', linewidth=2, label=f'Mean: {dbp_mean_diff:.2f}')
axes[0, 1].set_xlabel('Error (mmHg)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('DBP Error Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# SBP scatter
axes[1, 0].scatter(all_targets[:, 0], all_predictions[:, 0], alpha=0.3, s=10)
axes[1, 0].plot([all_targets[:, 0].min(), all_targets[:, 0].max()],[all_targets[:, 0].min(), all_targets[:, 0].max()],'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual SBP (mmHg)', fontsize=12)
axes[1, 0].set_ylabel('Predicted SBP (mmHg)', fontsize=12)
axes[1, 0].set_title('SBP: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# DBP scatter
axes[1, 1].scatter(all_targets[:, 1], all_predictions[:, 1], alpha=0.3, s=10)
axes[1, 1].plot([all_targets[:, 1].min(), all_targets[:, 1].max()],[all_targets[:, 1].min(), all_targets[:, 1].max()],'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual DBP (mmHg)', fontsize=12)
axes[1, 1].set_ylabel('Predicted DBP (mmHg)', fontsize=12)
axes[1, 1].set_title('DBP: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: error_analysis.png")


### Save Results

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

# Save test predictions
np.save('test_predictions.npy', all_predictions)
np.save('test_targets.npy', all_targets)
print(" Saved: test_predictions.npy, test_targets.npy")

# Save metrics to text file
with open('test_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FULL MODEL TEST RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test samples: {len(all_predictions):,}\n")
    f.write(f"Test loss: {test_loss:.4f}\n\n")

    f.write("TEST METRICS\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Metric':<20} {'SBP':>15} {'DBP':>15}\n")
    f.write("-"*80 + "\n")
    f.write(f"{'MAE (mmHg)':<20} {test_metrics['SBP']['MAE']:>15.3f} {test_metrics['DBP']['MAE']:>15.3f}\n")
    f.write(f"{'RMSE (mmHg)':<20} {test_metrics['SBP']['RMSE']:>15.3f} {test_metrics['DBP']['RMSE']:>15.3f}\n")
    f.write(f"{'ME (mmHg)':<20} {test_metrics['SBP']['ME']:>15.3f} {test_metrics['DBP']['ME']:>15.3f}\n")
    f.write(f"{'STD (mmHg)':<20} {test_metrics['SBP']['STD']:>15.3f} {test_metrics['DBP']['STD']:>15.3f}\n")
    f.write("-"*80 + "\n\n")

    iso_sbp = abs(test_metrics['SBP']['ME']) <= 5 and test_metrics['SBP']['STD'] <= 8
    iso_dbp = abs(test_metrics['DBP']['ME']) <= 5 and test_metrics['DBP']['STD'] <= 8
    f.write("ISO 81060-2:2018 COMPLIANCE\n")
    f.write("-"*80 + "\n")
    f.write(f"SBP: {'PASS' if iso_sbp else 'FAIL'} (ME ≤ 5 mmHg, STD ≤ 8 mmHg)\n")
    f.write(f"DBP: {'PASS' if iso_dbp else 'FAIL'} (ME ≤ 5 mmHg, STD ≤ 8 mmHg)\n")
    f.write("="*80 + "\n")

print("Saved: test_results.txt")


### Final Summary

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print("\n Generated Files:")
print("1. training_curves.png - Training/validation curves")
print("2. bland_altman_plots.png - Agreement analysis")
print("3. error_analysis.png - Error distributions and scatter plots")
print("4. test_predictions.npy - Model predictions")
print("5. test_targets.npy - Ground truth labels")
print("6. test_results.txt - Detailed text report")

print("\nModel evaluation complete!")
print("="*80 + "\n")
