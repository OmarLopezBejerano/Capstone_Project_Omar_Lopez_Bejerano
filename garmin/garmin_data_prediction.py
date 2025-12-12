import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, interpolate
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
Path('figures').mkdir(exist_ok=True)

# Load the R-R interval data from garmin_data_extract.py
garmin_data = pd.read_csv('output_data/garmin_rr_intervals.csv')
rr_intervals_ms = garmin_data['rr_interval_ms'].values

class GarminHRMProcessor:
    """
    Process Garmin HRM-Pro Plus data (R-R intervals) to ResNet152 input format.

    Pipeline:
    1. R-R intervals -> continuous heart rate signal
    2. HR signal -> pseudo-PPG waveform
    3. HRV patterns -> respiratory estimate
    4. Stack as 3 channels -> (3, 1250) for model
    """

    def __init__(self, target_fs=125, target_duration=10):
        self.target_fs = target_fs  # 125 Hz
        self.target_duration = target_duration  # 10 seconds
        self.target_length = target_fs * target_duration  # 1250 samples

    def rr_to_continuous_hr(self, rr_intervals_ms):
        """
        Convert R-R intervals to continuous heart rate signal.

        Args:
            rr_intervals_ms: R-R intervals in milliseconds

        Returns:
            hr_signal: Continuous heart rate at 125 Hz
            time_axis: Time points for each sample
        """
        # Convert R-R to instantaneous HR (bpm)
        hr_instantaneous = 60000 / rr_intervals_ms

        # Create cumulative time from R-R intervals
        time_beats = np.cumsum(rr_intervals_ms) / 1000  # Convert to seconds
        time_beats = np.insert(time_beats, 0, 0)  # Add t=0
        hr_at_beats = np.insert(hr_instantaneous, 0, hr_instantaneous[0])

        # Interpolate to 125 Hz
        duration = time_beats[-1]
        time_continuous = np.linspace(0, duration, int(duration * self.target_fs))

        # Use cubic spline interpolation for smooth HR signal
        interpolator = interpolate.interp1d(time_beats, hr_at_beats,
                                           kind='cubic', fill_value='extrapolate')
        hr_signal = interpolator(time_continuous)

        return hr_signal, time_continuous

    def hr_to_pseudo_ppg(self, hr_signal):
        """
        Generate pseudo-PPG waveform from heart rate signal.

        PPG morphology approximation:
        - Main pulse wave: systolic upstroke
        - Dicrotic notch: diastolic reflection

        Args:
            hr_signal: Continuous HR signal at 125 Hz

        Returns:
            ppg_signal: Synthetic PPG waveform
        """
        # Convert HR to frequency (Hz)
        hr_hz = hr_signal / 60

        # Generate phase accumulation
        phase = np.cumsum(2 * np.pi * hr_hz / self.target_fs)

        # PPG waveform with systolic and dicrotic components
        ppg = np.sin(phase)  # Main pulse
        ppg += 0.3 * np.sin(2 * phase - 0.5)  # Dicrotic notch
        ppg += 0.1 * np.sin(3 * phase - 1.0)  # Harmonics

        # Add HRV-driven amplitude modulation
        hrv_envelope = (hr_signal - hr_signal.mean()) / (hr_signal.std() + 1e-8)
        ppg = ppg * (1 + 0.15 * hrv_envelope)

        return ppg

    def extract_pseudo_ecg(self, ppg_signal):
        """
        Derive pseudo-ECG from PPG using differentiation.

        The second derivative approximates QRS complex morphology.
        """
        # First and second derivatives
        d1 = np.gradient(ppg_signal)
        d2 = np.gradient(d1)

        # Normalize
        pseudo_ecg = d2 / (np.abs(d2).max() + 1e-8)

        return pseudo_ecg

    def extract_pseudo_respiratory(self, hr_signal):
        """
        Extract respiratory component from HRV patterns.

        Respiratory sinus arrhythmia (RSA): HR increases during inspiration,
        decreases during expiration at respiratory frequency (0.1-0.5 Hz).
        """
        # Bandpass filter for respiratory range
        sos = signal.butter(4, [0.1, 0.5], btype='band',
                           fs=self.target_fs, output='sos')
        respiratory = signal.sosfilt(sos, hr_signal - hr_signal.mean())

        # Normalize
        respiratory = respiratory / (np.std(respiratory) + 1e-8)

        return respiratory

    def segment_and_pad(self, signal):
        """
        Extract or pad signal to exactly target_length (1250 samples).
        """
        current_length = len(signal)

        if current_length >= self.target_length:
            # Extract centered window
            start_idx = (current_length - self.target_length) // 2
            return signal[start_idx:start_idx + self.target_length]
        else:
            # Pad with edge values
            pad_needed = self.target_length - current_length
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            return np.pad(signal, (pad_left, pad_right), mode='edge')

    def create_non_overlapping_windows(self, signal):
        """
        Create non-overlapping 10-second windows from signal.

        Args:
            signal: Continuous signal at 125 Hz

        Returns:
            List of windows, each of length target_length (1250 samples)
        """
        current_length = len(signal)
        windows = []

        # Calculate how many complete windows we can extract
        num_windows = current_length // self.target_length

        for i in range(num_windows):
            start_idx = i * self.target_length
            end_idx = start_idx + self.target_length
            windows.append(signal[start_idx:end_idx])

        return windows

    def process(self, rr_intervals_ms, segment_mode='single'):
        """
        Complete processing pipeline.

        Args:
            rr_intervals_ms: R-R intervals from Garmin (milliseconds)
            segment_mode: 'single' (center window only) or 'multi' (all windows)

        Returns:
            If segment_mode='single':
                - Tensor (1, 3, 1250) ready for ResNet152
                - Dict of intermediate signals
            If segment_mode='multi':
                - Tensor (N, 3, 1250) with N windows
                - Dict of intermediate signals
        """
        # Step 1: R-R intervals -> continuous HR
        hr_signal, time_axis = self.rr_to_continuous_hr(rr_intervals_ms)

        # Step 2: HR -> pseudo-PPG
        ppg_signal = self.hr_to_pseudo_ppg(hr_signal)

        # Step 3: Derive pseudo-ECG from PPG
        ecg_signal = self.extract_pseudo_ecg(ppg_signal)

        # Step 4: Extract respiratory from HRV
        resp_signal = self.extract_pseudo_respiratory(hr_signal)

        if segment_mode == 'single':
            # Original behavior: single centered window
            ecg = self.segment_and_pad(ecg_signal)
            ppg = self.segment_and_pad(ppg_signal)
            resp = self.segment_and_pad(resp_signal)

            # Normalize each channel
            ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
            ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
            resp = (resp - resp.mean()) / (resp.std() + 1e-8)

            # Stack into 3 channels
            signals = np.stack([ecg, ppg, resp], axis=0)  # (3, 1250)

            # Convert to PyTorch tensor
            tensor = torch.FloatTensor(signals).unsqueeze(0)  # (1, 3, 1250)

        else:  # segment_mode == 'multi'
            # Extract all non-overlapping windows
            ecg_windows = self.create_non_overlapping_windows(ecg_signal)
            ppg_windows = self.create_non_overlapping_windows(ppg_signal)
            resp_windows = self.create_non_overlapping_windows(resp_signal)

            num_windows = len(ecg_windows)
            all_windows = []

            for i in range(num_windows):
                ecg = ecg_windows[i]
                ppg = ppg_windows[i]
                resp = resp_windows[i]

                # Normalize each channel
                ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
                ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
                resp = (resp - resp.mean()) / (resp.std() + 1e-8)

                # Stack into 3 channels
                signals = np.stack([ecg, ppg, resp], axis=0)  # (3, 1250)
                all_windows.append(signals)

            # Convert to PyTorch tensor (N, 3, 1250)
            tensor = torch.FloatTensor(np.stack(all_windows, axis=0))

        return tensor, {
            'hr_signal': hr_signal,
            'ppg_signal': ppg_signal,
            'ecg_signal': ecg_signal,
            'resp_signal': resp_signal,
            'time_axis': time_axis
        }

processor = GarminHRMProcessor()

# Process the R-R intervals in MULTI mode to get all windows
processed_tensor, intermediate_signals = processor.process(rr_intervals_ms, segment_mode='multi')
num_windows = processed_tensor.shape[0]

# Visualize the complete processing pipeline
fig, axes = plt.subplots(5, 1, figsize=(15, 12))

# Time axis for 10-second window
t = np.linspace(0, 10, 1250)

# Get processed signals from FIRST window for visualization
ecg = processed_tensor[0, 0].numpy()
ppg = processed_tensor[0, 1].numpy()
resp = processed_tensor[0, 2].numpy()

# Also show the full HR signal for context
t_hr = intermediate_signals['time_axis']
hr_full = intermediate_signals['hr_signal']

# Plot 1: Original R-R intervals
axes[0].stem(np.cumsum(rr_intervals_ms[:50])/1000, 60000/rr_intervals_ms[:50],
             linefmt='b-', markerfmt='bo', basefmt=' ')
axes[0].set_title('Original Garmin Data: R-R Intervals -> Instantaneous HR',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('Heart Rate (bpm)')
axes[0].set_xlabel('Time (seconds)')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 40)

# Plot 2: Continuous HR signal (full duration)
axes[1].plot(t_hr, hr_full, linewidth=1, color='red')
axes[1].set_title('Reconstructed Continuous HR Signal (125 Hz)',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Heart Rate (bpm)')
axes[1].set_xlabel('Time (seconds)')
axes[1].grid(True, alpha=0.3)
axes[1].axvspan(0, 10, alpha=0.2, color='green', label='10s window for model')
axes[1].legend()

# Plot 3: Processed ECG (10-second window)
axes[2].plot(t, ecg, linewidth=0.8, color='darkred')
axes[2].set_title('Derived Pseudo-ECG (10s window for ResNet152)',
                  fontsize=12, fontweight='bold')
axes[2].set_ylabel('Amplitude (normalized)')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 10)

# Plot 4: Processed PPG (10-second window)
axes[3].plot(t, ppg, linewidth=0.8, color='darkgreen')
axes[3].set_title('Derived Pseudo-PPG (10s window for ResNet152)',
                  fontsize=12, fontweight='bold')
axes[3].set_ylabel('Amplitude (normalized)')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(0, 10)

# Plot 5: Processed Respiratory (10-second window)
axes[4].plot(t, resp, linewidth=0.8, color='purple')
axes[4].set_title('Extracted Respiratory Signal (10s window for ResNet152)',
                  fontsize=12, fontweight='bold')
axes[4].set_ylabel('Amplitude (normalized)')
axes[4].set_xlabel('Time (seconds)')
axes[4].grid(True, alpha=0.3)
axes[4].set_xlim(0, 10)

plt.tight_layout()
plt.savefig('figures/garmin_hrm_processing_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()

# Define ResNet152_BP architecture
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


# Load FULL trained model (1524 patients)
MODEL_PATH = Path('../model/best_model_full.pth')

# Fallback to old model if full model not available
if not MODEL_PATH.exists():
    MODEL_PATH = Path('../model/resnet152_bp_model_full.pth')

if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')

    model = ResNet152_BP(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get model info for summary
    model_info = {}
    if 'epoch' in checkpoint:
        model_info['epochs'] = checkpoint['epoch'] + 1
    if 'val_loss' in checkpoint:
        model_info['val_loss'] = checkpoint['val_loss']
    if 'history' in checkpoint:
        history = checkpoint['history']
        if 'val_mae_sbp' in history and len(history['val_mae_sbp']) > 0:
            best_epoch = checkpoint.get('epoch', len(history['val_mae_sbp'])-1)
            model_info['sbp_mae'] = history['val_mae_sbp'][best_epoch]
            model_info['dbp_mae'] = history['val_mae_dbp'][best_epoch]
    elif 'test_metrics' in checkpoint:
        metrics = checkpoint['test_metrics']
        if 'sbp_mae' in metrics:
            model_info['sbp_mae'] = metrics['sbp_mae']
        if 'dbp_mae' in metrics:
            model_info['dbp_mae'] = metrics['dbp_mae']
else:
    model = None
    model_info = {}


if model is not None:
    # Make predictions for ALL windows
    with torch.no_grad():
        predictions = model(processed_tensor)  # (N, 2)

    # Extract SBP and DBP for each window
    sbp_predictions = predictions[:, 0].numpy()
    dbp_predictions = predictions[:, 1].numpy()

    # Calculate statistics
    sbp_mean = np.mean(sbp_predictions)
    sbp_std = np.std(sbp_predictions)
    sbp_min = np.min(sbp_predictions)
    sbp_max = np.max(sbp_predictions)

    dbp_mean = np.mean(dbp_predictions)
    dbp_std = np.std(dbp_predictions)
    dbp_min = np.min(dbp_predictions)
    dbp_max = np.max(dbp_predictions)

    # Use mean values for overall assessment
    sbp_pred = sbp_mean
    dbp_pred = dbp_mean
    pulse_pressure = sbp_pred - dbp_pred

    # Blood pressure category (based on mean)
    if sbp_pred < 120 and dbp_pred < 80:
        category = "Normal"
    elif sbp_pred < 130 and dbp_pred < 80:
        category = "Elevated"
    elif sbp_pred < 140 or dbp_pred < 90:
        category = "Stage 1 Hypertension"
    elif sbp_pred < 180 or dbp_pred < 120:
        category = "Stage 2 Hypertension"
    else:
        category = "Hypertensive Crisis"

    # Assess variability (high SD suggests unstable reading or model uncertainty)
    high_variability = sbp_std > 10 or dbp_std > 8

    # Calculate total recording duration
    total_duration = intermediate_signals['time_axis'][-1]

    # Final Summary
    print("\n" + "="*80)
    print("GARMIN HRM BLOOD PRESSURE PREDICTION - MULTI-WINDOW ANALYSIS")
    print("="*80)
    print(f"\nRecording Info:")
    print(f"  Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"  R-R Intervals: {len(rr_intervals_ms)}")
    print(f"  Mean HR: {60000/np.mean(rr_intervals_ms):.1f} bpm")
    print(f"  HRV (SDNN): {np.std(rr_intervals_ms):.1f} ms")
    print(f"  Analysis Windows: {num_windows} non-overlapping 10-second segments")

    print(f"\n{'─'*80}")
    print("Blood Pressure Predictions:")
    print(f"{'─'*80}")
    print(f"  Systolic BP:  {sbp_mean:.1f} ± {sbp_std:.1f} mmHg  (range: {sbp_min:.1f} - {sbp_max:.1f})")
    print(f"  Diastolic BP: {dbp_mean:.1f} ± {dbp_std:.1f} mmHg  (range: {dbp_min:.1f} - {dbp_max:.1f})")
    print(f"  Mean Arterial Pressure: {(sbp_mean + 2*dbp_mean)/3:.1f} mmHg")
    print(f"  Pulse Pressure: {pulse_pressure:.1f} mmHg")
    print(f"\n  Category: {category}")

    if high_variability:
        print(f"\n  ⚠️  HIGH VARIABILITY DETECTED")
        print(f"      Large BP fluctuations may indicate:")
        print(f"      - Unstable physiological state")
        print(f"      - Movement artifacts in recording")
        print(f"      - Model uncertainty")

    print(f"\n{'─'*80}")
    print("Per-Window Predictions:")
    print(f"{'─'*80}")
    for i in range(num_windows):
        time_start = i * 10
        time_end = time_start + 10
        print(f"  Window {i+1:2d} ({time_start:3d}-{time_end:3d}s): "
              f"SBP={sbp_predictions[i]:5.1f} mmHg, DBP={dbp_predictions[i]:5.1f} mmHg")

    if model_info:
        print(f"\n{'─'*80}")
        print("Model Performance Metrics:")
        print(f"{'─'*80}")
        for k, v in model_info.items():
            if isinstance(v, float):
                print(f"  {k.upper()}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    # Create BP trend visualization with confidence intervals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Time points (center of each 10-second window)
    time_points = np.arange(num_windows) * 10 + 5  # Center at 5, 15, 25, ... seconds

    # Plot SBP with confidence interval
    ax1.plot(time_points, sbp_predictions, 'o-', color='#d62728', linewidth=2,
             markersize=6, label='Systolic BP')
    ax1.fill_between(time_points,
                      sbp_predictions - sbp_std,
                      sbp_predictions + sbp_std,
                      alpha=0.3, color='#d62728', label=f'±1 SD ({sbp_std:.1f} mmHg)')
    ax1.axhline(sbp_mean, color='#d62728', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Mean: {sbp_mean:.1f} mmHg')

    # Reference zones for SBP
    ax1.axhspan(90, 120, alpha=0.1, color='green', label='Normal')
    ax1.axhspan(120, 130, alpha=0.1, color='yellow')
    ax1.axhspan(130, 140, alpha=0.1, color='orange')

    ax1.set_ylabel('Systolic BP (mmHg)', fontsize=12, fontweight='bold')
    ax1.set_title('Blood Pressure Temporal Trends - Multi-Window Analysis',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(max(80, sbp_min - 10), min(160, sbp_max + 10))

    # Plot DBP with confidence interval
    ax2.plot(time_points, dbp_predictions, 'o-', color='#1f77b4', linewidth=2,
             markersize=6, label='Diastolic BP')
    ax2.fill_between(time_points,
                      dbp_predictions - dbp_std,
                      dbp_predictions + dbp_std,
                      alpha=0.3, color='#1f77b4', label=f'±1 SD ({dbp_std:.1f} mmHg)')
    ax2.axhline(dbp_mean, color='#1f77b4', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Mean: {dbp_mean:.1f} mmHg')

    # Reference zones for DBP
    ax2.axhspan(60, 80, alpha=0.1, color='green', label='Normal')
    ax2.axhspan(80, 90, alpha=0.1, color='yellow')
    ax2.axhspan(90, 100, alpha=0.1, color='orange')

    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Diastolic BP (mmHg)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(max(40, dbp_min - 10), min(100, dbp_max + 10))
    ax2.set_xlim(0, total_duration)

    plt.tight_layout()
    plt.savefig('figures/garmin_bp_temporal_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'─'*80}")
    print(f"Figures saved:")
    print(f"  - figures/garmin_hrm_processing_pipeline.png")
    print(f"  - figures/garmin_bp_temporal_trends.png")
    print("="*80 + "\n")
else:
    print("\nERROR: Model file not found. Cannot make prediction.")
