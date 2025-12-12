"""
Processing Module
=================
Process Garmin .fit files and run BP prediction model.
"""

import numpy as np
import torch
import torch.nn as nn
from fitparse import FitFile
from scipy import signal, interpolate
from pathlib import Path
from typing import Dict, Tuple


# Get model path (relative to project root)
MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "best_model_full.pth"

# Fallback to alternative model if needed
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "resnet152_bp_model_full.pth"


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
        # Ensure input is numpy array
        rr_intervals_ms = np.array(rr_intervals_ms)

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

    def process(self, rr_intervals_ms, segment_mode='multi'):
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


# Model Architecture (ResNet152_BP)

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


# Load model globally to avoid reloading
_model = None

def get_model():
    """Load and return the trained model (singleton pattern)."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')
        _model = ResNet152_BP(num_classes=2)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()

    return _model


def extract_rr_intervals_from_fit(fit_file_path: str) -> np.ndarray:
    """
    Extract R-R intervals from Garmin .fit file.

    Args:
        fit_file_path: Path to .fit file

    Returns:
        R-R intervals in milliseconds
    """
    fitfile = FitFile(fit_file_path)
    hrv_messages = fitfile.get_messages('hrv')

    rr_intervals = []

    for msg in hrv_messages:
        msg_dict = {field.name: field.value for field in msg}

        # Extract R-R intervals from the 'time' tuple
        if 'time' in msg_dict and msg_dict['time'] is not None:
            time_values = msg_dict['time']
            if isinstance(time_values, tuple):
                for rr in time_values:
                    if rr is not None:
                        rr_intervals.append(rr * 1000)  # Convert to ms

    return np.array(rr_intervals)


def get_bp_category(sbp: float, dbp: float) -> str:
    """
    Determine BP category based on AHA guidelines.

    Args:
        sbp: Systolic blood pressure (mmHg)
        dbp: Diastolic blood pressure (mmHg)

    Returns:
        Category string
    """
    if sbp < 120 and dbp < 80:
        return "Normal"
    elif sbp < 130 and dbp < 80:
        return "Elevated"
    elif sbp < 140 or dbp < 90:
        return "Stage 1 Hypertension"
    elif sbp < 180 or dbp < 120:
        return "Stage 2 Hypertension"
    else:
        return "Hypertensive Crisis"


def process_fit_file(fit_file_path: str) -> Dict:
    """
    Complete pipeline: .fit file -> BP prediction.

    Args:
        fit_file_path: Path to Garmin .fit file

    Returns:
        Dictionary with predictions and metadata
    """
    try:
        # Extract R-R intervals
        rr_intervals_ms = extract_rr_intervals_from_fit(fit_file_path)

        if len(rr_intervals_ms) < 10:
            return {
                'success': False,
                'error': 'Insufficient R-R interval data in file'
            }

        # Calculate basic HRV metrics
        mean_hr = 60000 / np.mean(rr_intervals_ms)
        hrv_sdnn = np.std(rr_intervals_ms)

        # Process with GarminHRMProcessor
        processor = GarminHRMProcessor()
        processed_tensor, intermediate_signals = processor.process(rr_intervals_ms, segment_mode='multi')

        num_windows = processed_tensor.shape[0]
        total_duration = intermediate_signals['time_axis'][-1]

        # Load model and make predictions
        model = get_model()

        with torch.no_grad():
            predictions = model(processed_tensor)  # (N, 2)

        # Extract SBP and DBP for each window
        sbp_predictions = predictions[:, 0].numpy()
        dbp_predictions = predictions[:, 1].numpy()

        # Calculate statistics
        sbp_mean = float(np.mean(sbp_predictions))
        sbp_std = float(np.std(sbp_predictions))
        sbp_min = float(np.min(sbp_predictions))
        sbp_max = float(np.max(sbp_predictions))

        dbp_mean = float(np.mean(dbp_predictions))
        dbp_std = float(np.std(dbp_predictions))
        dbp_min = float(np.min(dbp_predictions))
        dbp_max = float(np.max(dbp_predictions))

        return {
            'success': True,
            'sbp_mean': sbp_mean,
            'sbp_std': sbp_std,
            'sbp_min': sbp_min,
            'sbp_max': sbp_max,
            'dbp_mean': dbp_mean,
            'dbp_std': dbp_std,
            'dbp_min': dbp_min,
            'dbp_max': dbp_max,
            'num_windows': num_windows,
            'total_duration': total_duration,
            'mean_hr': mean_hr,
            'hrv_sdnn': hrv_sdnn,
            'window_predictions': {
                'sbp': sbp_predictions.tolist(),
                'dbp': dbp_predictions.tolist()
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
