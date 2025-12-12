fit_data_file = "21014870855_ACTIVITY.fit"
from fitparse import FitFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from pathlib import Path

# Create figures and output_data directories if they don't exist
Path('figures').mkdir(exist_ok=True)
Path('output_data').mkdir(exist_ok=True)

# Explore all message types in the FIT file
fitfile = FitFile(fit_data_file)
message_types = set()
for message in fitfile.get_messages():
    message_types.add(message.name)

# Check for HRV data (R-R intervals)
fitfile2 = FitFile(fit_data_file)
hrv_messages = list(fitfile2.get_messages('hrv'))

if hrv_messages:
    # Extract HRV data
    hrv_data = []
    for msg in hrv_messages:
        msg_data = {}
        for field in msg:
            msg_data[field.name] = field.value
        if msg_data:
            hrv_data.append(msg_data)

    hrv_df = pd.DataFrame(hrv_data)

# Extract and flatten all R-R intervals from the HRV data
fitfile3 = FitFile(fit_data_file)
hrv_messages = fitfile3.get_messages('hrv')

rr_intervals = []
timestamps = []

for msg in hrv_messages:
    msg_dict = {field.name: field.value for field in msg}

    # Get timestamp if available
    timestamp = msg_dict.get('timestamp', None)

    # Extract R-R intervals from the 'time' tuple
    if 'time' in msg_dict and msg_dict['time'] is not None:
        time_values = msg_dict['time']
        if isinstance(time_values, tuple):
            for rr in time_values:
                if rr is not None:
                    rr_intervals.append(rr)
                    timestamps.append(timestamp)

# Create DataFrame with R-R intervals
rr_df = pd.DataFrame({
    'rr_interval_seconds': rr_intervals,
    'rr_interval_ms': [rr * 1000 for rr in rr_intervals],
    'timestamp': timestamps
})

# Calculate instantaneous heart rate from R-R intervals
rr_df['heart_rate_bpm'] = 60/rr_df['rr_interval_seconds']

# Calculate common HRV metrics

# Time-domain metrics
rr_intervals_ms = rr_df['rr_interval_ms'].values

# SDNN - Standard deviation of NN intervals (overall HRV)
sdnn = np.std(rr_intervals_ms, ddof=1)

# RMSSD - Root mean square of successive differences (parasympathetic activity)
successive_diffs = np.diff(rr_intervals_ms)
rmssd = np.sqrt(np.mean(successive_diffs**2))

# pNN50 - Percentage of successive RR intervals that differ by more than 50 ms
nn50 = np.sum(np.abs(successive_diffs) > 50)
pnn50 = (nn50 / len(successive_diffs)) * 100

# Mean HR and HR range
mean_hr = rr_df['heart_rate_bpm'].mean()
min_hr = rr_df['heart_rate_bpm'].min()
max_hr = rr_df['heart_rate_bpm'].max()

# SDSD - Standard deviation of successive differences
sdsd = np.std(successive_diffs, ddof=1)

# Visualize R-R intervals and heart rate

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: R-R Intervals over time
axes[0].plot(rr_df['rr_interval_ms'], linewidth=0.8, color='#2E86AB')
axes[0].set_title('R-R Intervals Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sample Number')
axes[0].set_ylabel('R-R Interval (ms)')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=rr_df['rr_interval_ms'].mean(), color='red',
                linestyle='--', label=f'Mean: {rr_df["rr_interval_ms"].mean():.1f} ms')
axes[0].legend()

# Plot 2: Instantaneous Heart Rate
axes[1].plot(rr_df['heart_rate_bpm'], linewidth=0.8, color='#A23B72')
axes[1].set_title('Instantaneous Heart Rate from R-R Intervals', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sample Number')
axes[1].set_ylabel('Heart Rate (bpm)')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=rr_df['heart_rate_bpm'].mean(), color='red',
                linestyle='--', label=f'Mean: {rr_df["heart_rate_bpm"].mean():.1f} bpm')
axes[1].legend()

# Plot 3: Distribution of R-R Intervals
axes[2].hist(rr_df['rr_interval_ms'], bins=30, color='#18A558', alpha=0.7, edgecolor='black')
axes[2].set_title('Distribution of R-R Intervals', fontsize=14, fontweight='bold')
axes[2].set_xlabel('R-R Interval (ms)')
axes[2].set_ylabel('Frequency')
axes[2].axvline(x=rr_df['rr_interval_ms'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Mean: {rr_df["rr_interval_ms"].mean():.1f} ms')
axes[2].axvline(x=rr_df['rr_interval_ms'].median(), color='orange',
                linestyle='--', linewidth=2, label=f'Median: {rr_df["rr_interval_ms"].median():.1f} ms')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/rr_intervals_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Poincaré Plot - Classic HRV visualization
fig, ax = plt.subplots(figsize=(10, 10))

# Get consecutive R-R intervals
rr_n = rr_intervals_ms[:-1]  # RR(n)
rr_n1 = rr_intervals_ms[1:]  # RR(n+1)

# Create scatter plot
ax.scatter(rr_n, rr_n1, alpha=0.6, s=50, color='#2E86AB', edgecolors='black', linewidth=0.5)

# Add identity line
min_val = min(rr_n.min(), rr_n1.min())
max_val = max(rr_n.max(), rr_n1.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Identity Line')

# Calculate SD1 and SD2 (Poincaré plot indices)
sd1 = np.sqrt(0.5 * np.var(rr_n - rr_n1))
sd2 = np.sqrt(2 * np.var(rr_n) - 0.5 * np.var(rr_n - rr_n1))

ax.set_xlabel('RR(n) - milliseconds', fontsize=12, fontweight='bold')
ax.set_ylabel('RR(n+1) - milliseconds', fontsize=12, fontweight='bold')
ax.set_title('Poincaré Plot of R-R Intervals\n(HRV Phase Space Representation)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Add SD1 and SD2 text
textstr = f'SD1: {sd1:.2f} ms (short-term variability)\nSD2: {sd2:.2f} ms (long-term variability)\nSD1/SD2 ratio: {sd1/sd2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

ax.legend()
plt.tight_layout()
plt.savefig('figures/poincare_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# Extract comprehensive data from record messages including HRV metrics

fitfile_full = FitFile(fit_data_file)
full_data = []

for record in fitfile_full.get_messages('record'):
    record_dict = {}
    for field in record:
        record_dict[field.name] = field.value
    full_data.append(record_dict)

full_df = pd.DataFrame(full_data)

# Save all extracted data to CSV files
rr_df.to_csv('output_data/garmin_rr_intervals.csv', index=False)
full_df.to_csv('output_data/garmin_comprehensive_data.csv', index=False)

# Estimate respiratory rate from HRV data using EDR (ECG-Derived Respiration)

# Resample R-R intervals to regular time series
rr_intervals_ms = rr_df['rr_interval_ms'].values

# Calculate time points for each R-R interval
cumulative_time = np.cumsum(rr_intervals_ms) / 1000  # Convert to seconds
cumulative_time = np.insert(cumulative_time, 0, 0)  # Start at 0

# Create uniformly sampled time series (4 Hz sampling rate)
fs = 4  # 4 Hz sampling frequency
time_uniform = np.arange(0, cumulative_time[-1], 1/fs)

# Interpolate R-R intervals to uniform time grid
rr_uniform = np.interp(time_uniform, cumulative_time[:-1], rr_intervals_ms)

# Apply bandpass filter to extract respiratory frequency (0.1-0.5 Hz = 6-30 breaths/min)
nyquist = fs / 2
low_cut = 0.1 / nyquist  # 6 breaths/min
high_cut = 0.5 / nyquist  # 30 breaths/min

b, a = signal.butter(3, [low_cut, high_cut], btype='band')
rr_filtered = signal.filtfilt(b, a, rr_uniform)

# Perform FFT to find dominant respiratory frequency
n = len(rr_filtered)
yf = fft(rr_filtered)
xf = fftfreq(n, 1/fs)

# Only look at positive frequencies in respiratory range
mask = (xf > 0.1) & (xf < 0.5)
xf_resp = xf[mask]
yf_resp = np.abs(yf[mask])

# Find peak frequency (dominant respiratory frequency)
peak_idx = np.argmax(yf_resp)
resp_freq_hz = xf_resp[peak_idx]
resp_rate_bpm = resp_freq_hz * 60  # Convert to breaths per minute

# Calculate time-varying respiratory rate (moving window analysis)
window_size = 30 * fs  # 30 second windows
hop_size = 5 * fs  # 5 second hops
resp_rates = []
times = []

for i in range(0, len(rr_filtered) - window_size, hop_size):
    window = rr_filtered[i:i+window_size]
    window_fft = np.abs(fft(window))
    window_freq = fftfreq(len(window), 1/fs)

    mask_w = (window_freq > 0.1) & (window_freq < 0.5)
    peak_idx_w = np.argmax(window_fft[mask_w])
    peak_freq_w = window_freq[mask_w][peak_idx_w]
    resp_rates.append(peak_freq_w * 60)
    times.append(time_uniform[i + window_size//2])

resp_rate_df = pd.DataFrame({
    'time_seconds': times,
    'respiratory_rate_bpm': resp_rates
})

# Visualize estimated respiratory rate
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Time-varying respiratory rate
axes[0].plot(resp_rate_df['time_seconds'], resp_rate_df['respiratory_rate_bpm'],linewidth=2, color='#E63946', marker='o', markersize=4)
axes[0].axhline(y=np.mean(resp_rates), color='blue', linestyle='--',linewidth=2, label=f'Mean: {np.mean(resp_rates):.1f} breaths/min')
axes[0].set_title('Estimated Respiratory Rate Over Time\n(Derived from R-R Intervals)',fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time (seconds)', fontsize=11)
axes[0].set_ylabel('Respiratory Rate (breaths/min)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_ylim([0, max(resp_rates) * 1.2])

# Plot 2: Frequency spectrum showing respiratory peak
axes[1].plot(xf_resp * 60, yf_resp, linewidth=1.5, color='#2A9D8F')
axes[1].axvline(x=resp_rate_bpm, color='red', linestyle='--',linewidth=2, label=f'Peak: {resp_rate_bpm:.1f} breaths/min')
axes[1].set_title('Frequency Spectrum of HRV (Respiratory Component)',fontsize=14, fontweight='bold')
axes[1].set_xlabel('Frequency (breaths/min)', fontsize=11)
axes[1].set_ylabel('Power', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_xlim([6, 30])  # Normal respiratory range

plt.tight_layout()
plt.savefig('figures/respiratory_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Save respiratory rate data
resp_rate_df.to_csv('output_data/garmin_estimated_respiratory_rate.csv', index=False)

# Final Summary
print("\n" + "="*70)
print("GARMIN DATA EXTRACTION COMPLETE")
print("="*70)
print(f"\nExtracted {len(rr_df)} R-R intervals | Mean HR: {mean_hr:.1f} bpm")
print(f"HRV Metrics: SDNN={sdnn:.1f}ms, RMSSD={rmssd:.1f}ms, pNN50={pnn50:.1f}%")
print(f"Estimated Respiratory Rate: {np.mean(resp_rates):.1f} breaths/min")
print(f"\nData files: output_data/garmin_rr_intervals.csv, output_data/garmin_comprehensive_data.csv, output_data/garmin_estimated_respiratory_rate.csv")
print(f"Figures: figures/rr_intervals_analysis.png, figures/poincare_plot.png, figures/respiratory_rate_analysis.png")
print("="*70)
