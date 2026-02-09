"""
Configuration file for Sleep Stage Scoring Interface.

INSTRUCTIONS FOR USERS:
1. Update OUT_DIR to point to your data directory
2. Adjust signal processing parameters if needed
3. Customize display settings to your preference
4. Save this file after making changes

SCIENTIFIC METHODS:
- EEG: No filtering (preserves all frequency components)
- EMG: Butterworth high-pass filter (100 Hz cutoff, 4th order, zero-phase)
- Scaling: Median Absolute Deviation (MAD) - robust to outliers
- PSD: Welch's method with Hann window (nperseg=1024, 50% overlap)
- Display clipping: Prevents artifacts from compressing signal (display only)

For detailed scientific documentation, see METHODS.md
"""

from pathlib import Path

# =========================
# DATA DIRECTORY (REQUIRED - UPDATE THIS!)
# =========================
# Point this to the folder containing your sleep data files:
# - all_eeg.npy
# - all_emg.npy
# - all_states.npy (optional)

# Examples:
# Windows: OUT_DIR = Path(r"C:\Users\YourName\Documents\sleep_data")
# Mac/Linux: OUT_DIR = Path("/home/username/sleep_data")

OUT_DIR = Path(r"./data")  # UPDATE THIS PATH!

# =========================
# SIGNAL PARAMETERS
# =========================
FS_HZ = 1000              # Sampling frequency in Hz
EPOCH_LEN_S = 4.0         # Epoch length in seconds

# =========================
# DISPLAY SETTINGS
# =========================
# Context window: how many epochs to show after current epoch
CONTEXT_EPOCHS_AFTER = 4  # Shows current + 4 next epochs (20s total)

# Hypnogram window: how many epochs to display in hypnogram
HYPNO_WINDOW_EPOCHS = 100  # Number of epochs visible at once

# Signal scaling multipliers (MAD-based robust scaling)
# Formula: y_limits = ±(multiplier × 1.4826 × MAD)
# where MAD = median(|x - median(x)|)
# Larger values = more vertical range (signals appear smaller)
# Typical range: 6-10 for EEG, 8-12 for EMG
EEG_MAD_MULT = 8.0        # EEG vertical scale multiplier
EMG_MAD_MULT = 10.0       # EMG vertical scale multiplier

# Display clipping (winsorization) - prevents extreme artifacts from crushing trace
# Applied for visualization only; does NOT modify raw data
# Clips values beyond ±(multiplier × 1.4826 × MAD)
CLIP_MULT_EEG = 10.0      # EEG clipping threshold multiplier
CLIP_MULT_EMG = 12.0      # EMG clipping threshold multiplier

# =========================
# PSD (POWER SPECTRAL DENSITY) SETTINGS
# =========================
# Method: Welch's periodogram
# Window: Hann window
# Overlap: 50% (default in scipy.signal.welch)
# Implementation: scipy.signal.welch

PSD_FMAX = 20.0           # Maximum frequency to display in PSD (Hz)
                          # Covers sleep-relevant bands:
                          # - Delta: 0.5-4 Hz
                          # - Theta: 4-8 Hz
                          # - Alpha: 8-13 Hz
                          # - Beta: 13-30 Hz

N_PER_SEG = 1024          # Number of points per segment for PSD
                          # Frequency resolution = sampling_rate / nperseg
                          # For fs=1000 Hz: Δf = 1000/1024 ≈ 0.98 Hz

# =========================
# AUTO-SAVE SETTINGS
# =========================
AUTOSAVE_INTERVAL = 10    # Auto-save every N epochs scored

# =========================
# COLOR SCHEME
# =========================
COLORS = {
    'Wake': 'black',
    'NREM': 'blue',
    'REM': 'green',
    'MA': 'red',
    'Unscored': 'lightgray'
}

# =========================
# EMG FILTERING
# =========================
# High-pass Butterworth IIR filter
# Purpose: Remove low-frequency movement artifacts, isolate muscle activity
# Method: Zero-phase filtering (forward + backward pass)
# Effective order: 2 × filter_order (due to bidirectional filtering)
EMG_HIGHPASS_CUTOFF_HZ = 100.0  # Cutoff frequency (Hz)
EMG_FILTER_ORDER = 4             # Butterworth filter order

# =========================
# NAVIGATION
# =========================
NAV_DEBOUNCE_MS = 140     # Debounce time for navigation (milliseconds)

# =========================
# OUTPUT FOLDER NAME
# =========================
OUTPUT_FOLDER_NAME = "Hypno-score"  # Name of output subfolder

# =========================
# VERIFY CONFIGURATION
# =========================
if not OUT_DIR.exists():
    print("=" * 60)
    print("⚠️  WARNING: Data directory does not exist!")
    print(f"   OUT_DIR is set to: {OUT_DIR}")
    print("")
    print("   Please update OUT_DIR in config.py to point to your")
    print("   data directory containing:")
    print("   - all_eeg.npy")
    print("   - all_emg.npy")
    print("   - all_states.npy (optional)")
    print("=" * 60)
else:
    print(f"✅ Configuration loaded. Data directory: {OUT_DIR}")
