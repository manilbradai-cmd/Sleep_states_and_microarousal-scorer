# Signal Processing Methods

## Overview

This document describes the signal processing algorithms and computational methods used in the Sleep Stage Scoring Interface.

---

## 1. EEG Signal Processing

### 1.1 Filtering
- **Applied filtering**: None
- **Rationale**: Preserves all frequency components for sleep stage discrimination
- **Display**: Raw signal with robust amplitude scaling

### 1.2 Amplitude Scaling

**Method**: Median Absolute Deviation (MAD) based scaling

**Formula**:
```
MAD = median(|x - median(x)|)
σ_robust = 1.4826 × MAD
y_limits = ±(multiplier × σ_robust)
```

**Default multiplier**: 8.0 (configurable in `config.py`)

**Constant 1.4826**: Consistency factor for Gaussian distribution equivalence

**Advantages**:
- Robust to outliers (50% breakdown point)
- Does not assume Gaussian distribution
- Resistant to extreme values and artifacts

### 1.3 Display Clipping (Winsorization)

**Method**: Symmetric clipping based on MAD

**Threshold**: ±(10 × σ_robust) - default, configurable

**Purpose**: Prevent single artifacts from compressing visible signal range

**Important**: Clipping is applied for display only; raw data remains unchanged

---

## 2. EMG Signal Processing

### 2.1 High-Pass Filtering

**Filter Type**: Butterworth IIR (Infinite Impulse Response)

**Specifications**:
- Cutoff frequency: 100 Hz (configurable)
- Filter order: 4 (configurable)
- Implementation: Zero-phase filtering using `scipy.signal.filtfilt`

**Transfer Function**:
```
H(s) = 1 / (1 + (ωc/s)^(2n))

where:
  ωc = cutoff frequency (100 Hz)
  n = filter order (4)
  s = Laplace variable
```

**Digital Implementation**:
- Forward-backward filtering (bidirectional)
- Zero phase distortion
- Effective filter order: 2 × 4 = 8

**Purpose**: Remove low-frequency movement artifacts and isolate muscle activity

### 2.2 Amplitude Scaling

**Method**: MAD-based (same as EEG)

**Default multiplier**: 10.0

**Applied**: After high-pass filtering

---

## 3. Power Spectral Density (PSD) Analysis

### 3.1 Method: Welch's Periodogram

**Implementation**: `scipy.signal.welch`

**Parameters**:
| Parameter | Value | Notes |
|-----------|-------|-------|
| Window type | Hann | Default in scipy |
| Segment length (`nperseg`) | 1024 samples | Configurable |
| Overlap | 50% | Default in scipy |
| Sampling frequency | 1000 Hz | Configurable |
| FFT length | nperseg | Same as segment length |

**Mathematical Formulation**:

For a signal x[n] divided into K overlapping segments:

```
PSD(f) = (1/K) × Σ(k=1 to K) |FFT(x_k[n] × w[n])|²

where:
  x_k[n] = k-th segment of the signal
  w[n] = Hann window function
  FFT = Fast Fourier Transform
```

**Hann Window Function**:
```
w[n] = 0.5 × (1 - cos(2πn/N))

for n = 0, 1, ..., N-1
where N = segment length
```

### 3.2 Normalization

**Method**: Maximum normalization

```
PSD_normalized(f) = PSD(f) / max(PSD(f))
```

**Purpose**:
- Compare spectral shapes across epochs
- Ignore absolute power differences
- Focus on relative frequency content

### 3.3 Frequency Range

**Maximum frequency displayed**: 20 Hz (configurable)

**Rationale**: Captures sleep-relevant frequency bands
- Delta: 0.5-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-13 Hz
- Beta: 13-30 Hz

**Frequency resolution**: `sampling_rate / nperseg`
- Example: 1000 Hz / 1024 ≈ 0.98 Hz

---

## 4. Epoch Segmentation

### 4.1 Temporal Parameters

| Parameter | Default Value | Unit | Configurable |
|-----------|--------------|------|--------------|
| Epoch length | 4.0 | seconds | Yes |
| Sampling rate | 1000 | Hz | Yes |
| Samples per epoch | 4000 | samples | Derived |

### 4.2 Context Window

**Display configuration**:
- Current epoch: 0-4 seconds (highlighted)
- Context epochs: 4 additional epochs (configurable)
- Total display window: 20 seconds (5 epochs)

**Purpose**: Provide temporal context for manual scoring decisions

---

## 5. Robust Statistical Methods

### 5.1 Median Absolute Deviation (MAD)

**Definition**:
```
MAD = median(|x_i - median(x)|)
```

**Conversion to Robust Standard Deviation**:
```
σ_robust = 1.4826 × MAD
```

**Properties**:
| Property | Value |
|----------|-------|
| Breakdown point | 50% |
| Gaussian efficiency | 37% |
| Outlier sensitivity | Low |
| Distributional assumption | None |

### 5.2 Comparison: MAD vs Standard Deviation

| Metric | Outlier Sensitivity | Breakdown Point | Assumption |
|--------|-------------------|-----------------|------------|
| Standard Deviation (SD) | High | 0% | Gaussian |
| Median Absolute Deviation (MAD) | Low | 50% | None |

**Breakdown point**: Proportion of outliers that can be present before the estimator becomes unreliable.

---

## 6. Data Structures

### 6.1 Input Data Format

**EEG Data** (`all_eeg.npy`):
```python
Shape: (n_epochs, samples_per_epoch)
Data type: float64
Units: μV (microvolts)
Example: (1000, 4000)  # 1000 epochs at 1 kHz for 4s each
```

**EMG Data** (`all_emg.npy`):
```python
Shape: (n_epochs, samples_per_epoch)
Data type: float64
Units: μV (microvolts)
```

**State Codes** (`all_states.npy`):
```python
Shape: (n_epochs,)
Data type: int
Values: 
  -1 = Unscored
   1 = Wake
   2 = NREM
   3 = REM
```

### 6.2 Output Data Format

**CSV Structure** (`sleep_stages_and_microarousals_manual.csv`):

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `epoch_index` | int | 0 to n-1 | Epoch number |
| `state_code` | int | -1, 1, 2, 3 | Sleep stage code |
| `state_name` | str | Text | Sleep stage name |
| `label_ma` | int | -1, 0, 1 | Microarousal label |

**Hypnogram Image** (`hypnogram_full.png`):
```
Format: PNG
Resolution: 150 DPI
Dimensions: 20 × 4 inches (3000 × 600 pixels)
Color encoding: Based on sleep stage/microarousal
```

---

## 7. Visualization Methods

### 7.1 Hypnogram

**Representation**: Vertical column format

**Specifications**:
- Each epoch = one vertical bar
- Bar height: Fixed (1.0 normalized unit)
- Bar color: Indicates sleep stage
- Bar width: 1 epoch unit
- X-axis: Epoch index (time)
- Y-axis: Not used (stage encoded by color)

### 7.2 Time-Domain Signal Plots

**EEG and EMG Plots**:
```
X-axis: Time (seconds)
Y-axis: Amplitude (μV)
Scale: Fixed global scale (MAD-based)
Highlighting: Yellow background for current epoch
Grid: Vertical lines at epoch boundaries
```

### 7.3 Frequency-Domain Plot (PSD)

**Specifications**:
```
X-axis: Frequency (Hz)
Y-axis: Normalized power (0-1 scale)
Traces:
  - Gray: Context window (entire 20s)
  - Blue: Current epoch (4s)
```

---

## 8. Quality Control

### 8.1 Flatline Detection

**Method**: Standard deviation threshold

**Criterion**: `std(EEG) < 1.0 μV`

**Indicator**: Warning message "⚠ EEG near-flat"

**Interpretation**: Possible electrode disconnection or signal loss

### 8.2 Global Scale Calculation

**Sampling strategy**:
- Sample every 10th epoch, or
- Maximum 100 epochs total

**Purpose**:
- Calculate representative global scale
- Ensure consistent display across all epochs
- Reduce computation time

---

## 9. Algorithm Complexity

### 9.1 Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| PSD (Welch) | O(n log n) | FFT per epoch |
| MAD calculation | O(n log n) | Sorting for median |
| Butterworth filter | O(n) | Per epoch |
| Display rendering | O(n) | Per redraw |

### 9.2 Space Complexity

| Data Structure | Complexity | Notes |
|----------------|-----------|-------|
| Signal buffers | O(n) | Per epoch |
| PSD arrays | O(nperseg) | Frequency domain |
| Filter state | O(n) | Temporary |

### 9.3 Performance Metrics

**Typical performance** (1000 epochs, 1 kHz sampling):
- Initial loading: 2-5 seconds
- Epoch navigation: <100 ms
- Display redraw: 50-100 ms
- Auto-save: <500 ms

---

## 10. Configuration Parameters

### 10.1 Adjustable Parameters

All parameters can be modified in `config.py`:

**Signal Processing**:
- `FS_HZ`: Sampling frequency
- `EPOCH_LEN_S`: Epoch length
- `EMG_HIGHPASS_CUTOFF_HZ`: EMG filter cutoff
- `EMG_FILTER_ORDER`: EMG filter order

**Display**:
- `EEG_MAD_MULT`: EEG scale multiplier
- `EMG_MAD_MULT`: EMG scale multiplier
- `CLIP_MULT_EEG`: EEG clipping threshold
- `CLIP_MULT_EMG`: EMG clipping threshold
- `CONTEXT_EPOCHS_AFTER`: Context window size
- `HYPNO_WINDOW_EPOCHS`: Hypnogram display width

**PSD**:
- `PSD_FMAX`: Maximum frequency
- `N_PER_SEG`: Segment length

**Other**:
- `AUTOSAVE_INTERVAL`: Auto-save frequency
- `NAV_DEBOUNCE_MS`: Navigation debounce time

---

## 11. Implementation Details

### 11.1 Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.21.0 | Array operations, signal data |
| SciPy | ≥1.7.0 | Signal processing (filtering, PSD) |
| Pandas | ≥1.3.0 | Data management, CSV I/O |
| Matplotlib | ≥3.4.0 | Visualization, plotting |
| Tkinter | Built-in | GUI framework |

### 11.2 Key Functions

**Signal Processing**:
- `emg_highpass()`: Apply Butterworth high-pass filter
- `normalized_psd()`: Calculate Welch PSD
- `mad()`: Calculate median absolute deviation
- `robust_limits_mad()`: Calculate robust y-limits
- `clip_for_display()`: Apply winsorization

**Visualization**:
- `draw_hypnogram()`: Render hypnogram
- `redraw()`: Update all plots
- `get_window()`: Extract display window

**Data Management**:
- `save_labels()`: Save CSV
- `save_full_hypnogram()`: Export hypnogram image
- `auto_save_check()`: Trigger auto-save

---

## 12. Computation Workflow

### 12.1 Initialization

1. Load configuration from `config.py`
2. Load data files (EEG, EMG, states)
3. Validate data shapes and parameters
4. Sample data for global scale calculation
5. Calculate MAD-based y-limits
6. Initialize GUI components

### 12.2 Per-Epoch Processing

1. Extract epoch window (current + context)
2. Apply EMG high-pass filter
3. Apply display clipping (winsorization)
4. Calculate PSD (Welch method)
5. Normalize PSD
6. Render time-domain plots
7. Render frequency-domain plot
8. Update hypnogram

### 12.3 User Interaction

1. Capture keyboard/mouse events
2. Update sleep stage or microarousal labels
3. Increment change counter
4. Check auto-save threshold
5. Redraw display
6. Update status bar

---

**Last Updated**: February 2025  
**Version**: 1.0  
**Author**: Manil Bradai
