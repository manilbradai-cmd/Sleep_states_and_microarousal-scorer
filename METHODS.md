# Scientific Methods and Signal Processing

## Overview

This document describes the signal processing methods, algorithms, and scientific approaches used in the Sleep Stage Scoring Interface.

---

## 1. Signal Processing Methods

### 1.1 EEG Signal Processing

#### Filtering
- **Type**: No filtering applied to EEG for display
- **Rationale**: Preserves all frequency components for sleep stage discrimination
- **Display**: Raw signal with robust amplitude scaling

#### Amplitude Scaling
- **Method**: Median Absolute Deviation (MAD) based scaling
- **Formula**: 
  ```
  σ_robust = 1.4826 × MAD
  MAD = median(|x - median(x)|)
  y_limits = ±(multiplier × σ_robust)
  ```
- **Default multiplier**: 8.0 (configurable in `config.py`)
- **Advantages**:
  - Robust to outliers and artifacts
  - Does not assume Gaussian distribution
  - Resistant to extreme values

#### Display Clipping (Winsorization)
- **Method**: Symmetric clipping based on MAD
- **Threshold**: ±10 × σ_robust (configurable)
- **Purpose**: Prevent single artifacts from compressing visible signal
- **Note**: Clipping is for display only; raw data unchanged

---

### 1.2 EMG Signal Processing

#### High-Pass Filtering
- **Filter Type**: Butterworth IIR (Infinite Impulse Response)
- **Cutoff Frequency**: 100 Hz (configurable)
- **Filter Order**: 4 (configurable)
- **Implementation**: Zero-phase filtering using `scipy.signal.filtfilt`
- **Purpose**: Remove low-frequency movement artifacts and isolate muscle activity

**Transfer Function**:
```
H(s) = 1 / (1 + (ωc/s)^(2n))
where:
  ωc = cutoff frequency (100 Hz)
  n = filter order (4)
  s = Laplace variable
```

**Digital Implementation**:
- Bi-directional filtering (forward and backward pass)
- Zero phase distortion
- Effective filter order: 2 × 4 = 8

#### Amplitude Scaling
- **Method**: MAD-based (same as EEG)
- **Default multiplier**: 10.0
- **Applied after**: High-pass filtering

---

## 2. Power Spectral Density (PSD) Analysis

### 2.1 Method: Welch's Method

**Implementation**: `scipy.signal.welch`

**Parameters**:
- **Window**: Hann window
- **Segment length** (`nperseg`): 1024 samples (configurable)
- **Overlap**: 50% (default in scipy)
- **Sampling frequency**: 1000 Hz (configurable)
- **FFT length**: Same as segment length

**Mathematical Formulation**:

For a signal x[n] divided into K overlapping segments:

```
PSD(f) = (1/K) × Σ(k=1 to K) |FFT(x_k[n] × w[n])|²
```

where:
- x_k[n] = k-th segment of the signal
- w[n] = Hann window function
- FFT = Fast Fourier Transform

**Hann Window**:
```
w[n] = 0.5 × (1 - cos(2πn/N))
for n = 0, 1, ..., N-1
```

### 2.2 Normalization

**Method**: Maximum normalization

```
PSD_normalized(f) = PSD(f) / max(PSD(f))
```

**Purpose**: 
- Compare spectral shapes across epochs
- Ignore absolute power differences
- Focus on relative frequency content

### 2.3 Frequency Range

- **Maximum frequency**: 20 Hz (configurable)
- **Rationale**: Sleep-relevant frequencies (delta, theta, alpha, beta)
  - Delta: 0.5-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-30 Hz

---

## 3. Epoch Segmentation

### 3.1 Temporal Parameters

- **Epoch length**: 4.0 seconds (configurable)
- **Sampling rate**: 1000 Hz (configurable)
- **Samples per epoch**: 4000 samples
- **Standard**: Based on AASM guidelines (30s epochs) scaled to 4s for microarousal detection

### 3.2 Context Window

- **Current epoch**: 0-4 seconds
- **Context epochs**: 4 additional epochs (configurable)
- **Total display**: 20 seconds (5 epochs)
- **Purpose**: Provide temporal context for scoring decisions

---

## 4. Sleep Stage Classification Criteria

### 4.1 Wake

**EEG Characteristics**:
- Low amplitude (10-30 μV)
- High frequency (alpha 8-13 Hz, beta >13 Hz)
- Eyes closed: Alpha rhythm
- Eyes open: Beta activity

**EMG Characteristics**:
- Moderate to high muscle tone
- Variable activity

### 4.2 NREM Sleep

**EEG Characteristics**:
- High amplitude (50-150 μV)
- Low frequency (delta 0.5-4 Hz, theta 4-8 Hz)
- Slow wave activity
- Sleep spindles (11-16 Hz)

**EMG Characteristics**:
- Reduced muscle tone compared to wake
- Stable, moderate amplitude

### 4.3 REM Sleep

**EEG Characteristics**:
- Low amplitude (similar to wake)
- Mixed frequency (theta 4-8 Hz predominant)
- Sawtooth waves
- Resembles wake but different context

**EMG Characteristics**:
- Very low muscle tone (muscle atonia)
- Minimal activity
- Occasional phasic twitches

---

## 5. Microarousal Detection

### 5.1 Definition (AASM Criteria)

**Criteria**:
- Duration: 3-15 seconds
- EEG frequency shift: Increase in alpha, theta, and/or >16 Hz activity
- Must occur during NREM or REM sleep
- Concurrent EMG increase in NREM (optional but common)

### 5.2 Visual Scoring Approach

Manual identification based on:
1. Brief EEG frequency acceleration
2. Return to background sleep pattern
3. Duration within 3-15 second window
4. Context: Surrounding sleep stage

---

## 6. Robust Statistics

### 6.1 Median Absolute Deviation (MAD)

**Formula**:
```
MAD = median(|x_i - median(x)|)
```

**Robust Standard Deviation Estimate**:
```
σ_robust = 1.4826 × MAD
```

**Constant 1.4826**: Consistency factor for Gaussian distribution

**Advantages**:
- Robust to outliers (breakdown point: 50%)
- Does not assume normality
- Suitable for artifact-contaminated signals

### 6.2 Comparison with Standard Deviation

| Metric | Sensitivity to Outliers | Breakdown Point | Assumption |
|--------|------------------------|-----------------|------------|
| SD     | High                   | 0%              | Gaussian   |
| MAD    | Low                    | 50%             | None       |

---

## 7. Data Structures

### 7.1 Input Data Format

**EEG Data** (`all_eeg.npy`):
- Shape: (n_epochs, samples_per_epoch)
- Data type: float64
- Units: μV (microvolts)
- Example: (1000, 4000) for 1000 epochs at 1 kHz

**EMG Data** (`all_emg.npy`):
- Shape: (n_epochs, samples_per_epoch)
- Data type: float64
- Units: μV (microvolts)

**State Codes** (`all_states.npy`):
- Shape: (n_epochs,)
- Data type: int
- Values: -1=Unscored, 1=Wake, 2=NREM, 3=REM

### 7.2 Output Data Format

**CSV Structure** (`sleep_stages_and_microarousals_manual.csv`):

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| epoch_index | int | 0 to n-1 | Epoch number |
| state_code | int | -1, 1, 2, 3 | Sleep stage code |
| state_name | str | Unscored, Wake, NREM, REM | Sleep stage name |
| label_ma | int | -1, 0, 1 | Microarousal label |

**Hypnogram Image** (`hypnogram_full.png`):
- Format: PNG
- Resolution: 150 DPI
- Size: 20 × 4 inches (3000 × 600 pixels)
- Color coding: As per configuration

---

## 8. Visualization Methods

### 8.1 Hypnogram

**Type**: Vertical column representation
- Each epoch = one vertical bar
- Height: Fixed (1.0)
- Color: Indicates sleep stage
- Width: 1 epoch unit

**Advantages**:
- Compact visualization
- Clear temporal progression
- Easy identification of stage transitions

### 8.2 Signal Display

**Time-Domain Plots**:
- X-axis: Time (seconds)
- Y-axis: Amplitude (μV)
- Fixed global scale across epochs
- Yellow highlighting: Current epoch

**Frequency-Domain Plot (PSD)**:
- X-axis: Frequency (Hz)
- Y-axis: Normalized power (0-1)
- Two traces:
  - Gray: Context window (20s)
  - Blue: Current epoch (4s)

---

## 9. Quality Control

### 9.1 Artifact Detection Indicators

**Flatline Detection**:
- Method: Standard deviation < 1.0 μV
- Warning: "⚠ EEG near-flat"
- Indicates: Electrode disconnection or signal loss

### 9.2 Signal Quality Metrics

**Visual Inspection**:
- Signal amplitude range
- Frequency content (PSD)
- Temporal continuity
- EMG/EEG correlation

---

## 10. Algorithm Complexity

### 10.1 Computational Performance

**PSD Calculation**:
- Time complexity: O(n log n) per epoch (FFT)
- Space complexity: O(n)

**MAD Calculation**:
- Time complexity: O(n log n) (sorting for median)
- Space complexity: O(n)

**Filtering (Butterworth)**:
- Time complexity: O(n) per epoch
- Space complexity: O(n)

### 10.2 Real-Time Performance

- Loading: ~2-5 seconds for 1000 epochs
- Navigation: <100 ms between epochs
- Rendering: ~50-100 ms per redraw
- Auto-save: <500 ms

---

## 11. References

### Signal Processing
1. Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms". IEEE Transactions on Audio and Electroacoustics. 15 (2): 70–73.

2. Oppenheim, A.V., & Schafer, R.W. (2009). Discrete-Time Signal Processing (3rd ed.). Prentice Hall.

3. Butterworth, S. (1930). "On the Theory of Filter Amplifiers". Experimental Wireless and the Wireless Engineer. 7: 536–541.

### Sleep Scoring
4. Berry, R.B., et al. (2012). The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications. American Academy of Sleep Medicine.

5. Iber, C., et al. (2007). The AASM Manual for the Scoring of Sleep and Associated Events. American Academy of Sleep Medicine.

### Microarousals
6. ASDA (1992). "EEG arousals: scoring rules and examples". Sleep. 15(2): 173-184.

7. Halász, P., et al. (2004). "The nature of arousal in sleep". Journal of Sleep Research. 13(1): 1-23.

### Robust Statistics
8. Rousseeuw, P.J., & Croux, C. (1993). "Alternatives to the Median Absolute Deviation". Journal of the American Statistical Association. 88(424): 1273-1283.

9. Leys, C., et al. (2013). "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median". Journal of Experimental Social Psychology. 49(4): 764-766.

---

## 12. Validation and Best Practices

### 12.1 Recommended Workflow

1. **Initial Setup**:
   - Verify sampling rate matches data
   - Check epoch length configuration
   - Confirm filter parameters appropriate for signal

2. **Scoring Session**:
   - Review PSD for frequency content
   - Use context window for continuity
   - Mark uncertain epochs for review
   - Save frequently

3. **Quality Assurance**:
   - Review statistics for anomalies
   - Check hypnogram for physiological plausibility
   - Verify microarousal distribution
   - Compare with automated scoring (if available)

### 12.2 Inter-Rater Reliability

**Recommended Metrics**:
- Cohen's Kappa for sleep stages
- Epoch-by-epoch agreement
- Confusion matrix for stage misclassifications
- Microarousal detection sensitivity/specificity

---

## 13. Future Enhancements

Potential methodological additions:
- [ ] Automated artifact rejection
- [ ] Machine learning pre-scoring
- [ ] Multi-channel EEG support
- [ ] Additional sleep stages (N1, N2, N3 sub-classification)
- [ ] Spectral feature extraction
- [ ] Heart rate variability integration
- [ ] Respiratory event detection

---

**Last Updated**: February 2025  
**Version**: 1.0  
**Author**: Manil Bradai
