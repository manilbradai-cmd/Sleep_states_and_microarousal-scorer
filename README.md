# Sleep Stage Scoring Interface

A Python-based GUI application for manual sleep stage scoring and microarousal detection with real-time hypnogram visualization.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

**Author**: Manil Bradai  
**GitHub**: [@manilbradai-cmd](https://github.com/manilbradai-cmd)

---

##  Features

-  **Manual Sleep Stage Scoring** - Score Wake, NREM, and REM sleep stages
-  **Microarousal Detection** - Label microarousals during NREM sleep
-  **Real-time Hypnogram** - Vertical column visualization with color coding
-  **Multi-Channel Display** - EEG, EMG (high-pass filtered), and PSD analysis
-  **Fixed Scale Mode** - Consistent signal amplitude across all epochs
-  **Auto-Save** - Automatic saving every 10 epochs
-  **Organized Output** - CSV data and high-resolution hypnogram images
-  **Keyboard Shortcuts** - Fast, efficient scoring workflow
-  **Fully Configurable** - All parameters easily adjustable in config.py

---

##  Color Scheme

- **Black** → Wake
- **Blue** → NREM (Non-REM sleep)
- **Green** → REM (Rapid Eye Movement sleep)
- **Red** → Microarousal
- **Light Gray** → Unscored epochs

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- Anaconda or Miniconda (recommended)

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/manilbradai-cmd/sleep-scoring-interface.git
cd sleep-scoring-interface

# Create conda environment
conda create -n sleep-scoring python=3.9
conda activate sleep-scoring

# Install dependencies
conda install numpy pandas matplotlib scipy
conda install -c anaconda tk
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/manilbradai-cmd/sleep-scoring-interface.git
cd sleep-scoring-interface

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Quick Start

### 1. Prepare Your Data

Create a folder with your sleep data files in NumPy format:

```python
# Required files:
data/
├── all_eeg.npy    # EEG data (n_epochs × samples_per_epoch)
├── all_emg.npy    # EMG data (n_epochs × samples_per_epoch)
└── all_states.npy # Initial state codes (optional)
```

**Data format example:**
```python
# For 1000 epochs of 4 seconds at 1000 Hz:
all_eeg.npy   → shape: (1000, 4000)
all_emg.npy   → shape: (1000, 4000)
all_states.npy → shape: (1000,)  # Optional
```

### 2. Configure the Application

Edit `config.py` and update the data directory:

```python
OUT_DIR = Path(r"C:\path\to\your\data\folder")  # Update this!
```

### 3. Run the Application

```bash
python sleep_scorer.py
```

Or use the Windows batch file:
```bash
run_sleep_scorer.bat
```

---

##  Keyboard Shortcuts

### Navigation
- `←` / `→` — Previous/Next epoch
- `J` — Jump to specific epoch

### Sleep Stage Scoring
- `W` — Score as Wake
- `N` — Score as NREM
- `R` — Score as REM

### Microarousal Labeling (NREM only)
- `0` — Not a microarousal
- `1` — Microarousal

### Display Controls
- `+` / `=` — Zoom in (increase signal detail)
- `-` / `_` — Zoom out (decrease signal detail)

### Utilities
- `S` — Save all (CSV + hypnogram image)
- Click ` Stats` — Show statistics

---

## ⚙️ Configuration

All settings can be adjusted in `config.py`:

```python
# Signal parameters
FS_HZ = 1000              # Sampling frequency in Hz
EPOCH_LEN_S = 4.0         # Epoch length in seconds

# Display settings
CONTEXT_EPOCHS_AFTER = 4  # Context window (number of epochs)
HYPNO_WINDOW_EPOCHS = 100 # Hypnogram window size
EEG_MAD_MULT = 8.0        # EEG vertical scale
EMG_MAD_MULT = 10.0       # EMG vertical scale

# Auto-save
AUTOSAVE_INTERVAL = 10    # Save every N epochs

# Colors
COLORS = {
    'Wake': 'black',
    'NREM': 'blue',
    'REM': 'green',
    'MA': 'red',
    'Unscored': 'lightgray'
}
```

---

## 📂 Output Files

The application creates a `Hypno-score` folder in your data directory:

```
your_data_folder/
├── all_eeg.npy
├── all_emg.npy
├── all_states.npy
└── Hypno-score/                          ← Created by the application
    ├── sleep_stages_and_microarousals_manual.csv  ← Scored data
    └── hypnogram_full.png                         ← Hypnogram visualization
```

### CSV Format

```csv
epoch_index,state_code,state_name,label_ma
0,1,Wake,-1
1,2,NREM,0
2,2,NREM,1
3,3,REM,-1
```

**Field definitions:**
- `epoch_index`: Epoch number (0 to n-1)
- `state_code`: -1=Unscored, 1=Wake, 2=NREM, 3=REM
- `state_name`: Text version of state code
- `label_ma`: -1=unscored, 0=Not MA, 1=MA (NREM only)

---

## 🔬 Key Features Explained

### Fixed Scale Mode
- Calculates one global scale at startup based on all data
- Maintains consistent Y-axis limits across all epochs
- Allows direct amplitude comparison between epochs
- Manual fine-tuning with `+/-` keys

### Robust Signal Processing
- **MAD-based scaling** - Resistant to outliers and artifacts
- **Display clipping** - Prevents extreme values from crushing the trace
- **EMG high-pass filtering** - 100 Hz cutoff for muscle activity
- **PSD analysis** - Normalized power spectral density for frequency inspection

### Auto-Save & Session Resume
- Automatically saves progress every 10 epochs
- Loads existing work on startup
- Never lose your scoring progress
- Manual save available anytime with `S` key

---

##  Scientific Methods

This application uses rigorous signal processing methods:

- **EEG Processing**: Raw signal with MAD-based robust scaling
- **EMG Filtering**: 4th-order Butterworth high-pass filter (100 Hz, zero-phase)
- **PSD Analysis**: Welch's method with Hann window (1024 samples, 50% overlap)
- **Robust Statistics**: Median Absolute Deviation (MAD) for outlier resistance
- **Display Clipping**: Winsorization to prevent artifact compression

**For detailed scientific documentation**, see **[METHODS.md](METHODS.md)**

---

##  Testing with Sample Data

Generate test data to try the interface:

```bash
python create_sample_data.py
```

This creates a `sample_data` folder with synthetic sleep data. Update `config.py` to point to this folder.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'config'" | Ensure `config.py` is in the same directory as `sleep_scorer.py` |
| "FileNotFoundError: all_eeg.npy" | Check that .npy files are in the directory specified in `config.py` |
| Script closes immediately | Run from command line/Anaconda Prompt, not by double-clicking |
| EEG signals look flat | Press `+` key several times to zoom in |
| "No module named 'tkinter'" | Run: `conda install -c anaconda tk` |

---

## 📖 Documentation

For detailed guides, see:
- [Installation Guide](docs/INSTALLATION.md)
- [User Manual](docs/USER_GUIDE.md)
- [Configuration Options](docs/CONFIGURATION.md)

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Citation

If you use this software in your research, please cite:

```bibtex
@software{sleep_scoring_interface,
  author = {Bradai, Manil},
  title = {Sleep Stage Scoring Interface},
  year = {2025},
  url = {https://github.com/manilbradai-cmd/sleep-scoring-interface}
}
```

---

## 📧 Contact

- **Author**: Manil Bradai
- **GitHub**: [@manilbradai-cmd](https://github.com/manilbradai-cmd)

---

##  Acknowledgments

- Built with Python, NumPy, Pandas, Matplotlib, and SciPy
- Inspired by sleep research and polysomnography analysis needs
- Thanks to the sleep research community for feedback

---

**Note**: This software is for research purposes. Clinical applications should be validated according to appropriate standards and regulations.
