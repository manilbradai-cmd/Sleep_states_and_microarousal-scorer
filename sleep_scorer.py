#!/usr/bin/env python3
"""
Sleep Stage Scoring Interface
A Python-based GUI for manual sleep stage scoring and microarousal detection.

Author: Manil Bradai
GitHub: https://github.com/manilbradai-cmd/sleep-scoring-interface
License: MIT

Instructions:
1. Edit config.py to set your data directory and preferences
2. Run this script: python sleep_scorer.py
3. Use keyboard shortcuts to score sleep stages
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import welch, butter, filtfilt

# Import configuration
try:
    from config import *
except ImportError:
    print("ERROR: config.py not found!")
    print("Please ensure config.py is in the same directory as this script.")
    exit(1)

# Calculate derived constants
SAMPLES_PER_EPOCH = int(FS_HZ * EPOCH_LEN_S)

# Create output directory
HYPNO_DIR = OUT_DIR / OUTPUT_FOLDER_NAME
HYPNO_DIR.mkdir(exist_ok=True)

# Output files
OUT_CSV = HYPNO_DIR / "sleep_stages_and_microarousals_manual.csv"
HYPNO_IMG = HYPNO_DIR / "hypnogram_full.png"

# =========================
# DISABLE MATPLOTLIB NAVIGATION
# =========================
mpl.rcParams["keymap.back"] = []
mpl.rcParams["keymap.forward"] = []
mpl.rcParams["keymap.home"] = []
mpl.rcParams["keymap.pan"] = []
mpl.rcParams["keymap.zoom"] = []
mpl.rcParams["keymap.save"] = []
mpl.rcParams["keymap.fullscreen"] = []

# =========================
# HELPER FUNCTIONS
# =========================
def state_name(code: int) -> str:
    """Convert state code to name."""
    return {1: "Wake", 2: "NREM", 3: "REM", -1: "Unscored"}.get(int(code), f"Other({int(code)})")

def emg_highpass(x, fs):
    """
    High-pass filter EMG signal using Butterworth IIR filter.
    
    Method: Zero-phase filtering (forward-backward pass)
    Filter type: Butterworth (maximally flat passband)
    Implementation: scipy.signal.filtfilt
    
    Parameters:
    -----------
    x : array_like
        Input EMG signal
    fs : float
        Sampling frequency (Hz)
    
    Returns:
    --------
    y : ndarray
        Filtered EMG signal
        
    Notes:
    ------
    - Effective filter order is 2 × EMG_FILTER_ORDER due to bidirectional filtering
    - Zero phase distortion (no time delay)
    - Removes low-frequency movement artifacts (<100 Hz)
    - Preserves muscle activity (>100 Hz)
    """
    b, a = butter(EMG_FILTER_ORDER, EMG_HIGHPASS_CUTOFF_HZ / (fs / 2), btype="highpass")
    return filtfilt(b, a, x)

def normalized_psd(x, fs, fmax, nperseg):
    """
    Calculate normalized power spectral density using Welch's method.
    
    Method: Welch's averaged periodogram
    Window: Hann window (default in scipy)
    Overlap: 50% (default in scipy)
    
    Parameters:
    -----------
    x : array_like
        Input signal
    fs : float
        Sampling frequency (Hz)
    fmax : float
        Maximum frequency to return (Hz)
    nperseg : int
        Length of each segment for FFT
        
    Returns:
    --------
    f : ndarray
        Frequency bins (Hz)
    pxx : ndarray
        Normalized power spectral density (0-1 scale)
        
    Notes:
    ------
    - Normalization: PSD(f) / max(PSD(f))
    - Frequency resolution: fs / nperseg
    - Smoothing: Averaging over multiple segments
    - Hann window reduces spectral leakage
    
    References:
    -----------
    Welch, P. (1967). IEEE Trans. Audio Electroacoustics, 15(2), 70-73.
    """
    f, pxx = welch(x, fs=fs, nperseg=min(nperseg, x.size))
    keep = f <= fmax
    f = f[keep]
    pxx = pxx[keep]
    if pxx.size:
        pxx = pxx / (np.max(pxx) + 1e-12)
    return f, pxx

def mad(x):
    """
    Calculate Median Absolute Deviation (MAD).
    
    MAD is a robust measure of statistical dispersion.
    
    Formula:
    --------
    MAD = median(|x_i - median(x)|)
    
    Parameters:
    -----------
    x : array_like
        Input data
        
    Returns:
    --------
    mad : float
        Median absolute deviation
        
    Notes:
    ------
    - Robust to outliers (50% breakdown point)
    - Does not assume Gaussian distribution
    - Conversion to SD: σ_robust = 1.4826 × MAD
    
    References:
    -----------
    Rousseeuw & Croux (1993). JASA, 88(424), 1273-1283.
    Leys et al. (2013). J. Exp. Soc. Psychol., 49(4), 764-766.
    """
    med = np.median(x)
    return np.median(np.abs(x - med))

def robust_limits_mad(x, mult=8.0, min_ylim=1.0):
    """
    Calculate symmetric y-axis limits using MAD-based robust statistics.
    
    Method: Median Absolute Deviation (MAD)
    Conversion: σ_robust = 1.4826 × MAD (for Gaussian equivalence)
    
    Parameters:
    -----------
    x : array_like
        Input signal
    mult : float
        Multiplier for y-limits (default: 8.0)
    min_ylim : float
        Minimum y-limit (default: 1.0)
        
    Returns:
    --------
    (y_min, y_max) : tuple
        Symmetric y-axis limits
        
    Notes:
    ------
    - Constant 1.4826 is consistency factor for Gaussian distribution
    - Larger mult → more vertical range (signals appear smaller)
    - Resistant to outliers and artifacts
    - Does not assume normal distribution
    """
    m = mad(x)
    sigma = 1.4826 * m
    ylim = float(mult) * float(sigma)
    if not np.isfinite(ylim) or ylim <= 0:
        ylim = float(min_ylim)
    return -ylim, ylim

def clip_for_display(x, mult=10.0):
    """
    Clip signal for display using MAD-based thresholds (winsorization).
    
    Purpose: Prevent extreme artifacts from compressing visible signal
    Application: Display only (raw data unchanged)
    
    Parameters:
    -----------
    x : array_like
        Input signal
    mult : float
        Clipping threshold multiplier (default: 10.0)
        
    Returns:
    --------
    x_clipped : ndarray
        Clipped signal for display
        
    Notes:
    ------
    - Clips at ±(mult × 1.4826 × MAD)
    - Applied ONLY for visualization
    - Does NOT modify raw data
    - Prevents single artifacts from crushing trace
    - More aggressive than robust_limits_mad
    """
    m = mad(x)
    sigma = 1.4826 * m
    if not np.isfinite(sigma) or sigma <= 0:
        return x
    lim = mult * sigma
    return np.clip(x, -lim, lim)

# =========================
# LOAD DATA
# =========================
print("=" * 60)
print("Sleep Stage Scoring Interface")
print("=" * 60)
print("📂 Loading data files...")

try:
    eeg = np.load(OUT_DIR / "all_eeg.npy")
    emg = np.load(OUT_DIR / "all_emg.npy")
    states = np.load(OUT_DIR / "all_states.npy").astype(int)
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Required data file not found!")
    print(f"   {e}")
    print(f"\n   Please ensure the following files are in {OUT_DIR}:")
    print("   - all_eeg.npy")
    print("   - all_emg.npy")
    print("   - all_states.npy")
    exit(1)

assert eeg.shape == emg.shape, "EEG and EMG must have the same shape"
n_epochs, n_samp = eeg.shape
assert n_samp == SAMPLES_PER_EPOCH, f"Expected {SAMPLES_PER_EPOCH} samples per epoch, got {n_samp}"
assert states.shape[0] == n_epochs, "States array must match number of epochs"

epoch_list = np.arange(n_epochs, dtype=int)
n_items = len(epoch_list)

print(f"✅ Loaded {n_epochs} epochs ({n_epochs * EPOCH_LEN_S / 60:.1f} minutes)")

# Initialize or load labels
if OUT_CSV.exists():
    try:
        labels = pd.read_csv(OUT_CSV)
        n_scored = (labels['state_code'] != -1).sum()
        print(f"✅ Loaded existing labels from {OUT_CSV.name}")
        print(f"   Progress: {n_scored}/{n_epochs} epochs scored ({n_scored/n_epochs*100:.1f}%)")
    except Exception as e:
        print(f"⚠️  Could not load existing labels: {e}")
        labels = pd.DataFrame({
            "epoch_index": epoch_list,
            "state_code": -1,
            "state_name": "Unscored",
            "label_ma": -1
        })
else:
    labels = pd.DataFrame({
        "epoch_index": epoch_list,
        "state_code": -1,
        "state_name": "Unscored",
        "label_ma": -1
    })
    print("📝 Starting new scoring session")

epochs_changed_since_save = 0

# =========================
# CALCULATE GLOBAL SCALE
# =========================
print("📊 Calculating global EEG/EMG scales...")

sample_indices = np.arange(0, n_epochs, max(1, n_epochs // 100))
eeg_sample = eeg[sample_indices].flatten()
emg_sample = emg[sample_indices].flatten()

GLOBAL_EEG_YLIM = robust_limits_mad(eeg_sample, mult=EEG_MAD_MULT, min_ylim=50.0)
GLOBAL_EMG_YLIM = robust_limits_mad(emg_highpass(emg_sample, FS_HZ), mult=EMG_MAD_MULT, min_ylim=10.0)

print(f"✅ Global EEG range: {GLOBAL_EEG_YLIM[0]:.1f} to {GLOBAL_EEG_YLIM[1]:.1f}")
print(f"✅ Global EMG range: {GLOBAL_EMG_YLIM[0]:.1f} to {GLOBAL_EMG_YLIM[1]:.1f}")

# =========================
# SAVE FUNCTIONS
# =========================
def save_labels():
    """Save labels to CSV."""
    global epochs_changed_since_save
    labels.to_csv(OUT_CSV, index=False)
    epochs_changed_since_save = 0
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_var.set(f"✅ Saved at {timestamp}: {OUT_CSV.name}")
    return True

def save_full_hypnogram():
    """Save complete hypnogram image."""
    try:
        fig_hypno = Figure(figsize=(20, 4), dpi=150)
        ax = fig_hypno.add_subplot(111)
        
        for ep in range(n_epochs):
            state_code = int(labels.loc[labels["epoch_index"] == ep, "state_code"].iloc[0])
            ma_label = int(labels.loc[labels["epoch_index"] == ep, "label_ma"].iloc[0])
            
            if ma_label == 1:
                color = COLORS['MA']
            elif state_code == 1:
                color = COLORS['Wake']
            elif state_code == 2:
                color = COLORS['NREM']
            elif state_code == 3:
                color = COLORS['REM']
            else:
                color = COLORS['Unscored']
            
            ax.bar(ep, 1.0, bottom=0, width=1.0, color=color, edgecolor='none', alpha=0.9)
        
        ax.set_xlim(-0.5, n_epochs - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel(f'Epoch Index (0 to {n_epochs-1}, {EPOCH_LEN_S}s per epoch)', fontsize=12)
        ax.set_ylabel('')
        ax.set_title('Complete Hypnogram - Sleep Stage Scoring', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['Wake'], label='Wake', edgecolor='black', linewidth=0.5),
            Patch(facecolor=COLORS['NREM'], label='NREM', edgecolor='black', linewidth=0.5),
            Patch(facecolor=COLORS['REM'], label='REM', edgecolor='black', linewidth=0.5),
            Patch(facecolor=COLORS['MA'], label='Microarousal', edgecolor='black', linewidth=0.5),
            Patch(facecolor=COLORS['Unscored'], label='Unscored', edgecolor='black', linewidth=0.5)
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                 ncol=5, fontsize=11, framealpha=0.95, edgecolor='black')
        
        n_wake = (labels['state_code'] == 1).sum()
        n_nrem = (labels['state_code'] == 2).sum()
        n_rem = (labels['state_code'] == 3).sum()
        n_ma = (labels['label_ma'] == 1).sum()
        n_unscored = (labels['state_code'] == -1).sum()
        
        wake_min = (n_wake * EPOCH_LEN_S) / 60
        nrem_min = (n_nrem * EPOCH_LEN_S) / 60
        rem_min = (n_rem * EPOCH_LEN_S) / 60
        
        stats_text = (f"Wake: {n_wake} epochs ({wake_min:.1f}min) | "
                     f"NREM: {n_nrem} epochs ({nrem_min:.1f}min) | "
                     f"REM: {n_rem} epochs ({rem_min:.1f}min) | "
                     f"MA: {n_ma} epochs | "
                     f"Unscored: {n_unscored} epochs")
        
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
        
        fig_hypno.tight_layout()
        fig_hypno.savefig(HYPNO_IMG, dpi=150, bbox_inches='tight')
        print(f"✅ Saved hypnogram: {HYPNO_IMG}")
        
        import matplotlib.pyplot as plt
        plt.close(fig_hypno)
        return True
    except Exception as e:
        print(f"❌ Error saving hypnogram: {e}")
        return False

def save_all():
    """Save both CSV and hypnogram."""
    csv_ok = save_labels()
    img_ok = save_full_hypnogram()
    
    if csv_ok and img_ok:
        messagebox.showinfo("Save Complete", 
                           f"✅ Saved successfully!\n\nCSV: {OUT_CSV.name}\nImage: {HYPNO_IMG.name}\n\nLocation: {HYPNO_DIR}")
    elif csv_ok:
        messagebox.showwarning("Partial Save", f"✅ CSV saved\n❌ Image failed\n\nLocation: {HYPNO_DIR}")
    else:
        messagebox.showerror("Save Failed", "❌ Failed to save files")

def auto_save_check():
    """Check if auto-save is needed."""
    global epochs_changed_since_save
    if epochs_changed_since_save >= AUTOSAVE_INTERVAL:
        save_labels()

# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Sleep Stage Scoring Interface")

fig = Figure(figsize=(14, 9), dpi=100)
ax_eeg = fig.add_subplot(411)
ax_emg = fig.add_subplot(412, sharex=ax_eeg)
ax_psd = fig.add_subplot(413)
ax_hypno = fig.add_subplot(414)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

controls = tk.Frame(root)
controls.pack(fill=tk.X)

status_var = tk.StringVar(value="")
status = tk.Label(root, textvariable=status_var, anchor="w", font=("Arial", 9))
status.pack(fill=tk.X)

idx = 0
_last_nav_ms = 0

def get_window(epoch_idx: int):
    """Get data window for display."""
    start_ep = epoch_idx
    end_ep = min(n_epochs - 1, epoch_idx + CONTEXT_EPOCHS_AFTER)
    eeg_w = eeg[start_ep:end_ep + 1].reshape(-1)
    emg_w = emg[start_ep:end_ep + 1].reshape(-1)
    t = np.arange(eeg_w.size) / FS_HZ
    cur_start = 0.0
    cur_end = EPOCH_LEN_S
    return t, eeg_w, emg_w, cur_start, cur_end, start_ep, end_ep

def draw_hypnogram():
    """Draw hypnogram."""
    ax_hypno.clear()
    center_ep = int(epoch_list[idx])
    half_window = HYPNO_WINDOW_EPOCHS // 2
    start_display = max(0, center_ep - half_window)
    end_display = min(n_epochs - 1, center_ep + half_window)
    display_epochs = np.arange(start_display, end_display + 1)
    
    for ep in display_epochs:
        state_code = int(labels.loc[labels["epoch_index"] == ep, "state_code"].iloc[0])
        ma_label = int(labels.loc[labels["epoch_index"] == ep, "label_ma"].iloc[0])
        
        if ma_label == 1:
            color = COLORS['MA']
        elif state_code == 1:
            color = COLORS['Wake']
        elif state_code == 2:
            color = COLORS['NREM']
        elif state_code == 3:
            color = COLORS['REM']
        else:
            color = COLORS['Unscored']
        
        ax_hypno.bar(ep, 1.0, bottom=0, width=1.0, color=color, edgecolor='none', alpha=0.9)
    
    current_ep = int(epoch_list[idx])
    ax_hypno.bar(current_ep, 1.0, bottom=0, width=1.0, fill=False, edgecolor='orange', linewidth=3, alpha=1.0)
    
    ax_hypno.set_xlim(start_display - 0.5, end_display + 0.5)
    ax_hypno.set_ylim(0, 1)
    ax_hypno.set_yticks([])
    ax_hypno.set_xlabel(f'Epoch Index (showing {start_display}-{end_display})', fontsize=10)
    ax_hypno.set_ylabel('')
    ax_hypno.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['Wake'], label='Wake'),
        Patch(facecolor=COLORS['NREM'], label='NREM'),
        Patch(facecolor=COLORS['REM'], label='REM'),
        Patch(facecolor=COLORS['MA'], label='MA'),
        Patch(facecolor=COLORS['Unscored'], label='Unscored')
    ]
    ax_hypno.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=8, framealpha=0.9)

def redraw():
    """Redraw all plots."""
    global GLOBAL_EEG_YLIM, GLOBAL_EMG_YLIM
    
    ep = int(epoch_list[idx])
    st = int(labels.loc[labels["epoch_index"] == ep, "state_code"].iloc[0])
    st_txt = state_name(st)

    t, eeg_w, emg_w, cur_start, cur_end, start_ep, end_ep = get_window(ep)

    ax_eeg.clear()
    ax_emg.clear()
    ax_psd.clear()

    # EEG plot
    eeg_disp = clip_for_display(eeg_w, mult=CLIP_MULT_EEG)
    ax_eeg.plot(t, eeg_disp, linewidth=0.8, color='blue')
    ax_eeg.set_ylabel("EEG (μV)", fontsize=10)
    ax_eeg.set_title(f"Epoch {ep} ({st_txt}) | {idx+1}/{n_items} | window {start_ep}..{end_ep} ({(end_ep-start_ep+1)*EPOCH_LEN_S:.0f}s)")
    ax_eeg.set_ylim(GLOBAL_EEG_YLIM[0], GLOBAL_EEG_YLIM[1])

    # EMG plot
    emg_hp = emg_highpass(emg_w, FS_HZ)
    emg_disp = clip_for_display(emg_hp, mult=CLIP_MULT_EMG)
    ax_emg.plot(t, emg_disp, linewidth=0.7, color='darkred')
    ax_emg.set_ylabel("EMG (HP>100 Hz)", fontsize=10)
    ax_emg.set_xlabel("Time (s)", fontsize=10)
    ax_emg.set_ylim(GLOBAL_EMG_YLIM[0], GLOBAL_EMG_YLIM[1])

    # Highlight current epoch
    ax_eeg.axvspan(cur_start, cur_end, alpha=0.2, color='yellow')
    ax_emg.axvspan(cur_start, cur_end, alpha=0.2, color='yellow')

    # Epoch boundaries
    nwin = end_ep - start_ep + 1
    for k in range(nwin + 1):
        x = k * EPOCH_LEN_S
        ax_eeg.axvline(x, linewidth=0.8, alpha=0.25, color='gray')
        ax_emg.axvline(x, linewidth=0.8, alpha=0.25, color='gray')

    # PSD
    f_w, p_w = normalized_psd(eeg_w, FS_HZ, PSD_FMAX, N_PER_SEG)
    f_t, p_t = normalized_psd(eeg[ep], FS_HZ, PSD_FMAX, N_PER_SEG)
    ax_psd.plot(f_w, p_w, linewidth=1.0, label="entire signal (context)", color='gray')
    ax_psd.plot(f_t, p_t, linewidth=1.5, label="current target (4 s)", color='blue')
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_psd.set_ylabel("Normalized Power", fontsize=10)
    ax_psd.legend(loc="upper right", fontsize=9)

    # Lock view
    if t.size:
        ax_eeg.set_xlim(0, float(t[-1]))
        ax_emg.set_xlim(0, float(t[-1]))
    ax_eeg.set_autoscale_on(False)
    ax_emg.set_autoscale_on(False)
    
    draw_hypnogram()

    # Status
    lbl = int(labels.loc[labels["epoch_index"] == ep, "label_ma"].iloc[0])
    lbl_txt = "UNSCORED" if lbl == -1 else ("Not MA (0)" if lbl == 0 else "MA (1)")
    n_scored = (labels['state_code'] != -1).sum()
    progress_pct = (n_scored / n_epochs) * 100
    epoch_std = float(np.std(eeg[ep]))
    flat_flag = " ⚠ EEG near-flat" if epoch_std < 1.0 else ""

    status_var.set(f"Epoch {ep} ({st_txt}) | MA: {lbl_txt} | Progress: {n_scored}/{n_epochs} ({progress_pct:.1f}%) | "
                   f"W=Wake, N=NREM, R=REM | 0/1=MA | ←/→ nav | +/- scale | s=save{flat_flag}")

    fig.tight_layout()
    canvas.draw()

def set_sleep_stage(stage_code: int):
    """Set sleep stage."""
    global epochs_changed_since_save
    ep = int(epoch_list[idx])
    labels.loc[labels["epoch_index"] == ep, "state_code"] = int(stage_code)
    labels.loc[labels["epoch_index"] == ep, "state_name"] = state_name(stage_code)
    epochs_changed_since_save += 1
    auto_save_check()
    redraw()

def set_ma_label(val: int):
    """Set microarousal label."""
    global epochs_changed_since_save
    ep = int(epoch_list[idx])
    st = int(labels.loc[labels["epoch_index"] == ep, "state_code"].iloc[0])
    if st != 2:
        status_var.set(f"Epoch {ep} is {state_name(st)} – MA labels allowed only during NREM (state=2).")
        return
    labels.loc[labels["epoch_index"] == ep, "label_ma"] = int(val)
    epochs_changed_since_save += 1
    auto_save_check()
    redraw()

def next_item():
    global idx
    if idx < n_items - 1:
        idx += 1
        redraw()

def prev_item():
    global idx
    if idx > 0:
        idx -= 1
        redraw()

def jump_to_epoch():
    """Jump to specific epoch."""
    dialog = tk.Toplevel(root)
    dialog.title("Jump to Epoch")
    dialog.geometry("300x100")
    tk.Label(dialog, text=f"Enter epoch (0-{n_epochs-1}):").pack(pady=10)
    entry = tk.Entry(dialog)
    entry.pack(pady=5)
    entry.focus()
    
    def do_jump():
        try:
            target = int(entry.get())
            if 0 <= target < n_epochs:
                global idx
                idx = target
                redraw()
                dialog.destroy()
            else:
                messagebox.showerror("Error", f"Epoch must be between 0 and {n_epochs-1}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    tk.Button(dialog, text="Jump", command=do_jump).pack(pady=5)
    entry.bind("<Return>", lambda e: do_jump())

def show_statistics():
    """Show statistics."""
    n_wake = (labels['state_code'] == 1).sum()
    n_nrem = (labels['state_code'] == 2).sum()
    n_rem = (labels['state_code'] == 3).sum()
    n_ma = (labels['label_ma'] == 1).sum()
    n_unscored = (labels['state_code'] == -1).sum()
    total_time_min = (n_epochs * EPOCH_LEN_S) / 60
    wake_time_min = (n_wake * EPOCH_LEN_S) / 60
    nrem_time_min = (n_nrem * EPOCH_LEN_S) / 60
    rem_time_min = (n_rem * EPOCH_LEN_S) / 60
    
    stats_msg = f"""📊 SCORING STATISTICS
    
Total Epochs: {n_epochs} ({total_time_min:.1f} minutes)

Sleep Stages:
• Wake:      {n_wake:4d} epochs ({wake_time_min:.1f} min)
• NREM:      {n_nrem:4d} epochs ({nrem_time_min:.1f} min)
• REM:       {n_rem:4d} epochs ({rem_time_min:.1f} min)
• Unscored:  {n_unscored:4d} epochs

Microarousals: {n_ma} epochs

Progress: {((n_epochs - n_unscored) / n_epochs * 100):.1f}% complete"""
    
    messagebox.showinfo("Statistics", stats_msg)

def _debounced(callable_fn):
    global _last_nav_ms
    now = int(root.tk.call("clock", "milliseconds"))
    if now - _last_nav_ms < NAV_DEBOUNCE_MS:
        return
    _last_nav_ms = now
    callable_fn()

def nav_prev_event(event=None):
    _debounced(prev_item)

def nav_next_event(event=None):
    _debounced(next_item)

def _ignore_event(event):
    return

for ev in ["scroll_event", "button_press_event", "button_release_event", "motion_notify_event", "key_press_event"]:
    canvas.mpl_connect(ev, _ignore_event)

# Controls
tk.Button(controls, text="← Prev", command=prev_item, font=("Arial", 9)).pack(side=tk.LEFT, padx=2, pady=5)
tk.Button(controls, text="Next →", command=next_item, font=("Arial", 9)).pack(side=tk.LEFT, padx=2, pady=5)
tk.Button(controls, text="Jump (J)", command=jump_to_epoch, font=("Arial", 9)).pack(side=tk.LEFT, padx=5, pady=5)

tk.Label(controls, text="|", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="W Wake", command=lambda: set_sleep_stage(1), 
          bg='black', fg='white', width=8, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2, pady=5)
tk.Button(controls, text="N NREM", command=lambda: set_sleep_stage(2), 
          bg='blue', fg='white', width=8, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2, pady=5)
tk.Button(controls, text="R REM", command=lambda: set_sleep_stage(3), 
          bg='green', fg='white', width=8, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2, pady=5)

tk.Label(controls, text="|", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="0 Not MA", command=lambda: set_ma_label(0), width=8, font=("Arial", 9)).pack(side=tk.LEFT, padx=2, pady=5)
tk.Button(controls, text="1 MA", command=lambda: set_ma_label(1), 
          bg='red', fg='white', width=8, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2, pady=5)

tk.Label(controls, text="|", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
tk.Button(controls, text="📊 Stats", command=show_statistics, font=("Arial", 9)).pack(side=tk.LEFT, padx=2, pady=5)
tk.Button(controls, text="💾 Save All (S)", command=save_all, 
          bg='lightgreen', width=12, font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=5, pady=5)

# Key bindings
root.bind("<Left>", nav_prev_event)
root.bind("<Right>", nav_next_event)
root.bind("w", lambda e: set_sleep_stage(1))
root.bind("W", lambda e: set_sleep_stage(1))
root.bind("n", lambda e: set_sleep_stage(2))
root.bind("N", lambda e: set_sleep_stage(2))
root.bind("r", lambda e: set_sleep_stage(3))
root.bind("R", lambda e: set_sleep_stage(3))
root.bind("0", lambda e: set_ma_label(0))
root.bind("1", lambda e: set_ma_label(1))
root.bind("s", lambda e: save_all())
root.bind("S", lambda e: save_all())
root.bind("j", lambda e: jump_to_epoch())
root.bind("J", lambda e: jump_to_epoch())

def zoom_in_eeg(event=None):
    global GLOBAL_EEG_YLIM
    current_range = GLOBAL_EEG_YLIM[1] - GLOBAL_EEG_YLIM[0]
    new_range = current_range * 0.7
    GLOBAL_EEG_YLIM = (-new_range/2, new_range/2)
    redraw()

def zoom_out_eeg(event=None):
    global GLOBAL_EEG_YLIM
    current_range = GLOBAL_EEG_YLIM[1] - GLOBAL_EEG_YLIM[0]
    new_range = current_range * 1.3
    GLOBAL_EEG_YLIM = (-new_range/2, new_range/2)
    redraw()

root.bind("+", zoom_in_eeg)
root.bind("=", zoom_in_eeg)
root.bind("-", zoom_out_eeg)
root.bind("_", zoom_out_eeg)

# Display startup info
print("=" * 60)
print(f"🎯 Sleep Scoring Interface Ready")
print(f"📁 Output directory: {HYPNO_DIR}")
print(f"📊 Total epochs: {n_epochs}")
print(f"⌨️  Keyboard shortcuts:")
print(f"   W=Wake, N=NREM, R=REM")
print(f"   0=Not MA, 1=MA (NREM only)")
print(f"   ←/→=Navigate, J=Jump, S=Save")
print(f"   +/- = Zoom EEG scale in/out")
print(f"💾 Auto-save every {AUTOSAVE_INTERVAL} epochs")
print(f"📏 Fixed scale mode enabled")
print("=" * 60)

# Start
redraw()
root.mainloop()

# Save on exit
print("\n🔄 Saving final data...")
save_all()
print("✅ Session complete!")
