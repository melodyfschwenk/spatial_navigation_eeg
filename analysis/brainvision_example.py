"""
BrainVision Data Processing Example for Spatial Navigation EEG
=============================================================

This script demonstrates how to:
1. Load BrainVision EEG data
2. Extract and visualize event codes
3. Map event codes to experimental conditions
4. Extract epochs for each condition
5. Prepare data for time-frequency and coherence analysis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne

# Add project to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.spatial_nav_toolkit import SpatialNavDataset, EEGPreprocessing, Visualization

# Path to BrainVision data
data_dir = r"c:\Users\melod\OneDrive\Desktop\spatial_navigation_eeg\data"
vhdr_file = os.path.join(data_dir, "sub-01_task-navigation.vhdr")

# Initialize dataset and preprocessing objects
dataset = SpatialNavDataset(data_dir)
eeg = EEGPreprocessing(dataset)

# Example 1: Basic workflow to explore BrainVision data
print("=== Example 1: Exploring BrainVision Data ===")
# Load data
raw = eeg.load_brainvision_data(vhdr_file)

# Check data quality
quality_metrics = eeg.check_data_quality(raw=raw)

# Inspect events
events, event_id = eeg.inspect_events(raw=raw)

# Create template condition mapping
eeg.create_condition_mapping_template(event_id=event_id)

# Example 2: Processing with a defined condition mapping
print("\n=== Example 2: Processing with Condition Mapping ===")
# Define your condition mapping based on the template
condition_mapping = {
    'S  1': 'fixation',
    'S 11': 'allocentric_easy',
    'S 12': 'allocentric_hard',
    'S 21': 'egocentric_easy',
    'S 22': 'egocentric_hard',
    'S 30': 'control'
}

# Extract epochs by condition
epochs_by_condition = eeg.extract_condition_epochs(
    condition_mapping=condition_mapping,
    raw=raw,
    tmin=-0.2,  # 200ms before trigger
    tmax=1.0,   # 1000ms after trigger
    baseline=(-0.2, 0),  # Baseline correction period
    reject={'eeg': 100e-6}  # Reject epochs with ±100 µV
)

# Example 3: Time-frequency analysis per condition
print("\n=== Example 3: Time-Frequency Analysis ===")
# Only proceed if we have epochs
if epochs_by_condition:
    for condition, epochs in epochs_by_condition.items():
        print(f"\nAnalyzing condition: {condition}")
        
        # Calculate power spectral density
        freqs = np.arange(4, 40)  # Frequencies from 4-40 Hz
        n_cycles = freqs / 2.  # Different number of cycles per frequency
        
        print(f"Computing time-frequency for {len(epochs)} epochs...")
        power = mne.time_frequency.tfr_morlet(
            epochs, 
            freqs=freqs, 
            n_cycles=n_cycles, 
            use_fft=True, 
            return_itc=False, 
            average=True
        )
        
        print(f"Time-frequency shape: {power.data.shape}")
        print(f"  Channels × Frequencies × Times: {power.data.shape[0]} × {power.data.shape[1]} × {power.data.shape[2]}")
        
        # Example: Extract power in alpha band (8-12 Hz)
        alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
        alpha_power = np.mean(power.data[:, alpha_idx, :], axis=1)
        
        print(f"Average alpha power: {np.mean(alpha_power):.6f} µV²")

# Example 4: Calculating coherence between channels
print("\n=== Example 4: Coherence Analysis ===")
# Only proceed if we have epochs
if epochs_by_condition:
    # Import from the standalone mne-connectivity package
    from mne_connectivity import spectral_connectivity
    
    # Choose a condition to analyze
    condition = list(epochs_by_condition.keys())[0]
    epochs = epochs_by_condition[condition]
    
    print(f"\nCalculating coherence for condition: {condition}")
    
    # Pick some channels of interest
    channels = ['Fz', 'Cz', 'Pz', 'C3', 'C4']
    if all(ch in epochs.ch_names for ch in channels):
        # Select only the channels of interest
        epochs_subset = epochs.copy().pick_channels(channels)
        
        # Calculate coherence
        print(f"Computing coherence between channels: {', '.join(channels)}")
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs_subset, 
            method='coh',  # Coherence
            mode='multitaper',
            sfreq=epochs.info['sfreq'],
            fmin=8,
            fmax=12,  # Alpha band
            faverage=True
        )
        
        print(f"Coherence matrix shape: {con.shape}")
        
        # Print coherence values
        print("\nCoherence values (alpha band):")
        n_channels = len(channels)
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                conn_idx = (i * n_channels + j) - (i * (i + 1) // 2) - i - 1
                print(f"  {channels[i]} - {channels[j]}: {con[conn_idx, 0, 0]:.4f}")
    else:
        missing = [ch for ch in channels if ch not in epochs.ch_names]
        print(f"Cannot calculate coherence. Missing channels: {missing}")

print("\nAnalysis complete!")
