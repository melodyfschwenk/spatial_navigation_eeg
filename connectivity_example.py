"""
Example of using mne-connectivity as a standalone module
========================================================

This script demonstrates how to use the mne-connectivity package
for computing various connectivity measures between EEG channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_connectivity import spectral_connectivity, seed_target_indices

print("=== MNE-Connectivity Standalone Example ===")

# Create simulated EEG data (3 channels, 1000 time points)
data = np.random.randn(3, 1000)  
# Make channel 0 and 1 correlated
data[1] = 0.8 * data[0] + 0.2 * data[1]  

# Create MNE objects
sfreq = 100  # 100 Hz sampling rate
ch_names = ['Fz', 'Cz', 'Pz']
ch_types = ['eeg', 'eeg', 'eeg']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create raw data and epoch it
raw = mne.io.RawArray(data, info)
events = np.array([[100, 0, 1], [500, 0, 2]])  # Two simulated events
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.3, baseline=(None, 0))
print(f"Created {len(epochs)} epochs with shape {epochs.get_data().shape}")

# Define which connections to analyze (all pairs in this case)
indices = seed_target_indices(np.arange(len(ch_names)))
print(f"Computing connectivity between {indices.shape[0]} channel pairs")

# Compute connectivity - here we'll try different methods
methods = ['coh', 'plv', 'pli', 'wpli']
results = {}

for method in methods:
    print(f"\nComputing {method}...")
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        epochs,
        method=method,
        indices=indices,
        sfreq=sfreq,
        fmin=8,
        fmax=12,  # Alpha band
        faverage=True  # Average across frequencies
    )
    results[method] = con

# Plot the results
plt.figure(figsize=(12, 8))
for i, method in enumerate(methods):
    plt.subplot(2, 2, i+1)
    
    # Reshape connectivity matrix for plotting
    con_matrix = np.zeros((len(ch_names), len(ch_names)))
    for j, (seed, target) in enumerate(indices):
        con_matrix[seed, target] = results[method][j, 0, 0]
        con_matrix[target, seed] = results[method][j, 0, 0]  # Make symmetric
    
    # Plot as heatmap
    im = plt.imshow(con_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.title(f"{method.upper()} Connectivity")
    plt.xticks(range(len(ch_names)), ch_names)
    plt.yticks(range(len(ch_names)), ch_names)

plt.tight_layout()
plt.savefig('connectivity_methods.png')
print(f"\nSaved connectivity plot to 'connectivity_methods.png'")
print("\nDone!")

# Print instructions on how to use mne_connectivity
print("\n=== How to use mne_connectivity in your code ===")
print("1. Install: pip install mne-connectivity")
print("2. Import: from mne_connectivity import spectral_connectivity")
print("3. Compute connectivity with various methods:")
print("   - 'coh': Coherence")
print("   - 'plv': Phase Locking Value")
print("   - 'pli': Phase Lag Index")
print("   - 'wpli': Weighted Phase Lag Index")
print("   - 'dpli': Directed Phase Lag Index")
print("   And many more!")
