import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os.path as op
import warnings

def load_brainvision_data(file_path, preload=True):
    """
    Load BrainVision EEG data (.vhdr file) using MNE-Python.
    
    Parameters:
    -----------
    file_path : str
        Path to the .vhdr file
    preload : bool
        Whether to load data into memory (True) or keep it on disk (False)
        
    Returns:
    --------
    raw : mne.io.Raw
        The loaded EEG data
    """
    print(f"Loading BrainVision data from: {file_path}")
    
    # Raise a warning if file doesn't exist
    if not op.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if file has .vhdr extension
    if not file_path.endswith('.vhdr'):
        warnings.warn("File doesn't have .vhdr extension. Make sure it's a BrainVision header file.")
    
    # Load the data
    try:
        raw = mne.io.read_raw_brainvision(file_path, preload=preload)
        print(f"Successfully loaded data with {len(raw.ch_names)} channels.")
        print(f"Sampling rate: {raw.info['sfreq']} Hz")
        print(f"Recording duration: {raw.times[-1]:.2f} seconds")
        return raw
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def extract_and_visualize_events(raw):
    """
    Extract events and annotations from the raw data and visualize them.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The loaded EEG data
        
    Returns:
    --------
    events : numpy.ndarray
        MNE events array (onset, duration, trigger_code)
    event_dict : dict
        Dictionary of event types and their trigger codes
    event_df : pandas.DataFrame
        DataFrame containing event information
    """
    # Extract events and event_id dictionary
    try:
        events, event_id = mne.events_from_annotations(raw)
        print(f"Extracted {len(events)} events with {len(event_id)} unique event types")
    except Exception as e:
        print(f"Error extracting events: {str(e)}")
        raise
    
    # Create a DataFrame for better visualization
    event_df = pd.DataFrame({
        'Sample': events[:, 0],
        'Time (s)': events[:, 0] / raw.info['sfreq'],
        'Trigger Code': events[:, 2],
        'Event ID': [key for code in events[:, 2] for key, val in event_id.items() if val == code]
    })
    
    # Count occurrences of each trigger code
    trigger_counts = Counter(events[:, 2])
    
    # Print unique trigger codes and their counts
    print("\nUnique trigger codes and their counts:")
    for code, count in sorted(trigger_counts.items()):
        # Find the event ID(s) that correspond to this code
        event_names = [name for name, val in event_id.items() if val == code]
        event_names_str = ", ".join(event_names)
        print(f"Trigger Code: {code}, Event ID: {event_names_str}, Count: {count}")
    
    # Visualize events distribution
    plt.figure(figsize=(12, 6))
    
    # Plot event distribution by time
    plt.subplot(2, 1, 1)
    plt.plot(event_df['Time (s)'], event_df['Trigger Code'], 'o', markersize=5)
    plt.xlabel('Time (s)')
    plt.ylabel('Trigger Code')
    plt.title('Event Distribution Over Time')
    plt.grid(True)
    
    # Plot event counts
    plt.subplot(2, 1, 2)
    labels, counts = zip(*sorted(trigger_counts.items()))
    plt.bar(labels, counts)
    plt.xlabel('Trigger Code')
    plt.ylabel('Count')
    plt.title('Event Counts by Trigger Code')
    plt.tight_layout()
    plt.show()
    
    return events, event_id, event_df

def create_condition_mapping_template(event_id):
    """
    Create a template for mapping event IDs to experimental conditions.
    
    Parameters:
    -----------
    event_id : dict
        Dictionary of event types and their trigger codes
        
    Returns:
    --------
    condition_mapping : dict
        Template dictionary for mapping trigger codes to conditions
    """
    condition_mapping = {}
    
    print("\nCondition Mapping Template:")
    print("Please modify this template based on your experimental design.")
    print("Example: {'S  1': 'condition_name', 'S  2': 'other_condition'}")
    
    # Create a template dictionary with placeholder condition names
    for event_name, trigger_code in sorted(event_id.items(), key=lambda x: x[1]):
        condition_name = f"condition_{trigger_code}"  # Placeholder
        condition_mapping[event_name] = condition_name
        print(f"'{event_name}': '{condition_name}',  # Trigger code: {trigger_code}")
    
    return condition_mapping

def extract_condition_epochs(raw, events, event_id, condition_mapping, 
                            tmin=-0.2, tmax=1.0, baseline=(None, 0),
                            reject=None, flat=None):
    """
    Extract epochs for specified conditions.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The loaded EEG data
    events : numpy.ndarray
        MNE events array
    event_id : dict
        Dictionary of event types and their trigger codes
    condition_mapping : dict
        Dictionary mapping event IDs to condition names
    tmin : float
        Start time of epoch relative to event in seconds
    tmax : float
        End time of epoch relative to event in seconds
    baseline : tuple
        Baseline period (start, end) in seconds
    reject : dict or None
        Rejection parameters (e.g., {'eeg': 100e-6} for 100 µV)
    flat : dict or None
        Flatness rejection parameters
        
    Returns:
    --------
    epochs_dict : dict
        Dictionary of condition-specific epochs
    """
    # Reverse the condition mapping to map condition names to event IDs
    condition_to_event = {}
    for event_name, condition in condition_mapping.items():
        if condition not in condition_to_event:
            condition_to_event[condition] = []
        condition_to_event[condition].append(event_name)
    
    # Create a dictionary to store epochs for each condition
    epochs_dict = {}
    
    # Extract epochs for each condition
    for condition, event_names in condition_to_event.items():
        # Skip placeholder conditions (those starting with 'condition_')
        if condition.startswith('condition_'):
            continue
        
        # Create a subset of the event_id dictionary for this condition
        condition_event_id = {name: event_id[name] for name in event_names if name in event_id}
        
        if not condition_event_id:
            print(f"Warning: No events found for condition '{condition}'")
            continue
        
        # Extract epochs for this condition
        try:
            epochs = mne.Epochs(raw, events, event_id=condition_event_id, 
                              tmin=tmin, tmax=tmax, baseline=baseline,
                              reject=reject, flat=flat, preload=True)
            
            epochs_dict[condition] = epochs
            print(f"Extracted {len(epochs)} epochs for condition '{condition}'")
            
            # Plot average ERP for this condition
            evoked = epochs.average()
            fig = evoked.plot(spatial_colors=True, gfp=True, 
                            titles=f"ERP for condition: {condition}")
            
        except Exception as e:
            print(f"Error extracting epochs for condition '{condition}': {str(e)}")
    
    return epochs_dict

def verify_data_quality(raw, epochs_dict=None):
    """
    Verify data quality and potential issues.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The loaded EEG data
    epochs_dict : dict or None
        Dictionary of condition-specific epochs
        
    Returns:
    --------
    issues : list
        List of identified issues
    """
    issues = []
    
    # Check sampling rate
    sfreq = raw.info['sfreq']
    if sfreq < 250:
        issues.append(f"Low sampling rate ({sfreq} Hz) may limit your frequency analysis")
    
    # Check for missing channels
    if len(raw.ch_names) < 64:
        issues.append(f"Expected 64 channels, but found {len(raw.ch_names)}")
    
    # Check for flat channels
    flat_chans = []
    data, _ = raw[:, :]
    for i, ch_name in enumerate(raw.ch_names):
        if np.std(data[i]) < 1e-7:
            flat_chans.append(ch_name)
    
    if flat_chans:
        issues.append(f"Found potentially flat channels: {', '.join(flat_chans)}")
    
    # Check baseline and ensure it's zero mean
    if epochs_dict:
        for condition, epochs in epochs_dict.items():
            baseline_data = epochs.get_data(tmin=epochs.tmin, tmax=0)
            mean_baseline = baseline_data.mean()
            if abs(mean_baseline) > 1e-5:
                issues.append(f"Condition '{condition}' has non-zero baseline mean: {mean_baseline}")
    
    # Check for DC offset in the raw data
    dc_offset = np.mean(data)
    if abs(dc_offset) > 1e-5:
        issues.append(f"Raw data has DC offset: {dc_offset}")
    
    # Print the issues
    if issues:
        print("\nPotential issues detected:")
        for i, issue in enumerate(issues):
            print(f"{i+1}. {issue}")
    else:
        print("\nNo potential issues detected. Data quality looks good!")
    
    return issues

def analyze_brainvision_data(file_path, condition_mapping=None, tmin=-0.2, tmax=1.0):
    """
    Complete workflow for analyzing BrainVision EEG data.
    
    Parameters:
    -----------
    file_path : str
        Path to the .vhdr file
    condition_mapping : dict or None
        Dictionary mapping event IDs to condition names
    tmin : float
        Start time of epoch relative to event in seconds
    tmax : float
        End time of epoch relative to event in seconds
        
    Returns:
    --------
    raw : mne.io.Raw
        The loaded EEG data
    epochs_dict : dict
        Dictionary of condition-specific epochs
    """
    # Load the data
    raw = load_brainvision_data(file_path)
    
    # Extract and visualize events
    events, event_id, event_df = extract_and_visualize_events(raw)
    
    # Create condition mapping if not provided
    if condition_mapping is None:
        condition_mapping = create_condition_mapping_template(event_id)
        print("\nPlease define your condition mapping and re-run with the mapping provided.")
        return raw, None
    
    # Extract epochs for each condition
    epochs_dict = extract_condition_epochs(raw, events, event_id, condition_mapping, 
                                         tmin=tmin, tmax=tmax)
    
    # Verify data quality
    verify_data_quality(raw, epochs_dict)
    
    return raw, epochs_dict

def visualize_channel_data(raw, channels=None, duration=10, n_channels=5):
    """
    Visualize raw data for selected channels.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The loaded EEG data
    channels : list or None
        List of channel names to plot (defaults to first n_channels)
    duration : float
        Duration of data to plot in seconds
    n_channels : int
        Number of channels to plot if channels is None
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if channels is None:
        channels = raw.ch_names[:n_channels]
    
    fig = raw.plot(duration=duration, n_channels=len(channels), 
                 scalings='auto', picks=channels, show=True)
    
    return fig

def prepare_for_time_frequency_analysis(epochs_dict, freqs=np.arange(4, 40, 1), 
                                      n_cycles=7, decim=3):
    """
    Prepare epochs for time-frequency analysis.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary of condition-specific epochs
    freqs : array-like
        Frequencies of interest in Hz
    n_cycles : int or array-like
        Number of cycles for each frequency
    decim : int
        Decimation factor
        
    Returns:
    --------
    tf_dict : dict
        Dictionary of time-frequency objects for each condition
    """
    tf_dict = {}
    
    for condition, epochs in epochs_dict.items():
        # Compute time-frequency representation using wavelets
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                           decim=decim, return_itc=False, average=False)
        
        tf_dict[condition] = power
        
        # Plot average power for a few channels
        power_avg = power.average()
        fig = power_avg.plot_topo(vmin=0, vmax=None, title=f'Power for {condition}')
    
    return tf_dict

def compute_coherence(tf_dict, channel_pairs=None):
    """
    Compute coherence between channel pairs for different conditions.
    
    Parameters:
    -----------
    tf_dict : dict
        Dictionary of time-frequency objects for each condition
    channel_pairs : list of tuples or None
        List of channel pairs to compute coherence for
        
    Returns:
    --------
    coh_dict : dict
        Dictionary of coherence values for each condition and channel pair
    """
    if channel_pairs is None:
        # Example channel pairs (frontal to parietal)
        channel_pairs = [('Fz', 'Pz'), ('F3', 'P3'), ('F4', 'P4')]
    
    coh_dict = {}
    
    for condition, power in tf_dict.items():
        coh_dict[condition] = {}
        
        for ch1, ch2 in channel_pairs:
            # Check if channels exist
            if ch1 not in power.ch_names or ch2 not in power.ch_names:
                print(f"Warning: Channels {ch1} and/or {ch2} not found in condition {condition}")
                continue
            
            # Get indices for the channels
            idx1 = power.ch_names.index(ch1)
            idx2 = power.ch_names.index(ch2)
            
            # Compute coherence
            coh = mne.connectivity.spectral_connectivity_epochs(
                power.data[:, [idx1, idx2], :, :], 
                method='coh', mode='multitaper', sfreq=power.info['sfreq'], 
                fmin=power.freqs.min(), fmax=power.freqs.max(), faverage=True
            )
            
            # Store coherence value for this pair
            coh_dict[condition][(ch1, ch2)] = coh
            
            # Plot coherence
            plt.figure(figsize=(8, 4))
            plt.plot(power.freqs, coh[0][0, 1, :].T)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Coherence')
            plt.title(f'Coherence between {ch1} and {ch2} for {condition}')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.show()
    
    return coh_dict

def main():
    """
    Example usage of all functions.
    """
    # Example file path - replace with your actual file path
    file_path = 'your_data_file.vhdr'
    
    # Load data and extract events
    raw = load_brainvision_data(file_path)
    events, event_id, event_df = extract_and_visualize_events(raw)
    
    # Create a condition mapping template
    condition_mapping = create_condition_mapping_template(event_id)
    
    # Example: Define your actual condition mapping
    # This is where you would define your actual experiment conditions
    # based on the template provided
    actual_condition_mapping = {
        'S  1': 'rest',
        'S 11': 'allocentric',
        'S 12': 'egocentric',
        # Add more mappings as needed
    }
    
    # Extract epochs for each condition
    epochs_dict = extract_condition_epochs(raw, events, event_id, actual_condition_mapping)
    
    # Verify data quality
    verify_data_quality(raw, epochs_dict)
    
    # Visualize channel data
    visualize_channel_data(raw, channels=['Fz', 'Cz', 'Pz', 'O1', 'O2'])
    
    # Prepare for time-frequency analysis
    tf_dict = prepare_for_time_frequency_analysis(epochs_dict)
    
    # Compute coherence
    coh_dict = compute_coherence(tf_dict)
    
    # Print summary information
    print("\nAnalysis Summary:")
    print(f"Loaded data from {file_path}")
    print(f"Found {len(events)} events with {len(event_id)} unique event types")
    print(f"Created epochs for {len(epochs_dict)} conditions")
    
    # Return important variables for further analysis
    return raw, epochs_dict, tf_dict, coh_dict

if __name__ == "__main__":
    main()