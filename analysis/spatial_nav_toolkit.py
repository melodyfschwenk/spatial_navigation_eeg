#!/usr/bin/env "C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe"
"""
Spatial Navigation EEG Analysis Toolkit
=======================================

This package provides a comprehensive toolkit for analyzing spatial navigation EEG experiments,
including behavioral analysis, EEG preprocessing, visualization, and statistical analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Ensure statsmodels is properly installed
import sys
import subprocess
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM
except ImportError:
    print("Installing statsmodels...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "statsmodels"])
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM
# Ensure MNE is properly installed
try:
    import mne
    from mne.time_frequency import tfr_morlet
except ImportError:
    print("Installing MNE-Python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "mne"])
    try:
        import mne
        from mne.time_frequency import tfr_morlet
    except ImportError:
        print("Failed to import MNE after installation. Installing additional dependencies...")
        # MNE might need additional dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy scipy matplotlib"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "mne"])
        import mne
        from mne.time_frequency import tfr_morlet
import warnings
from typing import Optional, List, Dict, Tuple, Union, Any

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class SpatialNavDataset:
    """
    Main class for handling spatial navigation experiment datasets.
    This class loads, validates, and provides access to all data files.
    """
    
    def __init__(self, data_dir, participant_ids=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data_dir : str
            Path to the directory containing all data files
        participant_ids : list, optional
            List of participant IDs to include, if None, all are included
        """
        self.data_dir = data_dir
        self.participant_ids = participant_ids
        self.main_data = None
        self.block_data = {}
        self.eeg_triggers = {}
        self.metadata = {}
        
        print(f"Initializing dataset from {data_dir}")
        self.load_data()
    
    def load_data(self):
        """Load all data files from the data directory."""
        # Load main data CSV
        main_csv = os.path.join(self.data_dir, "main_data.csv")
        if (os.path.exists(main_csv)):
            self.main_data = pd.read_csv(main_csv)
            print(f"Loaded main data: {len(self.main_data)} trials")
            
            # Filter by participant_ids if specified
            if self.participant_ids is not None:
                self.main_data = self.main_data[self.main_data['participant_id'].isin(self.participant_ids)]
                print(f"Filtered to {len(self.participant_ids)} participants")
        else:
            warnings.warn(f"Main data file not found at {main_csv}")
        
        # Load block summary CSVs
        block_files = [f for f in os.listdir(self.data_dir) if f.startswith("block_") and f.endswith(".csv")]
        for block_file in block_files:
            block_id = block_file.replace("block_", "").replace(".csv", "")
            self.block_data[block_id] = pd.read_csv(os.path.join(self.data_dir, block_file))
            
        print(f"Loaded {len(self.block_data)} block data files")
        
        # Load EEG trigger files
        trigger_files = [f for f in os.listdir(self.data_dir) if f.endswith("_triggers.json")]
        for trigger_file in trigger_files:
            participant_id = trigger_file.replace("_triggers.json", "")
            if self.participant_ids is None or participant_id in self.participant_ids:
                with open(os.path.join(self.data_dir, trigger_file), 'r') as f:
                    self.eeg_triggers[participant_id] = json.load(f)
        
        print(f"Loaded EEG trigger files for {len(self.eeg_triggers)} participants")
        
        # Load metadata files
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        if (os.path.exists(metadata_file)):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print("Loaded experiment metadata")
        else:
            warnings.warn(f"Metadata file not found at {metadata_file}")
    
    def get_participant_data(self, participant_id):
        """Get all data for a specific participant."""
        if self.main_data is None:
            return None
        
        return self.main_data[self.main_data['participant_id'] == participant_id]
    
    def get_condition_data(self, navigation_type=None, difficulty=None):
        """Get data for specific conditions."""
        if self.main_data is None:
            return None
        
        data = self.main_data.copy()
        
        if navigation_type is not None:
            data = data[data['navigation_type'] == navigation_type]
        
        if difficulty is not None:
            data = data[data['difficulty'] == difficulty]
            
        return data
    
    def summarize(self):
        """Print a summary of the dataset."""
        if self.main_data is None:
            print("No data loaded")
            return
        
        print("\n=== Dataset Summary ===")
        print(f"Total trials: {len(self.main_data)}")
        print(f"Participants: {self.main_data['participant_id'].nunique()}")
        print(f"Navigation types: {self.main_data['navigation_type'].unique()}")
        print(f"Difficulty levels: {self.main_data['difficulty'].unique()}")
        print(f"Average accuracy: {self.main_data['accuracy'].mean():.2f}")
        print(f"Average response time: {self.main_data['rt'].mean():.2f} ms")
        print("=====================\n")


class BehavioralAnalysis:
    """
    Class for analyzing behavioral data from spatial navigation experiments.
    """
    
    def __init__(self, dataset):
        """
        Initialize the behavioral analysis.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset to analyze
        """
        self.dataset = dataset
    
    def accuracy_by_condition(self):
        """Calculate mean accuracy by navigation type and difficulty."""
        if self.dataset.main_data is None:
            return None
        
        return self.dataset.main_data.groupby(['navigation_type', 'difficulty'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    def rt_by_condition(self):
        """Calculate mean response time by navigation type and difficulty."""
        if self.dataset.main_data is None:
            return None
        
        return self.dataset.main_data.groupby(['navigation_type', 'difficulty'])['rt'].agg(['mean', 'std', 'count']).reset_index()
    
    def calculate_switch_costs(self):
        """
        Calculate switch costs between navigation types.
        Switch cost is defined as the difference in RT between trials where the navigation type
        changes compared to when it stays the same.
        """
        if self.dataset.main_data is None:
            return None
        
        # Create a copy of the data
        data = self.dataset.main_data.copy().sort_values(['participant_id', 'session'])
        
        # Identify switches
        data['prev_nav_type'] = data.groupby('participant_id')['navigation_type'].shift(1)
        data['is_switch'] = (data['navigation_type'] != data['prev_nav_type']).astype(int)
        
        # For the first trial of each participant, set is_switch to NaN
        first_trials = data.groupby('participant_id').head(1).index
        data.loc[first_trials, 'is_switch'] = np.nan
        
        # Calculate mean RT for switch and non-switch trials
        switch_costs = data.groupby(['participant_id', 'is_switch'])['rt'].mean().reset_index()
        
        # Reshape to have one row per participant with switch and non-switch RT
        switch_costs = switch_costs.pivot(index='participant_id', columns='is_switch', values='rt')
        switch_costs.columns = ['non_switch_rt', 'switch_rt']
        
        # Calculate the switch cost (difference in RT)
        switch_costs['switch_cost'] = switch_costs['switch_rt'] - switch_costs['non_switch_rt']
        
        return switch_costs.reset_index()
    
    def analyze_learning_effects(self):
        """
        Analyze learning effects across blocks.
        Returns a DataFrame with accuracy and RT by block.
        """
        if self.dataset.main_data is None:
            return None
        
        # Ensure we have a condition column
        if 'condition' not in self.dataset.main_data.columns:
            self.dataset.main_data['condition'] = self.dataset.main_data.apply(
                lambda row: f"{row['navigation_type']}_{row['difficulty']}",
                axis=1
            )
        
        # Assuming there's a 'block' column, if not we'll create one
        if 'block' not in self.dataset.main_data.columns:
            # Try to derive block from session if possible
            if 'session' in self.dataset.main_data.columns:
                blocks_per_session = self.dataset.metadata.get('blocks_per_session', 1)
                self.dataset.main_data['block'] = (self.dataset.main_data['session'] // blocks_per_session) + 1
            else:
                warnings.warn("Cannot analyze learning effects: no block or session information available")
                return None
        
        # Calculate accuracy and RT by block
        learning = self.dataset.main_data.groupby(['block', 'condition']).agg({
            'accuracy': 'mean',
            'rt': 'mean'
        }).reset_index()
        
        # Determine block sequence within each condition
        condition_blocks = {}
        for condition in learning['condition'].unique():
            condition_blocks[condition] = learning[learning['condition'] == condition].sort_values('block')
            condition_blocks[condition]['condition_block'] = range(1, len(condition_blocks[condition]) + 1)
            
        # Combine back into one dataframe
        learning_with_condition_blocks = pd.concat(condition_blocks.values())
        
        return learning_with_condition_blocks
    
    def individual_differences(self):
        """
        Analyze individual differences between participants.
        Returns a DataFrame with accuracy and RT by participant.
        """
        if self.dataset.main_data is None:
            return None
        
        individual = self.dataset.main_data.groupby('participant_id').agg({
            'accuracy': 'mean',
            'rt': 'mean',
            'session': 'count'
        }).reset_index()
        
        individual.columns = ['participant_id', 'mean_accuracy', 'mean_rt', 'trial_count']
        
        return individual
    
    def run_repeated_measures_anova(self, dv='accuracy'):
        """
        Run a repeated measures ANOVA on the specified dependent variable.
        
        Parameters
        ----------
        dv : str
            The dependent variable to analyze ('accuracy' or 'rt')
            
        Returns
        -------
        DataFrame
            ANOVA results table
        """
        if self.dataset.main_data is None:
            return None
        
        # Prepare data for ANOVA: average by participant, navigation_type, and difficulty
        anova_data = self.dataset.main_data.groupby(['participant_id', 'navigation_type', 'difficulty'])[dv].mean().reset_index()
        
        # Run the ANOVA
        try:
            aovrm = AnovaRM(anova_data, dv, 'participant_id', within=['navigation_type', 'difficulty'])
            result = aovrm.fit()
            return result
        except Exception as e:
            warnings.warn(f"Error running ANOVA: {e}")
            return None


class EEGPreprocessing:
    """
    Class for preprocessing EEG data from spatial navigation experiments.
    """
    
    def __init__(self, dataset, eeg_data_dir=None):
        """
        Initialize the EEG preprocessing.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset containing trigger information
        eeg_data_dir : str, optional
            Directory containing raw EEG data files
        """
        self.dataset = dataset
        self.eeg_data_dir = eeg_data_dir or dataset.data_dir
        self.raw_eeg = {}
        self.epochs = {}
    
    def convert_triggers_to_mne(self, participant_id):
        """
        Convert EEG trigger files to MNE-Python format.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
            
        Returns
        -------
        dict
            Dictionary of event IDs and corresponding triggers
        """
        if participant_id not in self.dataset.eeg_triggers:
            warnings.warn(f"No trigger data found for participant {participant_id}")
            return None
        
        # Extract trigger information
        triggers = self.dataset.eeg_triggers[participant_id]
        
        # Create events array for MNE: [sample_idx, 0, event_id]
        events = []
        event_id = {}
        
        # Map event types to event IDs
        event_types = set()
        for t in triggers:
            event_types.add(t['type'])
        
        for i, event_type in enumerate(sorted(event_types)):
            event_id[event_type] = i + 1
        
        # Create events array
        for t in triggers:
            # Convert timestamp to sample index (assuming sample rate from metadata)
            sample_rate = self.dataset.metadata.get('eeg_sample_rate', 1000)  # Default to 1000 Hz
            sample_idx = int(t['timestamp'] * sample_rate)
            event_type_id = event_id[t['type']]
            events.append([sample_idx, 0, event_type_id])
        
        return np.array(events), event_id
    
    def load_raw_eeg(self, participant_id, format='edf'):
        """
        Load raw EEG data for a participant.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to load
        format : str
            The format of the EEG data file ('edf', 'bdf', 'fif', etc.)
            
        Returns
        -------
        mne.io.Raw
            The raw EEG data
        """
        # Construct the filename based on the format
        filename = os.path.join(self.eeg_data_dir, f"sub-{participant_id}_eeg.{format}")
        
        if not os.path.exists(filename):
            warnings.warn(f"EEG data file not found: {filename}")
            return None
        
        # Load the data based on the format
        if format.lower() == 'edf':
            raw = mne.io.read_raw_edf(filename, preload=True)
        elif format.lower() == 'bdf':
            raw = mne.io.read_raw_bdf(filename, preload=True)
        elif format.lower() == 'fif':
            raw = mne.io.read_raw_fif(filename, preload=True)
        else:
            warnings.warn(f"Unsupported EEG format: {format}")
            return None
        
        self.raw_eeg[participant_id] = raw
        return raw
    
    def extract_epochs(self, participant_id, event_id=None, tmin=-0.2, tmax=1.0):
        """
        Extract epochs time-locked to events.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        event_id : dict, optional
            Dictionary mapping event names to IDs
        tmin : float
            Start time of the epoch in seconds
        tmax : float
            End time of the epoch in seconds
            
        Returns
        -------
        mne.Epochs
            The extracted epochs
        """
        if participant_id not in self.raw_eeg:
            warnings.warn(f"No raw EEG data loaded for participant {participant_id}")
            return None
        
        # Get events
        events, auto_event_id = self.convert_triggers_to_mne(participant_id)
        if event_id is None:
            event_id = auto_event_id
        
        # Extract epochs
        epochs = mne.Epochs(
            self.raw_eeg[participant_id], 
            events, 
            event_id, 
            tmin, 
            tmax, 
            baseline=(tmin, 0),
            preload=True
        )
        
        self.epochs[participant_id] = epochs
        return epochs
    
    def apply_baseline_correction(self, participant_id, baseline=(None, 0)):
        """
        Apply baseline correction to epochs.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        baseline : tuple
            The baseline period (start, end) in seconds
            
        Returns
        -------
        mne.Epochs
            The baseline-corrected epochs
        """
        if participant_id not in self.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        # Apply baseline correction
        self.epochs[participant_id].apply_baseline(baseline)
        return self.epochs[participant_id]
    
    def detect_artifacts(self, participant_id, threshold=100e-6):
        """
        Detect artifacts in epochs.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        threshold : float
            Threshold for artifact detection in volts
            
        Returns
        -------
        list
            List of epoch indices with artifacts
        """
        if participant_id not in self.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        # Detect artifacts using peak-to-peak amplitude
        epochs = self.epochs[participant_id]
        data = epochs.get_data()
        
        # Calculate peak-to-peak amplitude for each epoch
        peak_to_peak = np.ptp(data, axis=2)
        
        # Find epochs where any channel exceeds the threshold
        bad_epochs = np.where(np.any(peak_to_peak > threshold, axis=1))[0]
        
        print(f"Detected {len(bad_epochs)} epochs with artifacts for participant {participant_id}")
        return bad_epochs
    
    def analyze_mu_rhythm(self, participant_id, freqs=np.arange(8, 13, 1), n_cycles=5):
        """
        Analyze mu rhythm (8-12 Hz) activity.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to process
        freqs : array
            The frequencies to analyze
        n_cycles : int or array
            Number of cycles for the wavelet transform
            
        Returns
        -------
        mne.time_frequency.EpochsTFR
            Time-frequency representation of the epochs
        """
        if participant_id not in self.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return None
        
        # Calculate time-frequency representation
        tfr = tfr_morlet(
            self.epochs[participant_id],
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=False
        )
        
        return tfr
    
    def export_to_eeglab(self, participant_id, filename):
        """
        Export epochs to EEGLAB format.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to export
        filename : str
            The output filename
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if participant_id not in self.epochs:
            warnings.warn(f"No epochs data found for participant {participant_id}")
            return False
        
        # Export to EEGLAB .set format
        self.epochs[participant_id].export(filename, fmt='eeglab')
        print(f"Exported epochs to {filename}")
        return True


class Visualization:
    """
    Class for creating visualizations of spatial navigation experiment data.
    """
    
    def __init__(self, dataset):
        """
        Initialize the visualization.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset to visualize
        """
        self.dataset = dataset
    
    def plot_accuracy_by_condition(self, save_path=None):
        """
        Plot accuracy by navigation type and difficulty.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if self.dataset.main_data is None:
            return None
        
        behavioral = BehavioralAnalysis(self.dataset)
        accuracy_data = behavioral.accuracy_by_condition()
        
        plt.figure(figsize=(10, 6))
        # Updated parameters for newer seaborn versions
        sns.barplot(
            x='navigation_type', 
            y='mean', 
            hue='difficulty',
            data=accuracy_data,
            palette='viridis'
        )
        plt.title('Accuracy by Navigation Type and Difficulty')
        plt.xlabel('Navigation Type')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_rt_by_condition(self, save_path=None):
        """
        Plot response time by navigation type and difficulty.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if self.dataset.main_data is None:
            return None
        
        behavioral = BehavioralAnalysis(self.dataset)
        rt_data = behavioral.rt_by_condition()
        
        plt.figure(figsize=(10, 6))
        # Updated parameters for newer seaborn versions
        sns.barplot(
            x='navigation_type', 
            y='mean', 
            hue='difficulty',
            data=rt_data,
            palette='viridis'
        )
        plt.title('Response Time by Navigation Type and Difficulty')
        plt.xlabel('Navigation Type')
        plt.ylabel('Response Time (ms)')
        plt.grid(True, axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_switch_costs(self, save_path=None):
        """
        Plot switch costs.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        behavioral = BehavioralAnalysis(self.dataset)
        switch_costs = behavioral.calculate_switch_costs()
        
        if switch_costs is None:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Plot switch vs. non-switch RT
        plt.subplot(1, 2, 1)
        mean_rts = switch_costs[['non_switch_rt', 'switch_rt']].mean()
        std_rts = switch_costs[['non_switch_rt', 'switch_rt']].std()
        
        bar_positions = np.arange(2)
        plt.bar(bar_positions, mean_rts, yerr=std_rts, capsize=10, color=['#2196F3', '#F44336'])
        plt.xticks(bar_positions, ['Non-Switch', 'Switch'])
        plt.ylabel('Response Time (ms)')
        plt.title('RT for Switch vs. Non-Switch Trials')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Plot individual switch costs
        plt.subplot(1, 2, 2)
        sns.histplot(switch_costs['switch_cost'], kde=True, bins=10, color='#4CAF50')
        plt.xlabel('Switch Cost (ms)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Switch Costs')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_learning_effects(self, save_path=None):
        """
        Plot learning effects across blocks.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        behavioral = BehavioralAnalysis(self.dataset)
        learning = behavioral.analyze_learning_effects()
        
        if learning is None:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy by block
        plt.subplot(1, 2, 1)
        plt.plot(learning['block'], learning['accuracy'], 'o-', color='#2196F3', linewidth=2)
        plt.xlabel('Block')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Across Blocks')
        plt.grid(True, alpha=0.3)
        
        # Plot RT by block
        plt.subplot(1, 2, 2)
        plt.plot(learning['block'], learning['rt'], 'o-', color='#F44336', linewidth=2)
        plt.xlabel('Block')
        plt.ylabel('Response Time (ms)')
        plt.title('Response Time Across Blocks')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_individual_differences(self, save_path=None):
        """
        Plot individual differences between participants.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        behavioral = BehavioralAnalysis(self.dataset)
        individual = behavioral.individual_differences()
        
        if individual is None:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Handle case when individual is empty
        if individual.empty:
            plt.text(0.5, 0.5, "No data available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            return plt.gcf()
            
        # Plot accuracy vs. RT scatter
        plt.scatter(
            individual['mean_rt'], 
            individual['mean_accuracy'],
            s=individual['trial_count'] / 2,  # Size proportional to number of trials
            alpha=0.7,
            c=np.arange(len(individual)),  # Color by participant index
            cmap='viridis'
        )
        
        plt.xlabel('Mean Response Time (ms)')
        plt.ylabel('Mean Accuracy')
        plt.title('Individual Differences: Accuracy vs. Response Time')
        plt.grid(True, alpha=0.3)
        
        # Add participant IDs as labels
        for i, row in individual.iterrows():
            plt.annotate(
                row['participant_id'],
                (row['mean_rt'], row['mean_accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_topographic_map(self, participant_id, time_window=(0.3, 0.5), freq_band=(8, 12), save_path=None):
        """
        Plot a topographic map of EEG activity.
        
        Parameters
        ----------
        participant_id : str
            The participant ID to plot
        time_window : tuple
            The time window to plot (start, end) in seconds
        freq_band : tuple
            The frequency band to plot (start, end) in Hz
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        eeg = EEGPreprocessing(self.dataset)
        
        # Check if raw EEG data is already loaded
        if participant_id not in eeg.raw_eeg:
            raw = eeg.load_raw_eeg(participant_id)
            if raw is None:
                warnings.warn(f"Could not load raw EEG data for participant {participant_id}")
                return None
        
        # Check if epochs are already extracted
        if participant_id not in eeg.epochs:
            epochs = eeg.extract_epochs(participant_id)
            if epochs is None:
                warnings.warn(f"Could not extract epochs for participant {participant_id}")
                return None
        
        # Analyze mu rhythm
        tfr = eeg.analyze_mu_rhythm(
            participant_id,
            freqs=np.linspace(freq_band[0], freq_band[1], 10),
            n_cycles=5
        )
        
        if tfr is None:
            warnings.warn(f"Could not analyze mu rhythm for participant {participant_id}")
            return None
        
        # Ensure there's data to plot before proceeding
        if not hasattr(tfr, 'data') or tfr.data.size == 0:
            warnings.warn(f"No data available for topographic map for participant {participant_id}")
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, "No data available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            return plt.gcf()
        
        # Calculate average power in the time window safely
        time_mask = (tfr.times >= time_window[0]) & (tfr.times <= time_window[1])
        if not any(time_mask):
            warnings.warn(f"No time points within specified window {time_window}")
            time_mask = np.ones_like(tfr.times, dtype=bool)  # Use all time points as fallback
        
        avg_power = tfr.data.mean(axis=0).mean(axis=1)[:, time_mask].mean(axis=1)
        
        # Plot topographic map
        plt.figure(figsize=(8, 8))
        mne.viz.plot_topomap(
            avg_power,
            tfr.info,
            cmap='RdBu_r',
            vmin=-np.max(np.abs(avg_power)),
            vmax=np.max(np.abs(avg_power)),
            contours=6,
            show=False
        )
        plt.title(f'Mu Rhythm Topography ({time_window[0]}-{time_window[1]}s, {freq_band[0]}-{freq_band[1]}Hz)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_navigation_strategy_comparison(self, navigation_types=None, metrics=None, save_path=None):
        """
        Plot a comparison of navigation strategies.
        
        Parameters
        ----------
        navigation_types : list, optional
            List of navigation types to compare
        metrics : list, optional
            List of metrics to compare ('accuracy', 'rt')
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if self.dataset.main_data is None:
            return None
        
        # Default parameters
        if navigation_types is None:
            navigation_types = self.dataset.main_data['navigation_type'].unique()
        
        if metrics is None:
            metrics = ['accuracy', 'rt']
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # For each metric, plot a comparison
        for i, metric in enumerate(metrics):
            # Calculate mean and std by navigation type
            data = self.dataset.main_data.groupby('navigation_type')[metric].agg(['mean', 'std']).reset_index()
            data = data[data['navigation_type'].isin(navigation_types)]
            
            # Plot
            axes[i].bar(
                data['navigation_type'],
                data['mean'],
                yerr=data['std'],
                capsize=10,
                color=sns.color_palette('viridis', len(data))
            )
            
            # Labels
            axes[i].set_xlabel('Navigation Type')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} by Navigation Type')
            axes[i].grid(True, axis='y', alpha=0.3)
            
            # For accuracy, limit y-axis to 0-1
            if metric == 'accuracy':
                axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class StatisticalAnalysis:
    """
    Class for statistical analysis of spatial navigation experiment data.
    """
    
    def __init__(self, dataset):
        """
        Initialize the statistical analysis.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset to analyze
        """
        self.dataset = dataset
    
    def repeated_measures_anova(self, dv='accuracy'):
        """
        Perform a repeated measures ANOVA.
        
        Parameters
        ----------
        dv : str
            The dependent variable to analyze
            
        Returns
        -------
        statsmodels.stats.anova.AnovaResults
            The ANOVA results
        """
        behavioral = BehavioralAnalysis(self.dataset)
        return behavioral.run_repeated_measures_anova(dv)
    
    def paired_comparisons(self, dv='accuracy', groupby='navigation_type', correction='bonferroni'):
        """
        Perform paired comparisons between conditions.
        
        Parameters
        ----------
        dv : str
            The dependent variable to analyze
        groupby : str
            The variable to group by
        correction : str
            The multiple comparisons correction method
            
        Returns
        -------
        pandas.DataFrame 
            DataFrame with comparison results
        """
        if self.dataset.main_data is None:
            return None
        
        # Get the unique values of the groupby variable
        groups = self.dataset.main_data[groupby].unique()
        
        # Prepare results DataFrame
        results = pd.DataFrame(
            columns=['group1', 'group2', 'mean_diff', 't_stat', 'p_value', 'sig']
        )
        
        # For each pair of groups, perform a paired t-test
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                # Get the data for each group, averaged by participant
                data1 = self.dataset.main_data[self.dataset.main_data[groupby] == group1].groupby('participant_id')[dv].mean()
                data2 = self.dataset.main_data[self.dataset.main_data[groupby] == group2].groupby('participant_id')[dv].mean()
                
                # Only include participants with data in both groups
                shared_participants = set(data1.index) & set(data2.index)
                if len(shared_participants) < 2:
                    continue
                
                data1 = data1.loc[list(shared_participants)]
                data2 = data2.loc[list(shared_participants)]
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(data1, data2)
                
                # Add to results
                results = pd.concat([results, pd.DataFrame({
                    'group1': [group1],
                    'group2': [group2],
                    'mean_diff': [data2.mean() - data1.mean()],
                    't_stat': [t_stat],
                    'p_value': [p_value],
                    'sig': [False]
                })], ignore_index=True)
        
        # Apply multiple comparisons correction
        if len(results) > 0:
            if correction == 'bonferroni':
                alpha = 0.05 / len(results)
                results['sig'] = results['p_value'] < alpha
            elif correction == 'fdr':
                try:
                    from statsmodels.stats.multitest import fdrcorrection
                    _, results['p_value_adj'] = fdrcorrection(results['p_value'])
                    results['sig'] = results['p_value_adj'] < 0.05
                except ImportError:
                    warnings.warn("statsmodels.stats.multitest module not available. Using Bonferroni correction instead.")
                    alpha = 0.05 / len(results)
                    results['p_value_adj'] = results['p_value']
                    results['sig'] = results['p_value'] < alpha
        
        return results
    
    def correlation_analysis(self, var1='accuracy', var2='rt'):
        """
        Perform correlation analysis between two variables.
        
        Parameters
        ----------
        var1 : str
            The first variable
        var2 : str
            The second variable
            
        Returns
        -------
        tuple
            Tuple containing correlation coefficient and p-value
        """
        if self.dataset.main_data is None:
            return None
        
        # Check if variables exist
        if var1 not in self.dataset.main_data.columns:
            warnings.warn(f"Variable {var1} not found in dataset")
            return None
        
        if var2 not in self.dataset.main_data.columns:
            warnings.warn(f"Variable {var2} not found in dataset")
            return None
        
        # Calculate correlation by participant
        participant_corrs = []
        
        for participant_id in self.dataset.main_data['participant_id'].unique():
            participant_data = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]
            
            # Only calculate if we have enough data points
            if len(participant_data) >= 5:
                corr, p_value = stats.pearsonr(participant_data[var1], participant_data[var2])
                participant_corrs.append((participant_id, corr, p_value))
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(participant_corrs, columns=['participant_id', 'correlation', 'p_value'])
        
        # Calculate group-level statistics
        group_corr = corr_df['correlation'].mean()
        
        # One-sample t-test to test if correlation is significantly different from 0
        t_stat, p_value = stats.ttest_1samp(corr_df['correlation'], 0)
        
        return group_corr, p_value, corr_df
    
    def mixed_effects_model(self, dv='accuracy', fixed_effects=None, random_effects=None):
        """
        Fit a mixed-effects model.
        
        Parameters
        ----------
        dv : str
            The dependent variable
        fixed_effects : list
            List of fixed effects
        random_effects : list
            List of random effects
            
        Returns
        -------
        statsmodels.regression.mixed_linear_model.MixedLMResults
            The mixed-effects model results
        """
        if self.dataset.main_data is None:
            return None
        
        # Default parameters
        if fixed_effects is None:
            fixed_effects = ['navigation_type', 'difficulty']
        
        if random_effects is None:
            random_effects = ['participant_id']
        
        # Prepare the formula
        fixed_formula = ' + '.join(fixed_effects)
        formula = f"{dv} ~ {fixed_formula}"
        
        # Fit the model
        model = smf.mixedlm(
            formula,
            self.dataset.main_data,
            groups=self.dataset.main_data[random_effects[0]]
        )
        
        try:
            result = model.fit()
            return result
        except Exception as e:
            warnings.warn(f"Error fitting mixed-effects model: {e}")
            return None


# Usage example

def run_full_analysis(data_dir, output_dir=None):
    """
    Run a full analysis of spatial navigation experiment data.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing all data files
    output_dir : str, optional
        Path to the directory to save output files
    """
    # Validate input directories
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return
        
    # Create output directory if it doesn't exist
    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return
    
    # Load the dataset
    try:
        dataset = SpatialNavDataset(data_dir)
        dataset.summarize()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Behavioral analysis
    behavioral = BehavioralAnalysis(dataset)
    
    print("\n=== Behavioral Analysis ===")
    
    # Accuracy by condition
    accuracy = behavioral.accuracy_by_condition()
    print("\nAccuracy by condition:")
    print(accuracy)
    
    # RT by condition
    rt = behavioral.rt_by_condition()
    print("\nResponse time by condition:")
    print(rt)
    
    # Switch costs
    switch_costs = behavioral.calculate_switch_costs()
    print("\nSwitch costs:")
    print(switch_costs.describe())
    
    # Learning effects
    learning = behavioral.analyze_learning_effects()
    print("\nLearning effects:")
    print(learning)
    
    # Individual differences
    individual = behavioral.individual_differences()
    print("\nIndividual differences:")
    print(individual.head())
    
    # Error handling for visualizations and analysis
    try:
        # Create visualizations
        vis = Visualization(dataset)
        print("\n=== Creating Visualizations ===")
        
        # Accuracy by condition
        if output_dir is not None:
            vis.plot_accuracy_by_condition(os.path.join(output_dir, 'accuracy_by_condition.png'))
            vis.plot_rt_by_condition(os.path.join(output_dir, 'rt_by_condition.png'))
            vis.plot_switch_costs(os.path.join(output_dir, 'switch_costs.png'))
            vis.plot_learning_effects(os.path.join(output_dir, 'learning_effects.png'))
            vis.plot_individual_differences(os.path.join(output_dir, 'individual_differences.png'))
            vis.plot_navigation_strategy_comparison(
                save_path=os.path.join(output_dir, 'navigation_strategy_comparison.png')
            )
        else:
            vis.plot_accuracy_by_condition()
            vis.plot_rt_by_condition()
            vis.plot_switch_costs()
            vis.plot_learning_effects()
            vis.plot_individual_differences()
            vis.plot_navigation_strategy_comparison()
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    try:
        # Statistical analysis
        stats = StatisticalAnalysis(dataset)
        print("\n=== Statistical Analysis ===")
        
        # Repeated measures ANOVA
        anova_accuracy = stats.repeated_measures_anova('accuracy')
        if anova_accuracy is not None:
            print("\nRepeated measures ANOVA (Accuracy):")
            print(anova_accuracy.summary())
        
        anova_rt = stats.repeated_measures_anova('rt')
        if anova_rt is not None:
            print("\nRepeated measures ANOVA (RT):")
            print(anova_rt.summary())
        
        # Paired comparisons
        paired_comparisons = stats.paired_comparisons('accuracy', 'navigation_type')
        if paired_comparisons is not None:
            print("\nPaired comparisons (Accuracy by Navigation Type):")
            print(paired_comparisons)
        
        # Correlation analysis
        corr_result = stats.correlation_analysis('accuracy', 'rt')
        if corr_result is not None:
            group_corr, p_value, corr_df = corr_result
            print(f"\nCorrelation between Accuracy and RT: r = {group_corr:.3f}, p = {p_value:.3f}")
        
        # Mixed-effects model
        mixed_model = stats.mixed_effects_model('accuracy')
        if mixed_model is not None:
            print("\nMixed-effects model (Accuracy):")
            print(mixed_model.summary())
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
    
    print("\n=== Analysis Complete ===")
    if output_dir is not None:
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage with proper checking
    import argparse
    
    parser = argparse.ArgumentParser(description='Run spatial navigation analysis')
    parser.add_argument('--data', default="path/to/your/data", help='Path to data directory')
    parser.add_argument('--output', default="path/to/your/output", help='Path to output directory')
    
    args = parser.parse_args()
    
    run_full_analysis(args.data, args.output)