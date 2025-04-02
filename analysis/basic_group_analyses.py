"""
Basic Group Comparison Analyses for Spatial Navigation Study
===========================================================

This module provides simple, reliable analyses for comparing navigation performance
and EEG patterns across deaf/hearing and signing/non-signing groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import mne
from mne.stats import permutation_cluster_test


class GroupAnalyses:
    """
    Class for performing basic group comparisons in spatial navigation data.
    """
    
    def __init__(self, dataset):
        """
        Initialize the group analyses.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset containing all participant data
        """
        self.dataset = dataset
        
        # Add group columns if they don't exist
        if 'hearing_status' not in self.dataset.main_data.columns:
            self.add_group_columns()
    
    def add_group_columns(self):
        """
        Add columns for hearing status and signing fluency based on participant metadata.
        This assumes the metadata contains this information.
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return
        
        # Check if metadata contains group information
        if not hasattr(self.dataset, 'metadata') or 'participants' not in self.dataset.metadata:
            print("Metadata does not contain participant information")
            return
        
        # Create a mapping from participant ID to group
        group_mapping = {}
        for participant_id, info in self.dataset.metadata['participants'].items():
            # Determine hearing status (deaf/hearing)
            if 'hearing_status' in info:
                is_deaf = info['hearing_status'].lower() == 'deaf'
            else:
                print(f"Warning: hearing_status not found for participant {participant_id}")
                continue
            
            # Determine signing fluency (fluent/non-fluent/non-signer)
            if 'signing_status' in info:
                signing_status = info['signing_status'].lower()
                is_fluent = signing_status == 'fluent'
                is_signer = signing_status in ['fluent', 'non-fluent']
            else:
                print(f"Warning: signing_status not found for participant {participant_id}")
                continue
            
            # Store in mapping
            group_mapping[participant_id] = {
                'is_deaf': is_deaf,
                'is_fluent': is_fluent,
                'is_signer': is_signer
            }
        
        # Add columns to main_data
        self.dataset.main_data['hearing_status'] = self.dataset.main_data['participant_id'].map(
            lambda x: 'deaf' if group_mapping.get(x, {}).get('is_deaf', False) else 'hearing'
        )
        
        self.dataset.main_data['signing_status'] = self.dataset.main_data['participant_id'].map(
            lambda x: 'fluent' if group_mapping.get(x, {}).get('is_fluent', False) else 
                     ('non-fluent' if group_mapping.get(x, {}).get('is_signer', False) else 'non-signer')
        )
        
        # Create combined group column
        self.dataset.main_data['group'] = self.dataset.main_data.apply(
            lambda row: f"{row['hearing_status']}_{row['signing_status']}",
            axis=1
        )
        
        # Add condition column if it doesn't exist
        if 'condition' not in self.dataset.main_data.columns:
            self.dataset.main_data['condition'] = self.dataset.main_data.apply(
                lambda row: f"{row['navigation_type']}_{row['difficulty']}",
                axis=1
            )
        
        print("Added group and condition columns to dataset")
    
    def compare_accuracy_by_group(self):
        """
        Compare accuracy across groups.
        
        Returns
        -------
        pandas.DataFrame
            Accuracy by group
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Calculate mean accuracy by participant and group
        accuracy_by_participant = self.dataset.main_data.groupby(['participant_id', 'hearing_status', 'signing_status', 'group'])['accuracy'].mean().reset_index()
        
        # Calculate group means and SEMs
        group_stats = accuracy_by_participant.groupby('group')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        group_stats['sem'] = group_stats['std'] / np.sqrt(group_stats['count'])
        
        return group_stats
    
    def compare_rt_by_group(self):
        """
        Compare reaction time across groups.
        
        Returns
        -------
        pandas.DataFrame
            Reaction time by group
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Calculate mean RT by participant and group
        rt_by_participant = self.dataset.main_data.groupby(['participant_id', 'hearing_status', 'signing_status', 'group'])['rt'].mean().reset_index()
        
        # Calculate group means and SEMs
        group_stats = rt_by_participant.groupby('group')['rt'].agg(['mean', 'std', 'count']).reset_index()
        group_stats['sem'] = group_stats['std'] / np.sqrt(group_stats['count'])
        
        return group_stats
    
    def run_two_way_anova(self, dv='accuracy'):
        """
        Run a two-way ANOVA with hearing status and signing status as factors.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
            
        Returns
        -------
        statsmodels.anova.anova_lm
            ANOVA results
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Calculate mean DV by participant and factors
        data = self.dataset.main_data.groupby(['participant_id', 'hearing_status', 'signing_status'])[dv].mean().reset_index()
        
        # Run two-way ANOVA
        formula = f"{dv} ~ C(hearing_status) * C(signing_status)"
        model = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        return anova_table
    
    def compare_navigation_types_by_group(self, dv='accuracy'):
        """
        Compare performance on different navigation types across groups.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
            
        Returns
        -------
        pandas.DataFrame
            Performance by navigation type and group
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Calculate mean performance by participant, group, and navigation type
        perf_by_nav_type = self.dataset.main_data.groupby(['participant_id', 'group', 'navigation_type'])[dv].mean().reset_index()
        
        # Calculate group means and SEMs
        group_nav_stats = perf_by_nav_type.groupby(['group', 'navigation_type'])[dv].agg(['mean', 'std', 'count']).reset_index()
        group_nav_stats['sem'] = group_nav_stats['std'] / np.sqrt(group_nav_stats['count'])
        
        return group_nav_stats
    
    def run_mixed_anova(self, dv='accuracy'):
        """
        Run a mixed ANOVA with group as between-subjects factor and navigation type as within-subjects factor.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
            
        Returns
        -------
        tuple
            AnovaResults objects for between and within effects
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None, None
        
        # Calculate mean DV by participant, group, and navigation type
        data = self.dataset.main_data.groupby(['participant_id', 'group', 'navigation_type'])[dv].mean().reset_index()
        
        # Ensure data is balanced (each participant has all navigation types)
        pivot_data = data.pivot(index=['participant_id', 'group'], columns='navigation_type', values=dv)
        
        # Check for missing values
        if pivot_data.isna().any().any():
            print("Warning: Missing data for some participants. Mixed ANOVA requires complete data.")
            # Drop participants with missing data
            pivot_data = pivot_data.dropna()
            
        # Reset index and melt back to long format
        balanced_data = pivot_data.reset_index().melt(
            id_vars=['participant_id', 'group'],
            value_vars=pivot_data.columns,
            var_name='navigation_type',
            value_name=dv
        )
        
        try:
            # Run mixed ANOVA using statsmodels
            formula = f"{dv} ~ group * navigation_type"
            model = ols(formula, data=balanced_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            return anova_table
        except Exception as e:
            print(f"Error running mixed ANOVA: {e}")
            return None
    
    def run_post_hoc_tests(self, dv='accuracy'):
        """
        Run post-hoc t-tests between groups.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
            
        Returns
        -------
        pandas.DataFrame
            Post-hoc test results
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Calculate mean DV by participant and group
        data = self.dataset.main_data.groupby(['participant_id', 'group'])[dv].mean().reset_index()
        
        # Get unique groups
        groups = data['group'].unique()
        
        # Create results DataFrame
        results = pd.DataFrame(columns=['group1', 'group2', 't_stat', 'p_value', 'sig'])
        
        # Run t-tests for all pairs of groups
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                # Get data for each group
                group1_data = data[data['group'] == group1][dv]
                group2_data = data[data['group'] == group2][dv]
                
                # Run t-test
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                
                # Add to results
                results = pd.concat([results, pd.DataFrame({
                    'group1': [group1],
                    'group2': [group2],
                    't_stat': [t_stat],
                    'p_value': [p_value],
                    'sig': [p_value < 0.05]
                })], ignore_index=True)
        
        # Sort by p-value
        results = results.sort_values('p_value')
        
        return results
    
    def compare_learning_effects(self, dv='accuracy'):
        """
        Compare learning effects (performance changes over blocks) across groups.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
            
        Returns
        -------
        pandas.DataFrame
            Performance by block and group
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Ensure there's a block column
        if 'block' not in self.dataset.main_data.columns:
            if 'session' in self.dataset.main_data.columns:
                # Create block from session
                blocks_per_session = self.dataset.metadata.get('blocks_per_session', 1)
                self.dataset.main_data['block'] = (self.dataset.main_data['session'] // blocks_per_session) + 1
            else:
                print("No block or session information available")
                return None
        
        # Calculate mean performance by participant, group, and block
        perf_by_block = self.dataset.main_data.groupby(['participant_id', 'group', 'block'])[dv].mean().reset_index()
        
        # Calculate group means and SEMs
        group_block_stats = perf_by_block.groupby(['group', 'block'])[dv].agg(['mean', 'std', 'count']).reset_index()
        group_block_stats['sem'] = group_block_stats['std'] / np.sqrt(group_block_stats['count'])
        
        return group_block_stats
    
    def compare_difficulty_effects(self, dv='accuracy'):
        """
        Compare effects of difficulty across groups.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
            
        Returns
        -------
        pandas.DataFrame
            Performance by difficulty and group
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Calculate mean performance by participant, group, and difficulty
        perf_by_difficulty = self.dataset.main_data.groupby(['participant_id', 'group', 'difficulty'])[dv].mean().reset_index()
        
        # Calculate group means and SEMs
        group_diff_stats = perf_by_difficulty.groupby(['group', 'difficulty'])[dv].agg(['mean', 'std', 'count']).reset_index()
        group_diff_stats['sem'] = group_diff_stats['std'] / np.sqrt(group_diff_stats['count'])
        
        return group_diff_stats
    
    def calculate_switch_costs_by_group(self):
        """
        Calculate switch costs between navigation types for each group.
        
        Returns
        -------
        pandas.DataFrame
            Switch costs by group
        """
        if self.dataset.main_data is None:
            print("No data loaded")
            return None
        
        # Create a copy of the data
        data = self.dataset.main_data.copy().sort_values(['participant_id', 'session'])
        
        # Identify switches
        data['prev_nav_type'] = data.groupby('participant_id')['navigation_type'].shift(1)
        data['is_switch'] = (data['navigation_type'] != data['prev_nav_type']).astype(int)
        
        # For the first trial of each participant, set is_switch to NaN
        first_trials = data.groupby('participant_id').head(1).index
        data.loc[first_trials, 'is_switch'] = np.nan
        
        # Calculate mean RT for switch and non-switch trials by participant
        switch_costs = data.groupby(['participant_id', 'group', 'is_switch'])['rt'].mean().reset_index()
        
        # Reshape to have one row per participant with switch and non-switch RT
        switch_costs = switch_costs.pivot(index=['participant_id', 'group'], columns='is_switch', values='rt')
        switch_costs.columns = ['non_switch_rt', 'switch_rt']
        
        # Calculate the switch cost (difference in RT)
        switch_costs['switch_cost'] = switch_costs['switch_rt'] - switch_costs['non_switch_rt']
        
        # Reset index
        switch_costs = switch_costs.reset_index()
        
        # Calculate group means and SEMs
        group_switch_stats = switch_costs.groupby('group')['switch_cost'].agg(['mean', 'std', 'count']).reset_index()
        group_switch_stats['sem'] = group_switch_stats['std'] / np.sqrt(group_switch_stats['count'])
        
        return group_switch_stats


class GroupEEGAnalyses:
    """
    Class for performing basic EEG analyses comparing groups.
    """
    
    def __init__(self, dataset, eeg_preprocessing):
        """
        Initialize the group EEG analyses.
        
        Parameters
        ----------
        dataset : SpatialNavDataset
            The dataset containing all participant data
        eeg_preprocessing : EEGPreprocessing
            The EEG preprocessing object with loaded data
        """
        self.dataset = dataset
        self.eeg = eeg_preprocessing
        
        # Add group information to epochs if not already there
        self.add_group_info_to_epochs()
    
    def add_group_info_to_epochs(self):
        """
        Add group information to epochs metadata.
        """
        # Check if main data has group columns
        if self.dataset.main_data is None or 'group' not in self.dataset.main_data.columns:
            group_analysis = GroupAnalyses(self.dataset)
            group_analysis.add_group_columns()
        
        # Create participant to group mapping
        participant_groups = self.dataset.main_data.groupby('participant_id')['group'].first().to_dict()
        
        # Add group info to epochs
        for participant_id, epochs in self.eeg.epochs.items():
            if participant_id in participant_groups:
                group = participant_groups[participant_id]
                
                # Add to metadata
                for i in range(len(epochs)):
                    epochs.metadata.loc[i, 'group'] = group
    
    def compare_erp_by_group(self, time_window=(0.1, 0.3), channels=None):
        """
        Compare ERPs across groups.
        
        Parameters
        ----------
        time_window : tuple
            The time window to analyze (start, end) in seconds
        channels : list, optional
            List of channels to include
            
        Returns
        -------
        dict
            ERP data by group
        """
        if not self.eeg.epochs:
            print("No epochs data loaded")
            return None
        
        # Initialize results
        erp_data = {}
        
        # Get all participant IDs with epochs
        participant_ids = list(self.eeg.epochs.keys())
        
        # Group participants by group
        participant_groups = {}
        for participant_id in participant_ids:
            if participant_id in self.dataset.main_data['participant_id'].values:
                group = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]['group'].iloc[0]
                if group not in participant_groups:
                    participant_groups[group] = []
                participant_groups[group].append(participant_id)
        
        # Calculate ERPs for each group
        for group, group_participants in participant_groups.items():
            # Average epochs within each participant
            participant_evokeds = []
            
            for participant_id in group_participants:
                epochs = self.eeg.epochs[participant_id]
                
                # Select channels if specified
                if channels is not None:
                    epochs = epochs.copy().pick_channels(channels)
                
                # Average epochs to get evoked response
                evoked = epochs.average()
                participant_evokeds.append(evoked)
            
            # Grand average across participants
            if participant_evokeds:
                grand_avg = mne.grand_average(participant_evokeds)
                
                # Extract data for the time window
                times = grand_avg.times
                time_mask = (times >= time_window[0]) & (times <= time_window[1])
                
                erp_data[group] = {
                    'times': times[time_mask],
                    'data': grand_avg.data[:, time_mask],
                    'info': grand_avg.info,
                    'participant_count': len(participant_evokeds)
                }
        
        return erp_data
    
    def compare_frequency_power_by_group(self, frequency_band=(8, 12), channels=None):
        """
        Compare frequency power across groups.
        
        Parameters
        ----------
        frequency_band : tuple
            The frequency band to analyze (min, max) in Hz
        channels : list, optional
            List of channels to include
            
        Returns
        -------
        dict
            Frequency power data by group
        """
        if not self.eeg.epochs:
            print("No epochs data loaded")
            return None
        
        # Initialize results
        power_data = {}
        
        # Get all participant IDs with epochs
        participant_ids = list(self.eeg.epochs.keys())
        
        # Group participants by group
        participant_groups = {}
        for participant_id in participant_ids:
            if participant_id in self.dataset.main_data['participant_id'].values:
                group = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]['group'].iloc[0]
                if group not in participant_groups:
                    participant_groups[group] = []
                participant_groups[group].append(participant_id)
        
        # Calculate power for each group
        for group, group_participants in participant_groups.items():
            # Average power within each participant
            participant_powers = []
            
            for participant_id in group_participants:
                epochs = self.eeg.epochs[participant_id]
                
                # Select channels if specified
                if channels is not None:
                    epochs = epochs.copy().pick_channels(channels)
                
                # Calculate PSD
                psds, freqs = mne.time_frequency.psd_welch(
                    epochs,
                    fmin=frequency_band[0],
                    fmax=frequency_band[1],
                    n_fft=256,
                    n_overlap=128
                )
                
                # Average across epochs
                participant_power = psds.mean(axis=0)
                participant_powers.append(participant_power)
            
            # Average across participants
            if participant_powers:
                group_power = np.stack(participant_powers).mean(axis=0)
                
                power_data[group] = {
                    'freqs': freqs,
                    'power': group_power,
                    'participant_count': len(participant_powers)
                }
        
        return power_data
    
    def compare_mu_rhythm_by_group(self, channels=None):
        """
        Compare mu rhythm (8-12 Hz) power across groups.
        
        Parameters
        ----------
        channels : list, optional
            List of channels to include
            
        Returns
        -------
        dict
            Mu rhythm power by group
        """
        # Mu rhythm is typically 8-12 Hz
        return self.compare_frequency_power_by_group(frequency_band=(8, 12), channels=channels)
    
    def compare_frontal_theta_by_group(self, channels=None):
        """
        Compare frontal theta (4-8 Hz) power across groups.
        
        Parameters
        ----------
        channels : list, optional
            List of channels to include (defaults to frontal channels)
            
        Returns
        -------
        dict
            Frontal theta power by group
        """
        # Default to frontal channels if not specified
        if channels is None:
            # Typical frontal channels
            channels = ['Fz', 'F3', 'F4', 'FC1', 'FC2']
        
        # Theta is typically 4-8 Hz
        return self.compare_frequency_power_by_group(frequency_band=(4, 8), channels=channels)
    
    def compare_alpha_lateralization_by_group(self):
        """
        Compare alpha lateralization across groups.
        
        Returns
        -------
        dict
            Alpha lateralization by group
        """
        if not self.eeg.epochs:
            print("No epochs data loaded")
            return None
        
        # Initialize results
        lat_data = {}
        
        # Get all participant IDs with epochs
        participant_ids = list(self.eeg.epochs.keys())
        
        # Group participants by group
        participant_groups = {}
        for participant_id in participant_ids:
            if participant_id in self.dataset.main_data['participant_id'].values:
                group = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]['group'].iloc[0]
                if group not in participant_groups:
                    participant_groups[group] = []
                participant_groups[group].append(participant_id)
        
        # Calculate lateralization for each group
        for group, group_participants in participant_groups.items():
            # Calculate lateralization for each participant
            participant_lat = []
            
            for participant_id in group_participants:
                epochs = self.eeg.epochs[participant_id]
                
                # Define left and right hemisphere channels
                # This is a simplification - adjust based on your electrode montage
                left_channels = ['P3', 'P7', 'PO3', 'O1']
                right_channels = ['P4', 'P8', 'PO4', 'O2']
                
                # Calculate PSD for left and right channels
                left_psds, freqs = mne.time_frequency.psd_welch(
                    epochs.copy().pick_channels(left_channels, ordered=True),
                    fmin=8,
                    fmax=12,  # Alpha band
                    n_fft=256,
                    n_overlap=128
                )
                
                right_psds, _ = mne.time_frequency.psd_welch(
                    epochs.copy().pick_channels(right_channels, ordered=True),
                    fmin=8,
                    fmax=12,  # Alpha band
                    n_fft=256,
                    n_overlap=128
                )
                
                # Average across channels and epochs
                left_power = left_psds.mean(axis=(0, 1))
                right_power = right_psds.mean(axis=(0, 1))
                
                # Calculate lateralization index: (R - L) / (R + L)
                lat_index = (right_power - left_power) / (right_power + left_power)
                
                participant_lat.append(np.mean(lat_index))  # Average across frequencies
            
            # Calculate group statistics
            if participant_lat:
                lat_data[group] = {
                    'lateralization_index': participant_lat,
                    'mean': np.mean(participant_lat),
                    'std': np.std(participant_lat),
                    'sem': np.std(participant_lat) / np.sqrt(len(participant_lat)),
                    'participant_count': len(participant_lat)
                }
        
        return lat_data
    
    def compare_erp_by_condition_and_group(self, condition_column='navigation_type', time_window=(0.1, 0.3), channels=None):
        """
        Compare ERPs across conditions and groups.
        
        Parameters
        ----------
        condition_column : str
            The column to use for conditions
        time_window : tuple
            The time window to analyze (start, end) in seconds
        channels : list, optional
            List of channels to include
            
        Returns
        -------
        dict
            ERP data by condition and group
        """
        if not self.eeg.epochs:
            print("No epochs data loaded")
            return None
        
        # Initialize results
        erp_data = {}
        
        # Get all participant IDs with epochs
        participant_ids = list(self.eeg.epochs.keys())
        
        # Get condition values
        condition_values = self.dataset.main_data[condition_column].unique()
        
        # Group participants by group
        participant_groups = {}
        for participant_id in participant_ids:
            if participant_id in self.dataset.main_data['participant_id'].values:
                group = self.dataset.main_data[self.dataset.main_data['participant_id'] == participant_id]['group'].iloc[0]
                if group not in participant_groups:
                    participant_groups[group] = []
                participant_groups[group].append(participant_id)
        
        # Calculate ERPs for each group and condition
        for group, group_participants in participant_groups.items():
            erp_data[group] = {}
            
            for condition in condition_values:
                # Average epochs within each participant for this condition
                participant_evokeds = []
                
                for participant_id in group_participants:
                    epochs = self.eeg.epochs[participant_id]
                    
                    # Select epochs for this condition
                    condition_epochs = epochs[condition_column + " == '" + condition + "'"]
                    
                    # If no epochs match, continue
                    if len(condition_epochs) == 0:
                        continue
                    
                    # Select channels if specified
                    if channels is not None:
                        condition_epochs = condition_epochs.copy().pick_channels(channels)
                    
                    # Average epochs to get evoked response
                    evoked = condition_epochs.average()
                    participant_evokeds.append(evoked)
                
                # Grand average across participants
                if participant_evokeds:
                    grand_avg = mne.grand_average(participant_evokeds)
                    
                    # Extract data for the time window
                    times = grand_avg.times
                    time_mask = (times >= time_window[0]) & (times <= time_window[1])
                    
                    erp_data[group][condition] = {
                        'times': times[time_mask],
                        'data': grand_avg.data[:, time_mask],
                        'info': grand_avg.info,
                        'participant_count': len(participant_evokeds)
                    }
        
        return erp_data


class GroupVisualization:
    """
    Class for creating visualizations comparing groups.
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
        
        # Set up colors for consistent group plotting
        self.group_colors = {
            'deaf_fluent': '#1f77b4',
            'deaf_non-fluent': '#ff7f0e',
            'hearing_fluent': '#2ca02c',
            'hearing_non-fluent': '#d62728',
            'hearing_non-signer': '#9467bd'
        }
    
    def plot_accuracy_by_group(self, save_path=None):
        """
        Plot accuracy by group.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        group_analysis = GroupAnalyses(self.dataset)
        accuracy_data = group_analysis.compare_accuracy_by_group()
        
        if accuracy_data is None:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(
            accuracy_data['group'],
            accuracy_data['mean'],
            yerr=accuracy_data['sem'],
            capsize=10,
            color=[self.group_colors.get(group, 'gray') for group in accuracy_data['group']]
        )
        
        # Add labels and title
        plt.xlabel('Group')
        plt.ylabel('Accuracy')
        plt.title('Navigation Accuracy by Group')
        plt.ylim(0, 1)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_rt_by_group(self, save_path=None):
        """
        Plot reaction time by group.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        group_analysis = GroupAnalyses(self.dataset)
        rt_data = group_analysis.compare_rt_by_group()
        
        if rt_data is None:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(
            rt_data['group'],
            rt_data['mean'],
            yerr=rt_data['sem'],
            capsize=10,
            color=[self.group_colors.get(group, 'gray') for group in rt_data['group']]
        )
        
        # Add labels and title
        plt.xlabel('Group')
        plt.ylabel('Reaction Time (ms)')
        plt.title('Navigation Response Time by Group')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_navigation_types_by_group(self, dv='accuracy', save_path=None):
        """
        Plot performance on different navigation types by group.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        group_analysis = GroupAnalyses(self.dataset)
        nav_data = group_analysis.compare_navigation_types_by_group(dv)
        
        if nav_data is None:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Convert to wide format for grouped bar plot
        plot_data = nav_data.pivot(index='group', columns='navigation_type', values='mean')
        error_data = nav_data.pivot(index='group', columns='navigation_type', values='sem')
        
        # Create grouped bar plot
        ax = plot_data.plot(
            kind='bar',
            yerr=error_data,
            capsize=5,
            color=['#1f77b4', '#ff7f0e'],
            figsize=(12, 6)
        )
        
        # Add labels and title
        plt.xlabel('Group')
        plt.ylabel(dv.capitalize())
        plt.title(f'{dv.capitalize()} by Navigation Type and Group')
        
        if dv == 'accuracy':
            plt.ylim(0, 1)
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title='Navigation Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_learning_effects(self, dv='accuracy', save_path=None):
        """
        Plot learning effects (performance over blocks) by group.
        
        Parameters
        ----------
        dv : str
            The dependent variable ('accuracy' or 'rt')
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        group_analysis = GroupAnalyses(self.dataset)
        learning_data = group_analysis.compare_learning_effects(dv)
        
        if learning_data is None:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Get unique groups
        groups = learning_data['group'].unique()
        
        # Plot line for each group
        for group in groups:
            group_data = learning_data[learning_data['group'] == group]
            plt.errorbar(
                group_data['block'],
                group_data['mean'],
                yerr=group_data['sem'],
                marker='o',
                label=group,
                color=self.group_colors.get(group, None)
            )
        
        # Add labels and title
        plt.xlabel('Block')
        plt.ylabel(dv.capitalize())
        plt.title(f'Learning Effects: {dv.capitalize()} over Blocks by Group')
        
        if dv == 'accuracy':
            plt.ylim(0, 1)
        
        plt.grid(True, alpha=0.3)
        plt.legend(title='Group')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_switch_costs(self, save_path=None):
        """
        Plot switch costs by group.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        group_analysis = GroupAnalyses(self.dataset)
        switch_data = group_analysis.calculate_switch_costs_by_group()
        
        if switch_data is None:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(
            switch_data['group'],
            switch_data['mean'],
            yerr=switch_data['sem'],
            capsize=10,
            color=[self.group_colors.get(group, 'gray') for group in switch_data['group']]
        )
        
        # Add labels and title
        plt.xlabel('Group')
        plt.ylabel('Switch Cost (ms)')
        plt.title('Navigation Strategy Switch Costs by Group')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_group_erp_comparison(self, erp_data, channel_name=None, save_path=None):
        """
        Plot ERP comparison across groups.
        
        Parameters
        ----------
        erp_data : dict
            ERP data by group from GroupEEGAnalyses.compare_erp_by_group
        channel_name : str, optional
            Name of the channel to plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if erp_data is None or not erp_data:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # If channel_name is not specified, use the first channel
        if channel_name is None:
            for group in erp_data:
                channel_name = erp_data[group]['info']['ch_names'][0]
                break
        
        # Plot ERP for each group
        for group, data in erp_data.items():
            # Find channel index
            try:
                ch_idx = data['info']['ch_names'].index(channel_name)
            except ValueError:
                print(f"Channel {channel_name} not found for group {group}")
                continue
            
            # Plot ERP for this channel
            plt.plot(
                data['times'],
                data['data'][ch_idx],
                label=f"{group} (n={data['participant_count']})",
                color=self.group_colors.get(group, None)
            )
        
        # Add labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.title(f'ERP Comparison at {channel_name} by Group')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Group')
        
        # Add vertical line at time 0
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add horizontal line at amplitude 0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_frequency_power_comparison(self, power_data, channel_name=None, save_path=None):
        """
        Plot frequency power comparison across groups.
        
        Parameters
        ----------
        power_data : dict
            Power data by group from GroupEEGAnalyses.compare_frequency_power_by_group
        channel_name : str, optional
            Name of the channel to plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if power_data is None or not power_data:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Plot power for each group
        for group, data in power_data.items():
            # If channel_name is specified, find the channel index
            # Otherwise, average across all channels
            if channel_name is not None:
                # This assumes power data includes channel names
                ch_names = data.get('ch_names', [f"Channel {i}" for i in range(data['power'].shape[0])])
                try:
                    ch_idx = ch_names.index(channel_name)
                    power = data['power'][ch_idx]
                except (ValueError, IndexError):
                    print(f"Channel {channel_name} not found for group {group}")
                    continue
            else:
                # Average across all channels
                power = data['power'].mean(axis=0) if data['power'].ndim > 1 else data['power']
            
            # Plot power spectrum
            plt.plot(
                data['freqs'],
                power,
                label=f"{group} (n={data['participant_count']})",
                color=self.group_colors.get(group, None)
            )
        
        # Add labels and title
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (µV²/Hz)')
        title = f'Power Spectrum at {channel_name} by Group' if channel_name else 'Power Spectrum by Group'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Group')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_alpha_lateralization(self, lat_data, save_path=None):
        """
        Plot alpha lateralization comparison across groups.
        
        Parameters
        ----------
        lat_data : dict
            Lateralization data by group from GroupEEGAnalyses.compare_alpha_lateralization_by_group
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if lat_data is None or not lat_data:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        groups = []
        means = []
        sems = []
        
        for group, data in lat_data.items():
            groups.append(group)
            means.append(data['mean'])
            sems.append(data['sem'])
        
        # Create bar plot
        bars = plt.bar(
            groups,
            means,
            yerr=sems,
            capsize=10,
            color=[self.group_colors.get(group, 'gray') for group in groups]
        )
        
        # Add labels and title
        plt.xlabel('Group')
        plt.ylabel('Alpha Lateralization Index\n(R-L)/(R+L)')
        plt.title('Alpha Lateralization by Group')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add horizontal line at 0 (no lateralization)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01 if height > 0 else height - 0.02,
                f'{height:.2f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_topography_by_group(self, erp_data, time_point=0.2, save_path=None):
        """
        Plot topography comparison across groups.
        
        Parameters
        ----------
        erp_data : dict
            ERP data by group from GroupEEGAnalyses.compare_erp_by_group
        time_point : float
            Time point to plot in seconds
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if erp_data is None or not erp_data:
            return None
        
        # Calculate number of groups for grid layout
        n_groups = len(erp_data)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if n_groups == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # Plot topography for each group
        for i, (group, data) in enumerate(erp_data.items()):
            # Find time index closest to time_point
            time_idx = np.abs(data['times'] - time_point).argmin()
            
            # Extract data at the time point
            topo_data = data['data'][:, time_idx]
            
            # Plot topography
            if i < len(axes):
                mne.viz.plot_topomap(
                    topo_data,
                    data['info'],
                    axes=axes[i],
                    show=False,
                    contours=6,
                    cmap='RdBu_r'
                )
                axes[i].set_title(f"{group} (n={data['participant_count']})")
        
        # Remove any unused axes
        for i in range(n_groups, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Topography at {time_point}s by Group')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def run_basic_group_analysis(data_dir, output_dir=None):
    """
    Run basic group analysis on spatial navigation EEG data.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing all data files
    output_dir : str, optional
        Path to the directory to save output files
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = SpatialNavDataset(data_dir)
    dataset.summarize()
    
    # Initialize group analysis
    group_analysis = GroupAnalyses(dataset)
    vis = GroupVisualization(dataset)
    
    print("\n=== Basic Group Analysis ===")
    
    # Accuracy by group
    accuracy_data = group_analysis.compare_accuracy_by_group()
    print("\nAccuracy by group:")
    print(accuracy_data)
    
    # RT by group
    rt_data = group_analysis.compare_rt_by_group()
    print("\nResponse time by group:")
    print(rt_data)
    
    # Two-way ANOVA (Hearing Status x Signing Status)
    anova_accuracy = group_analysis.run_two_way_anova('accuracy')
    print("\nTwo-way ANOVA (Accuracy):")
    print(anova_accuracy)
    
    anova_rt = group_analysis.run_two_way_anova('rt')
    print("\nTwo-way ANOVA (RT):")
    print(anova_rt)
    
    # Navigation types by group
    nav_accuracy = group_analysis.compare_navigation_types_by_group('accuracy')
    print("\nAccuracy by navigation type and group:")
    print(nav_accuracy)
    
    # Mixed ANOVA (Group x Navigation Type)
    mixed_anova = group_analysis.run_mixed_anova('accuracy')
    print("\nMixed ANOVA (Group x Navigation Type):")
    print(mixed_anova)
    
    # Switch costs by group
    switch_costs = group_analysis.calculate_switch_costs_by_group()
    print("\nSwitch costs by group:")
    print(switch_costs)
    
    # Post-hoc tests
    post_hoc = group_analysis.run_post_hoc_tests('accuracy')
    print("\nPost-hoc tests (Accuracy):")
    print(post_hoc)
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    
    if output_dir is not None:
        vis.plot_accuracy_by_group(os.path.join(output_dir, 'accuracy_by_group.png'))
        vis.plot_rt_by_group(os.path.join(output_dir, 'rt_by_group.png'))
        vis.plot_navigation_types_by_group('accuracy', os.path.join(output_dir, 'nav_type_accuracy.png'))
        vis.plot_navigation_types_by_group('rt', os.path.join(output_dir, 'nav_type_rt.png'))
        vis.plot_learning_effects('accuracy', os.path.join(output_dir, 'learning_effects.png'))
        vis.plot_switch_costs(os.path.join(output_dir, 'switch_costs.png'))
    else:
        vis.plot_accuracy_by_group()
        vis.plot_rt_by_group()
        vis.plot_navigation_types_by_group('accuracy')
        vis.plot_navigation_types_by_group('rt')
        vis.plot_learning_effects('accuracy')
        vis.plot_switch_costs()
    
    print("\n=== Basic Group Analysis Complete ===")
    if output_dir is not None:
        print(f"Results saved to {output_dir}")


def run_basic_eeg_analysis(data_dir, output_dir=None):
    """
    Run basic EEG analysis comparing groups on spatial navigation EEG data.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing all data files
    output_dir : str, optional
        Path to the directory to save output files
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = SpatialNavDataset(data_dir)
    dataset.summarize()
    
    # Initialize EEG preprocessing
    eeg = EEGPreprocessing(dataset)
    
    # Load EEG data for each participant
    participant_ids = dataset.main_data['participant_id'].unique()
    
    print("\n=== Loading EEG Data ===")
    for participant_id in participant_ids:
        print(f"Loading data for participant {participant_id}...")
        eeg.load_raw_eeg(participant_id)
        eeg.extract_epochs(participant_id)
    
    # Initialize group EEG analysis
    group_eeg = GroupEEGAnalyses(dataset, eeg)
    vis = GroupVisualization(dataset)
    
    print("\n=== Basic EEG Group Analysis ===")
    
    # Compare ERPs by group
    erp_data = group_eeg.compare_erp_by_group(time_window=(0.1, 0.3))
    print("\nERP comparison complete")
    
    # Compare mu rhythm by group
    mu_data = group_eeg.compare_mu_rhythm_by_group()
    print("\nMu rhythm comparison complete")
    
    # Compare frontal theta by group
    theta_data = group_eeg.compare_frontal_theta_by_group()
    print("\nFrontal theta comparison complete")
    
    # Compare alpha lateralization by group
    alpha_lat = group_eeg.compare_alpha_lateralization_by_group()
    print("\nAlpha lateralization comparison complete")
    
    # Compare ERPs by navigation type and group
    erp_nav_data = group_eeg.compare_erp_by_condition_and_group(
        condition_column='navigation_type',
        time_window=(0.1, 0.3)
    )
    print("\nERP by navigation type comparison complete")
    
    # Create visualizations
    print("\n=== Creating EEG Visualizations ===")
    
    if output_dir is not None:
        # Plot ERP comparison
        if erp_data:
            vis.plot_group_erp_comparison(
                erp_data,
                channel_name='Cz',
                save_path=os.path.join(output_dir, 'erp_by_group.png')
            )
        
        # Plot power spectrum comparison
        if mu_data:
            vis.plot_frequency_power_comparison(
                mu_data,
                channel_name='Cz',
                save_path=os.path.join(output_dir, 'mu_by_group.png')
            )
        
        # Plot theta power comparison
        if theta_data:
            vis.plot_frequency_power_comparison(
                theta_data,
                save_path=os.path.join(output_dir, 'theta_by_group.png')
            )
        
        # Plot alpha lateralization
        if alpha_lat:
            vis.plot_alpha_lateralization(
                alpha_lat,
                save_path=os.path.join(output_dir, 'alpha_lateralization.png')
            )
        
        # Plot topography
        if erp_data:
            vis.plot_topography_by_group(
                erp_data,
                time_point=0.2,
                save_path=os.path.join(output_dir, 'topography_by_group.png')
            )
    
    print("\n=== Basic EEG Analysis Complete ===")
    if output_dir is not None:
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/data"
    output_dir = "path/to/your/output"
    
    # Run behavioral analysis
    run_basic_group_analysis(data_dir, output_dir)
    
    # Run EEG analysis
    run_basic_eeg_analysis(data_dir, output_dir)