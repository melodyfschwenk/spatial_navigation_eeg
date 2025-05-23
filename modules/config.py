#!C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe
# -*- coding: utf-8 -*-

"""
Configuration module for Spatial Navigation EEG Experiment
==========================================================

This module contains the configuration parameters for the experiment,
with enhanced support for BrainVision EEG recording and mu rhythm analysis.
"""

import os
import time
import logging
import random
from modules.eeg import create_detailed_trigger_codes


class Config:
    """Experiment configuration parameters with BrainVision EEG support"""
    def __init__(self):
        # Store experiment start time for precise timing
        self.experiment_start_time = time.time()
        
        # Get base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths
        self.stim_paths = {
            'easy': os.path.join(self.base_dir, 'stimuli', 'easy'),
            'hard': os.path.join(self.base_dir, 'stimuli', 'hard'),
            'control': os.path.join(self.base_dir, 'stimuli', 'control')
        }
        self.stimulus_map_path = os.path.join(self.base_dir, 'config', 'stimulus_mappings.csv')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Experiment parameters
        self.fullscreen = True
        self.screen_resolution = (1920, 1080)
        self.background_color = (1, 1, 1)  # White background
        self.text_color = (-0.7, -0.7, -0.7)  # Dark gray text
        
        # Display settings for stimuli
        self.stimulus_size = (500, 500)  # Size in pixels
        
        # Trial parameters
        self.max_response_time = 3.0  # seconds
        self.feedback_duration = 0.5  # seconds
        self.intertrial_interval = (0.5, 0.8)  # Range in seconds
        self.instruction_duration = 10.0  # seconds per instruction screen
        
        # Condition parameters - UPDATED FOR BLOCK STRUCTURE
        self.navigation_types = ['egocentric', 'allocentric', 'control']
        self.difficulty_levels = ['easy', 'hard']
        
        # Define repetitions (number of blocks per condition)
        self.repetitions = 2  # Each main condition appears twice
        
        # Define valid combinations of navigation and difficulty
        self.valid_conditions = [
            ('egocentric', 'easy'),
            ('egocentric', 'hard'),
            ('allocentric', 'easy'),
            ('allocentric', 'hard'),
            ('control', 'control')  # Control navigation only has control difficulty
        ]
        
        # Define block counts for each condition (updated block structure)
        self.condition_blocks = {
            'egocentric_easy': 2,    # Reduced from 3 to 2
            'egocentric_hard': 2,    # Reduced from 3 to 2
            'allocentric_easy': 2,   # Reduced from 3 to 2
            'allocentric_hard': 2,   # Reduced from 3 to 2
            'control_control': 1     # Reduced from 2 to 1
        }
        
        # For backward compatibility with tuple access format
        for nav_type, difficulty in self.valid_conditions:
            key = f"{nav_type}_{difficulty}"
            if key in self.condition_blocks:
                self.condition_blocks[(nav_type, difficulty)] = self.condition_blocks[key]
        
        self.trials_per_block = 15        # Reduced from 20 to 15
        
        
        # Add practice trials setting (number of practice trials total)
        self.num_practice_trials = 4  # Reduced from 6 to 4 to stay under 3 minutes
        
        # Calculate total trials based on condition_blocks
        self.total_trials = sum(count * self.trials_per_block 
                               for count in self.condition_blocks.values())
        
        # EEG parameters - general
        self.use_eeg = True
        
        # Enhanced EEG trigger codes
        self.eeg_trigger_codes = create_detailed_trigger_codes()
        
        # BrainVision EEG parameters
        self.use_brainvision_rcs = True  # Use BrainVision Remote Control Server
        self.brainvision_rcs_host = "127.0.0.1"  # Default localhost
        self.brainvision_rcs_port = 6700  # Default RCS port
        self.brainvision_skip_connection_test = False  # Set to True to skip connection testing
        
        # LSL configuration (can work with BrainVision through LSL)
        self.use_lsl = True  # Use Lab Streaming Layer
        self.brainvision_lsl_name = "BrainVision RDA Markers"  # Standard name for BrainVision
        
        # Parallel port (alternative method for BrainVision TriggerBox)
        self.use_parallel = True  # Enable by default for BrainVision TTL triggers
        self.parallel_port_address = 0x378  # Standard LPT1 address - change to 0xD010 if using PCI card
        self.parallel_reset_delay = 0.001  # 1ms delay for trigger reset (adjust based on BrainVision settings)
        
        # TCP marker support (for custom setups)
        self.use_tcp_markers = False  # Disabled by default
        self.tcp_marker_host = "127.0.0.1"
        self.tcp_marker_port = 5678
        
        # Advanced EEG settings
        self.extra_trigger_metadata = True  # Include extra metadata with each trigger
        self.verify_triggers = True  # Verify triggers were sent successfully
        self.save_trigger_timing = True  # Save precise timing information
        
        # Mu rhythm specific parameters
        self.baseline_duration = 0.5  # Baseline period before stimulus in seconds
        self.post_response_record = 1.0  # Time to continue recording after response
        
        # Mu rhythm frequency bands
        self.mu_freq_range = (8, 13)  # Classic mu rhythm range in Hz
        self.beta_freq_range = (13, 30)  # Beta band for motor activity

        # Add monitor selection for dual monitor support
        self.monitor_index = 0  # Change to 1 to use the secondary monitor

        # Define RT bin thresholds
        self.rt_thresholds = {
            'fast': 500,     # < 500ms
            'medium': 1000,  # 500-1000ms
            'slow': float('inf')  # > 1000ms
        }
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        # Check data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Check log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Check stimulus directories
        for path in self.stim_paths.values():
            if not os.path.exists(path):
                os.makedirs(path)
    
    def get_combined_condition_code(self, navigation_type, difficulty):
        """Get the combined condition trigger code
        
        Args:
            navigation_type: 'egocentric' or 'allocentric'
            difficulty: 'easy', 'hard', or 'control'
            
        Returns:
            int: Combined condition trigger code
        """
        # Create condition key
        condition_key = f"{navigation_type[:4]}_{difficulty}_condition"
        
        # Return the code if it exists
        return self.eeg_trigger_codes.get(condition_key, 0)
    
    def get_response_code(self, navigation_type, response_key):
        """Get the combined response trigger code
        
        Args:
            navigation_type: 'egocentric' or 'allocentric'
            response_key: 'up', 'down', 'left', or 'right'
            
        Returns:
            int: Combined response trigger code
        """
        # Create response key
        response_code_key = f"{navigation_type[:4]}_{response_key}_response"
        
        # Return the code if it exists
        return self.eeg_trigger_codes.get(response_code_key, 0)
    
    def get_outcome_code(self, navigation_type, difficulty, is_correct):
        """Get the combined outcome trigger code
        
        Args:
            navigation_type: 'egocentric' or 'allocentric'
            difficulty: 'easy', 'hard', or 'control'
            is_correct: Whether the response was correct
            
        Returns:
            int: Combined outcome trigger code
        """
        # Create outcome key
        outcome_type = 'correct' if is_correct else 'incorrect'
        outcome_key = f"{navigation_type[:4]}_{difficulty}_{outcome_type}"
        
        # Return the code if it exists
        return self.eeg_trigger_codes.get(outcome_key, 0)
    
    def get_direction_code(self, navigation_type, direction):
        """Get the combined direction trigger code
        
        Args:
            navigation_type: 'egocentric' or 'allocentric'
            direction: Direction string (e.g., 'forward', 'north')
            
        Returns:
            int: Combined direction trigger code
        """
        # Create direction key
        direction_key = f"{navigation_type[:4]}_{direction}_direction"
        
        # Return the code if it exists
        return self.eeg_trigger_codes.get(direction_key, 0)
    
    def create_trigger_metadata(self, navigation_type, difficulty, block_num, trial_num,
                               stimulus_id, correct_direction=None, response=None):
        """Create detailed metadata for a trigger
        
        Args:
            navigation_type: 'egocentric' or 'allocentric'
            difficulty: 'easy', 'hard', or 'control'
            block_num: Current block number
            trial_num: Current trial number
            stimulus_id: ID of the current stimulus
            correct_direction: Optional correct response direction
            response: Optional participant response
            
        Returns:
            dict: Metadata dictionary
        """
        if not self.extra_trigger_metadata:
            return None
        
        # Regular task metadata
        metadata = {
            'navigation_type': navigation_type,
            'difficulty': difficulty,
            'block_num': block_num,
            'trial_num': trial_num,
            'stimulus_id': stimulus_id
        }
        
        if correct_direction is not None:
            metadata['correct_direction'] = correct_direction
        
        if response is not None:
            metadata['response'] = response
        
        return metadata

    def get_rt_bin(self, rt):
        """Determine reaction time bin based on RT value
        
        Args:
            rt: Reaction time in milliseconds
            
        Returns:
            str: RT bin category ('fast', 'medium', or 'slow')
        """
        if rt < self.rt_thresholds['fast']:
            return 'fast'
        elif rt < self.rt_thresholds['medium']:
            return 'medium'
        else:
            return 'slow'
    
    def get_rt_bin_code(self, rt_in_sec, is_correct, navigation_type):
        """Get the RT bin trigger code
        
        Args:
            rt_in_sec: Reaction time in seconds
            is_correct: Whether the response was correct
            navigation_type: 'egocentric', 'allocentric', or 'control'
            
        Returns:
            int: RT bin trigger code
        """
        # Convert RT to milliseconds for binning
        rt_ms = rt_in_sec * 1000
        
        # Determine RT bin
        rt_bin = self.get_rt_bin(rt_ms)
        
        # Determine accuracy string
        accuracy = 'correct' if is_correct else 'incorrect'
        
        # Create key for trigger codes dict
        key = f"{rt_bin}_{accuracy}_{navigation_type}"
        
        # Return the code if it exists
        return self.eeg_trigger_codes.get(key, 0)
    
    def get_combined_performance_code(self, rt_in_sec, is_correct, navigation_type, difficulty):
        """Get the combined performance + condition trigger code
        
        Args:
            rt_in_sec: Reaction time in seconds
            is_correct: Whether the response was correct
            navigation_type: 'egocentric', 'allocentric', or 'control'
            difficulty: 'easy', 'hard', or 'control'
            
        Returns:
            int: Combined performance + condition trigger code
        """
        # Convert RT to milliseconds for binning
        rt_ms = rt_in_sec * 1000
        
        # Determine RT bin
        rt_bin = self.get_rt_bin(rt_ms)
        
        # Determine accuracy string
        accuracy = 'correct' if is_correct else 'incorrect'
        
        # Create key for trigger codes dict
        key = f"{rt_bin}_{accuracy}_{navigation_type}_{difficulty}"
        
        # Return the code if it exists
        return self.eeg_trigger_codes.get(key, 0)
    
    def create_block_sequence(self, counterbalance, participant_id=None):
        """Create a sequence of blocks with navigation types and difficulties
        
        Args:
            counterbalance: Counterbalance condition (1-4 for Latin Square design)
            participant_id: Optional participant ID for consistent randomization
            
        Returns:
            list: List of (navigation_type, difficulty, block_num) tuples
        """
        # Convert counterbalance to integer if it's a string
        if isinstance(counterbalance, str):
            try:
                counterbalance = int(counterbalance)
            except ValueError:
                logging.warning(f"Invalid counterbalance value '{counterbalance}', defaulting to 1")
                counterbalance = 1
                
        # Set participant-specific seed for reproducible yet unique randomization
        if participant_id:
            # Create a seed from participant_id
            try:
                seed = int(participant_id)
            except ValueError:
                seed = sum(ord(c) for c in str(participant_id))
                
            # Store current state, set seed, and we'll restore state at the end
            rand_state = random.getstate()
            random.seed(seed)
            logging.info(f"Using participant-specific seed {seed} for block randomization")
        
        # Define all main condition combinations
        main_conditions = [
            ('egocentric', 'easy'),
            ('egocentric', 'hard'),
            ('allocentric', 'easy'),
            ('allocentric', 'hard')
        ]
        
        # Define Latin Square for complete counterbalancing (4 conditions × 4 orders)
        latin_square = [
            # Each row represents a different counterbalance condition
            [0, 1, 2, 3],  # CB1: egocentric_easy → egocentric_hard → allocentric_easy → allocentric_hard
            [1, 3, 0, 2],  # CB2: egocentric_hard → allocentric_hard → egocentric_easy → allocentric_easy
            [2, 0, 3, 1],  # CB3: allocentric_easy → egocentric_easy → allocentric_hard → egocentric_hard
            [3, 2, 1, 0]   # CB4: allocentric_hard → allocentric_easy → egocentric_hard → egocentric_easy
        ]
        
        # Ensure counterbalance is valid (1-4)
        if counterbalance < 1 or counterbalance > 4:
            logging.warning(f"Invalid counterbalance condition {counterbalance}, defaulting to 1")
            counterbalance = 1
        
        # Get the condition order for this counterbalance condition (adjust for 0-indexing)
        condition_order = latin_square[counterbalance - 1]
        
        # Implement round-robin distribution based on Latin Square
        # First, get the ordered conditions from the Latin Square
        ordered_conditions = [main_conditions[idx] for idx in condition_order]
        
        # Create blocks with round-robin distribution
        all_ordered_blocks = []
        
        # Track the number of blocks we need to add for each condition
        condition_blocks_needed = {}
        for nav_type, difficulty in self.valid_conditions:
            if nav_type != 'control':  # Skip control blocks for now
                key = (nav_type, difficulty)
                condition_blocks_needed[key] = self.condition_blocks.get(key, 0)
        
        # Use round-robin approach to distribute blocks
        # This prevents consecutive repetition of the same condition
        while sum(condition_blocks_needed.values()) > 0:
            for nav_type, difficulty in ordered_conditions:
                key = (nav_type, difficulty)
                if condition_blocks_needed.get(key, 0) > 0:
                    all_ordered_blocks.append(key)
                    condition_blocks_needed[key] -= 1
        
        # Get control blocks separately
        control_blocks = []
        for _ in range(self.condition_blocks.get(('control', 'control'), 0)):
            control_blocks.append(('control', 'control'))
        
        # Insert control blocks at approximately equidistant positions with random offset
        if control_blocks:
            n_controls = len(control_blocks)
            n_main = len(all_ordered_blocks)
            
            for i, control_block in enumerate(control_blocks):
                # Calculate position based on equal distribution
                # Add random jitter to the position (±1 position) as specified in methods
                target_pos = int((i + 1) * (n_main / (n_controls + 1)))
                jitter = random.randint(-1, 1)
                insert_pos = max(0, min(n_main, target_pos + jitter))
                
                # Insert control block
                all_ordered_blocks.insert(insert_pos, control_block)
        
        # Assign block numbers
        block_sequence = []
        for i, (nav_type, difficulty) in enumerate(all_ordered_blocks, 1):
            block_sequence.append((nav_type, difficulty, i))
        
        # Log the counterbalanced block sequence
        sequence_summary = [f"Block {b[2]}: {b[0]}/{b[1]}" for b in block_sequence]
        logging.info(f"Created Latin square sequence with round-robin distribution (CB{counterbalance}) for participant {participant_id}: {', '.join(sequence_summary)}")
        
        # Restore random state if we changed it
        if participant_id:
            random.setstate(rand_state)
        
        return block_sequence