#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stimulus Module for Spatial Navigation EEG Experiment
====================================================

This module handles stimulus loading and management.
"""

import os
import logging
import pandas as pd
import random
import numpy as np
from modules.utils import validate_response_mapping


def load_stimulus_mapping(config):
    """Load the stimulus mapping CSV file
    
    Args:
        config: The experiment configuration object
        
    Returns:
        DataFrame or None: Pandas DataFrame containing stimulus mapping information
    """
    try:
        stimulus_map = pd.read_csv(config.stimulus_map_path)
        logging.info(f"Loaded {len(stimulus_map)} stimuli from mapping file")
        
        # Verify required columns exist
        required_columns = ['stimulus_id', 'difficulty', 'file_path', 
                           'egocentric_correct_response', 'allocentric_correct_response']
        for col in required_columns:
            if col not in stimulus_map.columns:
                logging.error(f"Required column '{col}' not found in stimulus mapping file")
                raise ValueError(f"Required column '{col}' not found in stimulus mapping file")
        
        # Summarize stimuli by difficulty
        for difficulty in config.difficulty_levels:
            count = sum(stimulus_map['difficulty'] == difficulty)
            logging.info(f"Found {count} stimuli with {difficulty} difficulty")
        
        # Validate response keys in mapping file
        if not validate_response_mapping(stimulus_map):
            logging.warning("Some response mappings may be invalid. Check the stimulus mapping file.")
        
        return stimulus_map
    
    except Exception as e:
        logging.error(f"Failed to load stimulus mapping file: {e}")
        print(f"Error loading stimulus mapping: {e}")
        return None

def load_stimuli(config, stimulus_map, navigation_type, difficulty):
    """Load stimuli for a specific navigation type and difficulty
    
    Args:
        config: The experiment configuration object
        stimulus_map: DataFrame containing stimulus mapping
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        
    Returns:
        tuple: (stimulus_files, stimulus_ids, correct_directions)
    """
    try:
        # Check that stimulus path exists
        stim_path = config.stim_paths.get(difficulty)
        if not stim_path:
            logging.error(f"Stimulus path not defined for difficulty '{difficulty}'")
            return [], [], []
            
        if not os.path.exists(stim_path):
            logging.error(f"Stimulus path does not exist: {stim_path}")
            print(f"ERROR: Stimulus path does not exist: {stim_path}")
            print(f"Creating directory: {stim_path}")
            os.makedirs(stim_path, exist_ok=True)
            return [], [], []
        
        # Map navigation type to correct response column
        if navigation_type == 'egocentric':
            correct_direction_col = 'egocentric_correct_response'
        else:  # allocentric
            correct_direction_col = 'allocentric_correct_response'
        
        # Modified: Filter just by difficulty since navigation_type column doesn't exist
        condition_stimuli = stimulus_map[stimulus_map['difficulty'] == difficulty]
        
        if condition_stimuli.empty:
            logging.error(f"No stimuli found for {navigation_type}/{difficulty} in stimulus mapping")
            print(f"ERROR: No stimuli found for {navigation_type}/{difficulty} in stimulus mapping")
            return [], [], []
        
        # Get stimulus files, ids, and correct responses
        full_stimulus_ids = condition_stimuli['stimulus_id'].tolist()
        correct_directions = condition_stimuli[correct_direction_col].tolist()
        
        # Extract numeric parts from stimulus IDs (e.g., "hard_013" -> "13")
        # This handles both formats: "hard_013" and just "13"
        numeric_ids = []
        for stim_id in full_stimulus_ids:
            # Try to extract numeric part if there's an underscore
            if '_' in str(stim_id):
                parts = str(stim_id).split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    numeric_ids.append(parts[1].lstrip('0'))  # Remove leading zeros
                else:
                    numeric_ids.append(stim_id)  # Keep original if no numeric part found
            else:
                # For IDs that are already numeric
                numeric_ids.append(str(stim_id).lstrip('0'))  # Remove leading zeros if any
        
        # Build full paths to stimulus files
        stimulus_files = []
        missing_files = []
        for i, stim_id in enumerate(numeric_ids):
            file_path = os.path.join(stim_path, f"{stim_id}.jpg")  # Assuming jpg format
            
            # Check multiple common image formats if the jpg doesn't exist
            if not os.path.isfile(file_path):
                for ext in ['.png', '.bmp', '.gif']:
                    alt_path = os.path.join(stim_path, f"{stim_id}{ext}")
                    if os.path.isfile(alt_path):
                        file_path = alt_path
                        break
            
            # If file exists, add it to the list, otherwise note it's missing
            if os.path.isfile(file_path):
                stimulus_files.append(file_path)
            else:
                missing_files.append(full_stimulus_ids[i])
        
        # Log any missing files
        if missing_files:
            logging.warning(f"Missing stimulus files for {navigation_type}/{difficulty}: {missing_files}")
            print(f"WARNING: Missing {len(missing_files)} stimulus files for {navigation_type}/{difficulty}")
            print(f"First few missing IDs: {missing_files[:5]}")
            print(f"Was looking for numeric versions: {[stim_id for stim_id in numeric_ids if os.path.join(stim_path, f'{stim_id}.jpg') not in stimulus_files and os.path.join(stim_path, f'{stim_id}.png') not in stimulus_files][:5]}")
            
            # Create placeholder files for missing stimuli (optional)
            if hasattr(config, 'create_placeholder_stimuli') and config.create_placeholder_stimuli:
                from psychopy import visual
                import numpy as np
                
                print("Creating placeholder stimuli...")
                for stim_id in missing_files:
                    placeholder_path = os.path.join(stim_path, f"{stim_id}.png")
                    # Create a simple placeholder image with the stimulus ID
                    img_array = np.ones((256, 256, 3)) * 128  # Gray background
                    placeholder_img = visual.ImageStim(win=None, image=img_array)
                    placeholder_img.save(placeholder_path)
                    stimulus_files.append(placeholder_path)
        
        # Check if we have any valid stimulus files
        if not stimulus_files:
            logging.error(f"No valid stimulus files found for {navigation_type}/{difficulty}")
            print(f"ERROR: No valid stimulus files found for {navigation_type}/{difficulty}")
            print(f"Please place stimulus images in {stim_path}")
            print(f"Expected formats: JPG, PNG, BMP, GIF")
            return [], [], []
        
        return stimulus_files, full_stimulus_ids, correct_directions
        
    except Exception as e:
        logging.error(f"Error loading stimuli for {navigation_type}/{difficulty}: {e}")
        print(f"ERROR loading stimuli: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []


def prepare_block_stimuli(config, stimulus_files, stimulus_ids, correct_directions, participant_id=None):
    """Prepare a subset of stimuli for a block with enhanced randomization
    
    Args:
        config: The experiment configuration object
        stimulus_files: List of stimulus file paths
        stimulus_ids: List of stimulus IDs
        correct_directions: List of correct responses
        participant_id: Optional participant ID for consistent randomization
        
    Returns:
        tuple: (selected_files, selected_ids, selected_directions)
    """
    # If we don't have enough stimuli, return what we have
    if len(stimulus_files) <= config.trials_per_block:
        return stimulus_files, stimulus_ids, correct_directions
    
    # Create a participant-specific seed for randomization if provided
    if participant_id:
        # Convert participant_id to an integer seed
        try:
            seed = int(participant_id)
        except ValueError:
            # If participant_id isn't a number, create a seed from its characters
            seed = sum(ord(c) for c in str(participant_id))
        
        # Set the random seed for reproducibility while ensuring participant-specific randomization
        import random as rand
        rand_state = rand.getstate()  # Store current state
        rand.seed(seed)  # Set participant-specific seed
    
    # Create a randomized copy of all stimuli indices
    all_indices = list(range(len(stimulus_files)))
    random.shuffle(all_indices)
    
    # Take the first n indices for this block
    selected_indices = all_indices[:config.trials_per_block]
    
    # Select the corresponding stimuli
    selected_files = [stimulus_files[i] for i in selected_indices]
    selected_ids = [stimulus_ids[i] for i in selected_indices]
    selected_directions = [correct_directions[i] for i in selected_indices]
    
    # Further randomize the order of selected stimuli
    combined = list(zip(selected_files, selected_ids, selected_directions))
    
    # Multiple shuffle passes to maximize randomization
    for _ in range(3):
        random.shuffle(combined)
    
    selected_files, selected_ids, selected_directions = zip(*combined)
    
    # Restore random state if we changed it
    if participant_id:
        random.setstate(rand_state)
    
    logging.info(f"Selected {len(selected_files)} stimuli with randomized order for participant {participant_id}")
    return list(selected_files), list(selected_ids), list(selected_directions)


def ensure_enough_stimuli(config, stimulus_files, stimulus_ids, correct_directions):
    """Ensure we have enough stimuli for a block, repeating if necessary
    
    Args:
        config: The experiment configuration object
        stimulus_files: List of stimulus file paths
        stimulus_ids: List of stimulus IDs
        correct_directions: List of correct responses
        
    Returns:
        tuple: (extended_files, extended_ids, extended_directions)
    """
    # If we already have enough, return the lists as is
    if len(stimulus_files) >= config.trials_per_block:
        return stimulus_files, stimulus_ids, correct_directions
    
    # If the list is empty, we can't do anything
    if len(stimulus_files) == 0:
        logging.error("Cannot ensure enough stimuli: stimulus list is empty")
        return [], [], []
    
    # Randomize before repeating to ensure variability
    combined = list(zip(stimulus_files, stimulus_ids, correct_directions))
    random.shuffle(combined)
    stimulus_files, stimulus_ids, correct_directions = zip(*combined)
    stimulus_files, stimulus_ids, correct_directions = list(stimulus_files), list(stimulus_ids), list(correct_directions)
    
    # Calculate how many times we need to repeat
    repetitions_needed = (config.trials_per_block // len(stimulus_files)) + 1
    
    # Create extended lists
    extended_files = stimulus_files * repetitions_needed
    extended_ids = stimulus_ids * repetitions_needed
    extended_directions = correct_directions * repetitions_needed
    
    # Trim to the required number
    extended_files = extended_files[:config.trials_per_block]
    extended_ids = extended_ids[:config.trials_per_block]
    extended_directions = extended_directions[:config.trials_per_block]
    
    logging.info(f"Extended stimulus set from {len(stimulus_files)} to {len(extended_files)} by repetition")
    
    return extended_files, extended_ids, extended_directions