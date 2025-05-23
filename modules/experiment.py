#!C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe
# -*- coding: utf-8 -*-

"""
Experiment Module for Spatial Navigation EEG Experiment
======================================================

This module contains the core experiment functionality with enhanced
EEG logging for detailed mu rhythm and frontal coherence analysis.
"""

import logging
import random
import os
import time
from psychopy import core, event
import numpy as np

# Import all required modules at the top level
from modules import config
from modules.instructions import show_instructions
from modules.ui import show_navigation_transition, show_text_screen, show_welcome_screen, show_block_end_screen, show_completion_screen
from modules.ui import display_stimulus, display_fixation, display_feedback
from modules.data_handler import save_trial_data, save_block_summary
from modules.stimulus import load_stimuli, prepare_block_stimuli, ensure_enough_stimuli

# These need to be imported carefully to avoid circular imports
# We'll import them inside functions where needed


def initialize_eeg_system(config, participant_info):
    """Initialize advanced logging and EEG system
    
    Args:
        config: The experiment configuration object
        participant_info: Dictionary containing participant information
        
    Returns:
        EEGMarkerSystem: The initialized EEG marker system
    """
    # Import here to avoid circular imports
    from modules.advanced_logging import setup_logging
    from modules.eeg import setup_eeg
    
    # First set up the logging system
    loggers = setup_logging(config, participant_info['participant_id'])
    
    # Then set up the EEG system with the loggers
    eeg = setup_eeg(config, loggers)
    
    # Log initialization
    if loggers and 'system' in loggers:
        loggers['system'].info(f"EEG system initialized for participant {participant_info['participant_id']}")
    
    return eeg


def run_trial(window, stimulus_file, stimulus_id, correct_direction, navigation_type, difficulty,
             trial_clock, eeg, config, block_num, trial_num, absolute_trial_num):
    """Run a single trial with enhanced EEG logging
    
    Args:
        window: PsychoPy window object
        stimulus_file: Path to the stimulus image file
        stimulus_id: ID of the stimulus
        correct_direction: The correct response direction
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        trial_clock: PsychoPy clock for timing
        eeg: EEG marker system
        config: The experiment configuration object
        block_num: Current block number
        trial_num: Current trial number within block
        absolute_trial_num: Trial number across all blocks
        
    Returns:
        dict: Trial data containing response, rt, and accuracy
    """
    # Import the consolidated input handler
    from modules.utils import get_input
    
    # Create trigger metadata for this trial
    trigger_metadata = config.create_trigger_metadata(
        navigation_type, difficulty, block_num, trial_num, 
        stimulus_id, correct_direction
    )
    
    # Send EEG trigger for trial start
    eeg.send_trigger(
        config.eeg_trigger_codes['trial_start'],
        f"Trial start: {navigation_type}/{difficulty}",
        trigger_metadata
    )
    
    # Present fixation cross for baseline
    fixation = display_fixation(window, config)
    
    # Send EEG trigger for fixation onset
    eeg.send_trigger(
        config.eeg_trigger_codes['fixation_onset'],
        "Fixation onset",
        trigger_metadata
    )
    
    # Wait for randomized baseline duration
    baseline_duration = random.uniform(*config.intertrial_interval)
    core.wait(baseline_duration)
    
    # Present stimulus
    stimulus = display_stimulus(window, stimulus_file)
    
    # CRITICAL: First draw stimulus but don't flip yet
    stimulus.draw()
    
    # Prepare the stimulus onset trigger before visual presentation
    trigger_code = config.eeg_trigger_codes['stimulus_onset']
    stimulus_metadata = {**trigger_metadata, 'onset_type': 'visual_stimulus'}
    
    # SYNCHRONIZATION POINT: Flip window AND send trigger as close together as possible
    window.callOnFlip(eeg.send_trigger, trigger_code, "Stimulus onset", stimulus_metadata)
    
    # Record exact stimulus presentation time and flip window to show stimulus
    window.flip()
    stimulus_onset_time = time.time()
    
    # Send combined condition code (navigation + difficulty) right after stimulus onset
    combined_condition_code = config.get_combined_condition_code(navigation_type, difficulty)
    if combined_condition_code:
        eeg.send_trigger(
            combined_condition_code,
            f"Combined condition: {navigation_type}/{difficulty}",
            trigger_metadata
        )
    
    # Reset trial clock
    trial_clock.reset()
    
    # ===== Use the consolidated input handler with RT =====
    # Allowed response keys
    allowed_keys = ['up', 'down', 'left', 'right', 'escape']
    
    # Wait for response with timeout and get reaction time
    response, rt = get_input(config, max_wait=config.max_response_time, allowed_keys=allowed_keys, return_rt=True)
    
    # Record response time for EEG analysis
    response_time = time.time()
 
    # Process response
    if response is None:  # No response
        response = 'none'
        rt = None
        accuracy = 0
        
        # Send EEG trigger for no response
        eeg.send_trigger(
            config.eeg_trigger_codes['no_response'],
            "No response",
            {**trigger_metadata, 'rt': config.max_response_time}
        )
        
        logging.info(f"No response within time limit for stimulus {stimulus_id}")
    
    else:
        if response == 'escape':
            logging.info("User pressed escape during trial, quitting")
            
            # Send EEG trigger for experiment terminated
            eeg.send_trigger(
                999,  # Special code for manual termination
                "Experiment manually terminated",
                trigger_metadata
            )
            
            core.quit()
        
        # Calculate and record precise RT (from stimulus onset)
        precise_rt = response_time - stimulus_onset_time
        
        # FIXED: Response mapping issue - now we directly compare to CSV values
        # The CSV stores correct responses as key names ('up', 'down', 'left', 'right')
        # So we need to compare the raw key press to the correct_direction from the CSV
        accuracy = 1 if response == correct_direction else 0
        
        # Map response keys to directions for logging and EEG markers only
        ego_map = {
            'up': 'forward',
            'down': 'backward',
            'left': 'left',
            'right': 'right'
        }
        
        allo_map = {
            'up': 'north',
            'down': 'south',
            'left': 'west',
            'right': 'east'
        }
        
        # Get the direction for logging and EEG triggers, but not for accuracy checking
        if navigation_type == 'egocentric':
            resp_direction = ego_map.get(response, 'unknown')
        else:  # allocentric
            resp_direction = allo_map.get(response, 'unknown')
        
        # Send EEG trigger for combined response type
        combined_response_code = config.get_response_code(navigation_type, response)
        eeg.send_trigger(
            combined_response_code, 
            f"{navigation_type.capitalize()} {response} key response",
            {**trigger_metadata, 'rt': rt, 'precise_rt': precise_rt, 'response': response}
        )
        
        # Send EEG trigger for direction
        direction_code = config.get_direction_code(navigation_type, resp_direction)
        if direction_code:
            eeg.send_trigger(
                direction_code,
                f"{navigation_type.capitalize()} {resp_direction} direction",
                {**trigger_metadata, 'response_direction': resp_direction}
            )
        
        # Send EEG trigger for response outcome (correct/incorrect)
        if accuracy:
            eeg.send_trigger(
                config.eeg_trigger_codes['correct_response'],
                "Correct response",
                {**trigger_metadata, 'response': response, 'rt': rt}
            )
        else:
            eeg.send_trigger(
                config.eeg_trigger_codes['incorrect_response'],
                "Incorrect response",
                {**trigger_metadata, 'response': response, 'rt': rt}
            )
        
        # Send combined outcome code
        outcome_code = config.get_outcome_code(navigation_type, difficulty, accuracy == 1)
        if outcome_code:
            eeg.send_trigger(
                outcome_code,
                f"{navigation_type.capitalize()} {difficulty} {'correct' if accuracy else 'incorrect'}",
                {**trigger_metadata, 'accuracy': accuracy}
            )
        
        # NEW: Send RT binning code
        rt_bin_code = config.get_rt_bin_code(precise_rt, accuracy == 1, navigation_type)
        if rt_bin_code:
            # Get RT bin name for logging
            rt_bin = config.get_rt_bin(precise_rt * 1000)
            eeg.send_trigger(
                rt_bin_code,
                f"{rt_bin.capitalize()} RT {'correct' if accuracy else 'incorrect'} {navigation_type}",
                {**trigger_metadata, 'accuracy': accuracy, 'rt_bin': rt_bin, 'precise_rt': precise_rt}
            )
        
        # NEW: Send combined performance + condition code
        perf_code = config.get_combined_performance_code(precise_rt, accuracy == 1, navigation_type, difficulty)
        if perf_code:
            # Get RT bin name for logging
            rt_bin = config.get_rt_bin(precise_rt * 1000)
            eeg.send_trigger(
                perf_code,
                f"{rt_bin.capitalize()} RT {'correct' if accuracy else 'incorrect'} {navigation_type} {difficulty}",
                {**trigger_metadata, 'accuracy': accuracy, 'rt_bin': rt_bin, 'precise_rt': precise_rt, 
                 'difficulty': difficulty, 'navigation': navigation_type}
            )
        
        logging.info(f"Response: {response}, Direction: {resp_direction}, Correct: {correct_direction}, Accuracy: {accuracy}")
    
    # Present feedback
    if block_num == 0:
        # Detailed feedback for practice trials
        display_feedback(window, accuracy)
    else:
        # No detailed feedback provided during main experiment; proceed without it
        pass
    
    # Send EEG trigger for feedback onset
    eeg.send_trigger(
        config.eeg_trigger_codes['feedback_onset'],
        "Feedback onset",
        {**trigger_metadata, 'accuracy': accuracy}
    )
    
    # Wait for feedback duration
    core.wait(config.feedback_duration)
    
    # Send EEG trigger for feedback offset
    eeg.send_trigger(
        config.eeg_trigger_codes['feedback_offset'],
        "Feedback offset",
        {**trigger_metadata, 'accuracy': accuracy}
    )
    
    # Send EEG trigger for trial end
    eeg.send_trigger(
        config.eeg_trigger_codes['trial_end'],
        "Trial end",
        {**trigger_metadata, 'accuracy': accuracy, 'response': response, 'rt': rt}
    )
    
    # Return trial data with additional behavioral logging data
    trial_data = {
        'response': response,
        'rt': rt,
        'accuracy': accuracy,
        'stimulus_onset_time': stimulus_onset_time,
        'response_time': response_time if response != 'none' else None,
        'precise_rt': response_time - stimulus_onset_time if response != 'none' else None,
        'baseline_duration': baseline_duration
    }
    
    # Add behavioral logging if loggers are provided through config
    if hasattr(config, 'loggers') and config.loggers:
        try:
            # Map response keys to directions for behavioral logging
            ego_map = {'up': 'forward', 'down': 'backward', 'left': 'left', 'right': 'right'}
            allo_map = {'up': 'north', 'down': 'south', 'left': 'west', 'right': 'east'}
            
            # Determine response direction based on navigation type
            if response in ['up', 'down', 'left', 'right']:
                if navigation_type == 'egocentric':
                    resp_direction = ego_map.get(response, 'unknown')
                else:  # allocentric
                    resp_direction = allo_map.get(response, 'unknown')
            else:
                resp_direction = 'none' if response == 'none' else 'unknown'
            
            # Log the trial to behavioral log
            from modules.advanced_logging import log_trial
            log_trial(
                config.loggers,
                block_num,
                trial_num,
                navigation_type,
                difficulty, 
                stimulus_id,
                response,
                resp_direction,
                correct_direction,
                accuracy,
                rt if rt is not None else 0.0
            )
            logging.info(f"Logged behavioral data for trial {trial_num} in block {block_num}")
        except Exception as e:
            logging.error(f"Failed to log behavioral data: {e}")
    
    return trial_data


def run_block(window, config, stimulus_files, stimulus_ids, correct_directions, 
              navigation_type, difficulty, block_num, participant_info, datafile, 
              eeg, absolute_trial_counter):
    """Run a block of trials for a specific navigation type and difficulty
    
    Args:
        window: PsychoPy window object
        config: The experiment configuration object
        stimulus_files: List of stimulus file paths
        stimulus_ids: List of stimulus IDs
        correct_directions: List of correct responses
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        block_num: Current block number
        participant_info: Dictionary containing participant information
        datafile: Path to the data file
        eeg: EEG marker system
        absolute_trial_counter: Counter for trials across all blocks
        
    Returns:
        int: Updated absolute trial counter
    """
    logging.info(f"Starting block {block_num}: {navigation_type}/{difficulty} with {len(stimulus_files)} stimuli")
    
    # Check if we have stimuli before proceeding
    if len(stimulus_files) == 0:
        logging.error(f"No stimuli found for {navigation_type}/{difficulty} - skipping block")
        return absolute_trial_counter
    
    # Create block metadata for EEG markers with enhanced condition tracking
    block_metadata = {
        'block_num': block_num,
        'navigation_type': navigation_type,
        'difficulty': difficulty,
        'condition': f"{navigation_type}_{difficulty}",
        'participant_id': participant_info['participant_id'],
        'block_time': time.time() - config.experiment_start_time,
        'total_blocks': sum(config.condition_blocks.values()),
        'condition_blocks': config.condition_blocks.get(f"{navigation_type}_{difficulty}", 0),
        'trial_count': len(stimulus_files)
    }
    
    # Show instructions for this navigation type and difficulty
    # Pass block_num to show brief instructions during main experiment
    show_instructions(window, navigation_type, difficulty, config, block_num)
    
    # Send EEG trigger for block start and condition with enhanced metadata
    eeg.send_trigger(
        config.eeg_trigger_codes['block_start'],
        f"Block start: {navigation_type}/{difficulty}",
        block_metadata
    )
    
    # Send navigation type trigger
    nav_code = config.eeg_trigger_codes[f'{navigation_type}_condition']
    eeg.send_trigger(
        nav_code,
        f"Navigation type: {navigation_type}",
        block_metadata
    )
    
    # Send difficulty trigger
    diff_code = config.eeg_trigger_codes[f'{difficulty}_difficulty']
    eeg.send_trigger(
        diff_code,
        f"Difficulty: {difficulty}",
        block_metadata
    )
    
    # Send combined condition code
    combined_code = config.get_combined_condition_code(navigation_type, difficulty)
    if combined_code:
        eeg.send_trigger(
            combined_code,
            f"Combined condition: {navigation_type}/{difficulty}",
            block_metadata
        )
    
    # Create trial clock
    trial_clock = core.Clock()
    
    # Ensure we have enough stimuli, either by selecting a subset or by repetition
    if len(stimulus_files) > config.trials_per_block:
        # Select random subset
        selected_files, selected_ids, selected_directions = prepare_block_stimuli(
            config, stimulus_files, stimulus_ids, correct_directions
        )
    else:
        # Repeat stimuli if needed
        selected_files, selected_ids, selected_directions = ensure_enough_stimuli(
            config, stimulus_files, stimulus_ids, correct_directions
        )
    
    # Create randomized trial sequence with enhanced randomization
    trials = list(zip(selected_files, selected_ids, selected_directions))
    
    # Add extra shuffling to maximize randomization
    for _ in range(3):  # Multiple shuffle passes
        random.shuffle(trials)
    
    # Log trial sequence for verification
    trial_sequence_summary = ", ".join([f"{id}" for _, id, _ in trials[:5]]) + "..."
    logging.info(f"Randomized trial sequence (first 5): {trial_sequence_summary}")
    
    # Prepare to collect block statistics
    block_responses = []
    block_rts = []
    block_accuracies = []
    block_precise_rts = []  # For EEG analysis
    
    # Run each trial
    for trial_num, (stimulus_file, stimulus_id, correct_direction) in enumerate(trials, 1):
        logging.info(f"Running trial {trial_num}/{len(trials)} with stimulus {stimulus_id}")
        
        # Increment absolute trial counter
        absolute_trial_counter += 1
        
        # Run the trial with enhanced EEG logging
        trial_data = run_trial(
            window, stimulus_file, stimulus_id, correct_direction, 
            navigation_type, difficulty, trial_clock, eeg, config,
            block_num, trial_num, absolute_trial_counter
        )
        
        # Collect data for block summary
        if trial_data['response'] != 'none':
            block_responses.append(trial_data['response'])
            block_rts.append(trial_data['rt'])
            block_precise_rts.append(trial_data['precise_rt'])
            block_accuracies.append(trial_data['accuracy'])
        
        # Save trial data with enhanced EEG info
        save_trial_data(
            datafile, participant_info, block_num, trial_num, navigation_type, difficulty,
            stimulus_id, stimulus_file, correct_direction, trial_data['response'], 
            trial_data['accuracy'], trial_data['rt'], absolute_trial_counter,
            trial_data.get('precise_rt'), trial_data.get('baseline_duration'),
            trial_data.get('stimulus_onset_time')
        )
        
        # Check for quit
        if event.getKeys(keyList=['escape']):
            logging.info("User pressed escape between trials, quitting")
            eeg.send_trigger(
                999,  # Special code for manual termination
                "Experiment manually terminated between trials",
                block_metadata
            )
            core.quit()
    
    # Calculate block statistics
    num_trials = len(block_responses)
    correct_trials = sum(block_accuracies)
    mean_rt = np.mean(block_rts) if block_rts else 0
    mean_precise_rt = np.mean(block_precise_rts) if block_precise_rts else 0
    
    # Add detailed metrics for EEG analysis
    block_eeg_metrics = {
        'mean_precise_rt': mean_precise_rt,
        'sd_precise_rt': np.std(block_precise_rts) if block_precise_rts else 0,
        'min_rt': np.min(block_precise_rts) if block_precise_rts else 0,
        'max_rt': np.max(block_precise_rts) if block_precise_rts else 0,
        'correct_percentage': (correct_trials / num_trials * 100) if num_trials > 0 else 0,
        'condition_count': 1, # Default to 1 if not calculated elsewhere
        'total_condition_blocks': config.repetitions
    }
    
    # Calculate condition-specific block count if possible
    condition_key = f"{navigation_type}_{difficulty}"
    if hasattr(config, 'condition_counts') and condition_key in config.condition_counts:
        block_eeg_metrics['condition_count'] = config.condition_counts[condition_key]
    
    # Save block summary with enhanced metrics
    save_block_summary(
        datafile, block_num, navigation_type, difficulty,
        num_trials, correct_trials, mean_rt, block_eeg_metrics
    )
    
    # Log block summary to advanced logging system if available
    if hasattr(config, 'loggers') and config.loggers:
        try:
            from modules.advanced_logging import log_block_summary
            log_block_summary(
                config.loggers, block_num, navigation_type, difficulty,
                num_trials, correct_trials, mean_rt, block_eeg_metrics
            )
        except Exception as e:
            logging.error(f"Failed to log block summary: {e}")
    
    # Send EEG trigger for block end with block statistics
    end_metadata = {**block_metadata, **block_eeg_metrics}
    eeg.send_trigger(
        config.eeg_trigger_codes['block_end'],
        "Block end",
        end_metadata
    )
    
    # Calculate total blocks for progress display
    total_blocks = sum(config.condition_blocks.values())
    
    # Show block completion message with progress information
    show_block_end_screen(
        window, config, block_num, total_blocks, 
        absolute_trial_counter, config.total_trials
    )
    
    return absolute_trial_counter


def run_practice_trials(window, config, stimulus_map, participant_info, eeg):
    """Run a set of practice trials to familiarize the participant.
    These trials are not saved to the main data file.
    """
    logging.info("Starting practice trials")
    from modules.ui import show_practice_instructions, show_practice_condition_screen
    
    # First show general practice instructions
    show_practice_instructions(window, config)
    
    num_practice = config.num_practice_trials
    # Fixed order: PLAYER VIEW (egocentric) then MAP VIEW (allocentric)
    practice_order = ['egocentric', 'allocentric']
    practice_difficulty = 'easy'
    total_practice_trials = 0
    
    for nav_type in practice_order:
        # ADDED: Show a clear screen announcing which practice condition is coming up
        show_practice_condition_screen(window, config, nav_type)
        
        # Load stimuli for this practice condition
        stimulus_files, stimulus_ids, correct_directions = load_stimuli(config, stimulus_map, nav_type, practice_difficulty)
        if not stimulus_files:
            logging.error(f"No practice stimuli for {nav_type}/{practice_difficulty}")
            continue
            
        n_trials = num_practice // len(practice_order)
        selected_files = stimulus_files[:n_trials]
        selected_ids = stimulus_ids[:n_trials]
        selected_directions = correct_directions[:n_trials]
        
        trial_clock = core.Clock()
        for i in range(n_trials):
            total_practice_trials += 1
            logging.info(f"Practice trial {total_practice_trials}: {nav_type} / {practice_difficulty}")
            run_trial(
                window, selected_files[i], selected_ids[i], selected_directions[i],
                nav_type, practice_difficulty, trial_clock, eeg, config,
                block_num=0, trial_num=total_practice_trials, absolute_trial_num=total_practice_trials
            )
        
        # After completing a practice block, show appropriate message
        if nav_type == 'egocentric':
            # MODIFIED: Removed redundant condition name from the message since we're now showing the condition screen
            show_text_screen(window,
                "Practice complete.\n\nPress SPACE to start the next practice type.",
                text_color=config.text_color
            )
    
    show_text_screen(window,
        "All practice trials complete.\n\n"
        "The main experiment consists of\n"
        "9 sections.\n"
        "Press SPACE to continue with the main experiment.",
        text_color=config.text_color
    )
    
    logging.info("Practice trials completed")
    return


def run_experiment_blocks(window, config, stimulus_map, participant_info, datafile, eeg):
    """Run all blocks of the experiment with comprehensive EEG logging
    
    Args:
        window: PsychoPy window object
        config: The experiment configuration object
        stimulus_map: DataFrame containing stimulus mapping
        participant_info: Dictionary containing participant information
        datafile: Path to the data file
        eeg: EEG marker system
    """
    from modules.utils import setup_escape_handler
    
    # FIXED: Don't set up escape handler if universal escape is already set up
    # This check assumes universal escape has already been set up in main.py
    from psychopy import event
    if 'escape' not in event.globalKeys:
        # Set up global escape handler only if not already set up
        setup_escape_handler(window, eeg, loggers=None)
        logging.info("Escape handler registered in experiment blocks")
    else:
        logging.info("Using existing escape handler for experiment blocks")
    
    # Create experiment metadata for EEG logging
    experiment_metadata = {
        'experiment_name': 'Spatial Navigation EEG',
        'participant_id': participant_info['participant_id'],
        'session': participant_info['session'],
        'age': participant_info['age'],
        'gender': participant_info['gender'],
        'handedness': participant_info['handedness'],
        'counterbalance': participant_info['counterbalance'],
        'total_blocks': len(config.navigation_types) * len(config.difficulty_levels) * config.repetitions,
        'total_trials': config.total_trials,
        'navigation_types': config.navigation_types,
        'difficulty_levels': config.difficulty_levels,
        'trials_per_block': config.trials_per_block,
        'max_response_time': config.max_response_time,
        'datetime': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add experiment metadata to EEG log
    if eeg:
        eeg.add_metadata_to_log(experiment_metadata)
    
    # Send EEG trigger for experiment start with metadata
    if eeg:
        eeg.send_trigger(
            config.eeg_trigger_codes['experiment_start'],
            "Experiment start",
            experiment_metadata
        )
    
    # Run practice trials before main experiment
    run_practice_trials(window, config, stimulus_map, participant_info, eeg)
    
    # Show welcome screen for main experiment
    total_blocks = sum(config.condition_blocks.values())
    show_welcome_screen(window, config, config.total_trials, total_blocks)
    
    # Get the alternating block sequence with enhanced randomization
    # Pass participant_id to ensure consistent randomization for this participant
    block_sequence = config.create_block_sequence(
        participant_info['counterbalance'],
        participant_id=participant_info['participant_id']
    )
    
    # Log the condition distribution for verification
    condition_counts = {}
    for nav_type, diff, _ in block_sequence:
        key = f"{nav_type}/{diff}"
        if key not in condition_counts:
            condition_counts[key] = 0
        condition_counts[key] += 1
    
    logging.info(f"Block condition distribution: {condition_counts}")
    
    logging.info(f"Created alternating block sequence with {len(block_sequence)} blocks")
    
    # Initialize absolute trial counter
    absolute_trial_counter = 0
    
    # Track the current navigation type to show transitions
    current_nav_type = None
    
    # Run each block
    for navigation_type, difficulty, block_num in block_sequence:
        # Log block start
        logging.info(f"Preparing block {block_num}/{total_blocks}: {navigation_type}/{difficulty}")
        
        # Show navigation transition screen if navigation type is changing
        show_navigation_transition(window, current_nav_type, navigation_type, config)
        
        # Update current navigation type
        current_nav_type = navigation_type
        
        # Load stimuli for this block
        stimulus_files, stimulus_ids, correct_directions = load_stimuli(
            config, stimulus_map, navigation_type, difficulty
        )
        
        # Prepare block stimuli with participant-specific randomization
        stimulus_files, stimulus_ids, correct_directions = prepare_block_stimuli(
            config, stimulus_files, stimulus_ids, correct_directions, 
            participant_id=participant_info['participant_id']
        )
        
        # Run the block with enhanced EEG logging
        absolute_trial_counter = run_block(
            window, config, stimulus_files, stimulus_ids, correct_directions, 
            navigation_type, difficulty, block_num, participant_info, 
            datafile, eeg, absolute_trial_counter
        )
    
    # Create experiment summary for EEG log
    experiment_summary = {
        'completed_trials': absolute_trial_counter,
        'completed_blocks': total_blocks,
        'completion_time': time.time() - config.experiment_start_time,
        'experiment_end_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add experiment summary to EEG log
    if eeg:
        eeg.add_metadata_to_log({**experiment_metadata, **experiment_summary})
    
    # Show completion screen
    show_completion_screen(window, config, absolute_trial_counter)
    
    # Send EEG trigger for experiment end with summary
    if eeg:
        eeg.send_trigger(
            config.eeg_trigger_codes['experiment_end'],
            "Experiment end",
            experiment_summary
        )
        
        # Close EEG connections
        eeg.close()
    
    # Log completion
    logging.info(f"Experiment completed successfully with {absolute_trial_counter} trials")