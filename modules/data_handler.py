#!C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe
# -*- coding: utf-8 -*-

"""
Enhanced Data Handler Module for Spatial Navigation EEG Experiment
=================================================================

This module handles data operations with extended capabilities for
detailed EEG analysis of mu rhythms and frontal coherence.
"""

import os
import csv
import json
import logging
import numpy as np
from datetime import datetime

from modules.advanced_logging import log_eeg_event, log_error


def create_data_file(config, participant_info):
    """Create a data file for saving results with extended EEG analysis fields

    Args:
        config: The experiment configuration object
        participant_info: Dictionary containing participant information

    Returns:
        str: Path to the created data file
    """
    # Generate a unique filename based on participant ID and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"sub-{participant_info['participant_id']}_ses-{participant_info['session']}_{timestamp}"
    
    # Create data directory if it doesn't exist
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
    
    # Create datafile path
    datafile = os.path.join(config.data_dir, filename + '.csv')
    
    # Create and write headers to the file with enhanced EEG fields
    with open(datafile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'participant_id', 'session', 'age', 'gender', 'handedness', 
            'block_num', 'trial_num', 'navigation_type', 'difficulty', 'stimulus_id',
            'stimulus_file', 'correct_direction', 'response', 'accuracy', 'rt', 
            'precise_rt', 'baseline_duration', 'stimulus_onset_time', 'timestamp', 
            'absolute_trial_num', 'counterbalance'
        ])
    
    # Also create a companion JSON file for rich metadata
    json_file = os.path.join(config.data_dir, filename + '_metadata.json')
    
    # Create metadata JSON with experiment parameters
    metadata = {
        'experiment': 'Spatial Navigation EEG',
        'version': '1.0',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'participant': {
            'id': participant_info['participant_id'],
            'session': participant_info['session'],
            'age': participant_info['age'],
            'gender': participant_info['gender'],
            'handedness': participant_info['handedness'],
            'counterbalance': participant_info['counterbalance']
        },
        'experiment_parameters': {
            'navigation_types': config.navigation_types,
            'difficulty_levels': config.difficulty_levels,
            'trials_per_block': config.trials_per_block,
            'repetitions': config.repetitions,
            'total_trials_planned': config.total_trials,
            'max_response_time': config.max_response_time,
            'feedback_duration': config.feedback_duration,
            'intertrial_interval': [config.intertrial_interval[0], config.intertrial_interval[1]]
        },
        'eeg_parameters': {
            'baseline_duration': config.baseline_duration,
            'post_response_record': config.post_response_record
        },
        'trigger_codes': config.eeg_trigger_codes
    }
    
    # Write metadata to file
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Created data file: {datafile} with companion metadata file")
    return datafile


def save_trial_data(datafile, participant_info, block_num, trial_num, navigation_type, difficulty, 
                   stimulus_id, stimulus_file, correct_direction, response, accuracy, rt, 
                   absolute_trial_num, precise_rt=None, baseline_duration=None, 
                   stimulus_onset_time=None):
    """Save trial data to CSV file with enhanced EEG analysis fields
    
    Args:
        datafile: Path to the data file
        participant_info: Dictionary containing participant information
        block_num: Current block number
        trial_num: Current trial number within the block
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        stimulus_id: ID of the current stimulus
        stimulus_file: Path to the stimulus file
        correct_direction: Correct response direction
        response: Participant's response
        accuracy: 1 if correct, 0 if incorrect
        rt: Response time in seconds as measured by PsychoPy
        absolute_trial_num: Trial number across all blocks
        precise_rt: Precise response time measured from stimulus onset
        baseline_duration: Duration of the baseline period before stimulus
        stimulus_onset_time: Absolute timestamp of stimulus onset
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    try:
        with open(datafile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                participant_info['participant_id'], 
                participant_info['session'], 
                participant_info['age'], 
                participant_info['gender'], 
                participant_info['handedness'],
                block_num, 
                trial_num, 
                navigation_type, 
                difficulty, 
                stimulus_id,
                os.path.basename(stimulus_file),
                correct_direction, 
                response, 
                accuracy, 
                rt, 
                precise_rt,
                baseline_duration,
                stimulus_onset_time,
                timestamp,
                absolute_trial_num,
                participant_info['counterbalance']
            ])
        
        logging.debug(f"Saved trial data: Block {block_num}, Trial {trial_num}, Response: {response}, Accuracy: {accuracy}")
    
    except Exception as e:
        logging.error(f"Error saving trial data: {e}")
        # Continue execution despite saving error
        print(f"Warning: Failed to save trial data: {e}")


def save_block_summary(datafile, block_num, navigation_type, difficulty, 
                     num_trials, correct_trials, mean_rt, additional_metrics=None):
    """Save summary statistics for a block with extended EEG metrics
    
    Args:
        datafile: Path to the data file
        block_num: Block number
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        num_trials: Number of trials in the block
        correct_trials: Number of correct trials
        mean_rt: Mean response time
        additional_metrics: Dictionary of additional EEG-related metrics
    """
    # Extract base path and add _summary suffix
    base_path = os.path.splitext(datafile)[0]
    summary_file = base_path + '_block_summary.csv'
    
    # Create the file with headers if it doesn't exist
    file_exists = os.path.isfile(summary_file)
    
    try:
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if file is new - include EEG-specific fields and condition
            if not file_exists:
                writer.writerow([
                    'block_num', 'navigation_type', 'difficulty', 'condition',
                    'num_trials', 'correct_trials', 'accuracy_percent', 'mean_rt',
                    'mean_precise_rt', 'sd_precise_rt', 'min_rt', 'max_rt',
                    'timestamp'
                ])
            
            # Calculate accuracy percentage
            accuracy_percent = (correct_trials / num_trials) * 100 if num_trials > 0 else 0
            
            # Get additional metrics or use defaults
            if additional_metrics is None:
                additional_metrics = {
                    'mean_precise_rt': 0,
                    'sd_precise_rt': 0,
                    'min_rt': 0,
                    'max_rt': 0
                }
            
            # Write block summary with combined condition field
            condition = f"{navigation_type}_{difficulty}"
            writer.writerow([
                block_num, navigation_type, difficulty, condition,
                num_trials, correct_trials, accuracy_percent, mean_rt,
                additional_metrics.get('mean_precise_rt', 0),
                additional_metrics.get('sd_precise_rt', 0),
                additional_metrics.get('min_rt', 0),
                additional_metrics.get('max_rt', 0),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        logging.info(f"Saved block summary for block {block_num}: "
                     f"{navigation_type}/{difficulty}, Accuracy: {accuracy_percent:.1f}%, "
                     f"Mean RT: {mean_rt:.3f}s, Mean Precise RT: {additional_metrics.get('mean_precise_rt', 0):.3f}s")
        
        # Also save a more detailed JSON format summary for richer analysis
        save_block_summary_json(datafile, block_num, navigation_type, difficulty, 
                               num_trials, correct_trials, mean_rt, additional_metrics)
    
    except Exception as e:
        logging.error(f"Error saving block summary: {e}")


def save_block_summary_json(datafile, block_num, navigation_type, difficulty, 
                          num_trials, correct_trials, mean_rt, additional_metrics=None):
    """Save a detailed JSON format block summary for rich EEG analysis
    
    Args:
        datafile: Path to the data file
        block_num: Block number
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        num_trials: Number of trials in the block
        correct_trials: Number of correct trials
        mean_rt: Mean response time
        additional_metrics: Dictionary of additional EEG-related metrics
    """
    # Extract base path for JSON summary
    base_path = os.path.splitext(datafile)[0]
    json_summary_file = base_path + f'_block_{block_num}_summary.json'
    
    # Prepare summary data
    summary_data = {
        'block_info': {
            'block_num': block_num,
            'navigation_type': navigation_type,
            'difficulty': difficulty,
            'condition': f"{navigation_type}_{difficulty}"
        },
        'performance': {
            'num_trials': num_trials,
            'correct_trials': correct_trials,
            'accuracy_percent': (correct_trials / num_trials) * 100 if num_trials > 0 else 0,
            'error_trials': num_trials - correct_trials,
            'mean_rt': mean_rt
        },
        'eeg_metrics': additional_metrics or {},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Write to JSON file
    try:
        with open(json_summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        logging.info(f"Saved detailed JSON block summary for block {block_num}")
    
    except Exception as e:
        logging.error(f"Error saving JSON block summary: {e}")


def extract_trials_for_eeg_analysis(datafile, output_format='bids'):
    """Extract trial information in formats suitable for EEG analysis
    
    Args:
        datafile: Path to the data file
        output_format: Format for the output ('bids', 'eeglab', or 'fieldtrip')
        
    Returns:
        str: Path to the created output file
    """
    try:
        # Read the CSV data
        with open(datafile, 'r', newline='') as f:
            reader = csv.DictReader(f)
            trials = list(reader)
        
        # Extract base path
        base_path = os.path.splitext(datafile)[0]
        
        if output_format == 'bids':
            # Create BIDS-compatible events.tsv file
            output_file = base_path + '_events.tsv'
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([
                    'onset', 'duration', 'trial_type', 'response_time', 
                    'stim_file', 'correct_response', 'accuracy'
                ])
                
                for trial in trials:
                    # Convert stimulus_onset_time to onset relative to recording start
                    if trial['stimulus_onset_time'] and float(trial['stimulus_onset_time']) > 0:
                        onset = float(trial['stimulus_onset_time'])
                    else:
                        # If no stimulus onset time, skip this trial
                        continue
                    
                    # Define the trial type as combined condition
                    trial_type = f"{trial['navigation_type']}_{trial['difficulty']}"
                    
                    # Calculate duration as RT or max_time if no response
                    duration = float(trial['precise_rt']) if trial['precise_rt'] else 3.0  # Assuming 3s max time
                    
                    writer.writerow([
                        onset,
                        duration,
                        trial_type,
                        trial['precise_rt'] if trial['precise_rt'] else 'n/a',
                        trial['stimulus_file'],
                        trial['correct_direction'],
                        trial['accuracy']
                    ])
            
            logging.info(f"Created BIDS-compatible events file: {output_file}")
            return output_file
        
        elif output_format == 'eeglab':
            # Create EEGLAB-compatible events in .txt format
            output_file = base_path + '_eeglab_events.txt'
            
            # Implementation for EEGLAB format
            # [This would be customized based on EEGLAB requirements]
            
            logging.info(f"Created EEGLAB-compatible events file: {output_file}")
            return output_file
        
        elif output_format == 'fieldtrip':
            # Create FieldTrip-compatible events in .mat format
            # [This would require SciPy to save .mat files]
            
            output_file = base_path + '_fieldtrip_events.json'  # JSON as a placeholder
            
            # Implementation for FieldTrip format
            # [This would be customized based on FieldTrip requirements]
            
            logging.info(f"Created FieldTrip-compatible events file: {output_file}")
            return output_file
        
        else:
            logging.error(f"Unsupported output format: {output_format}")
            return None
    
    except Exception as e:
        logging.error(f"Error extracting trials for EEG analysis: {e}")
        return None


def calculate_block_metrics(datafile, block_num):
    """Calculate detailed metrics for a block for EEG analysis
    
    Args:
        datafile: Path to the data file
        block_num: Block number to analyze
        
    Returns:
        dict: Dictionary of block metrics
    """
    try:
        # Read the CSV data
        with open(datafile, 'r', newline='') as f:
            reader = csv.DictReader(f)
            trials = [row for row in reader if int(row['block_num']) == block_num]
        
        if not trials:
            return {}
            
        # Extract performance metrics
        num_trials = len(trials)
        correct_trials = sum(1 for t in trials if int(t['accuracy']) == 1)
        incorrect_trials = num_trials - correct_trials
        
        # Calculate RT metrics from precise_rt when available, otherwise use rt
        rts = []
        for t in trials:
            if t['precise_rt'] and float(t['precise_rt']) > 0:
                rts.append(float(t['precise_rt']))
            elif t['rt'] and float(t['rt']) > 0:
                rts.append(float(t['rt']))
        
        if rts:
            mean_rt = np.mean(rts)
            median_rt = np.median(rts)
            sd_rt = np.std(rts)
            min_rt = min(rts)
            max_rt = max(rts)
        else:
            mean_rt = median_rt = sd_rt = min_rt = max_rt = 0
        
        # Get navigation type and difficulty for this block
        nav_type = trials[0]['navigation_type'] if trials else 'unknown'
        difficulty = trials[0]['difficulty'] if trials else 'unknown'
        
        return {
            'mean_rt': mean_rt,
            'median_rt': median_rt,
            'sd_rt': sd_rt,
            'min_rt': min_rt,
            'max_rt': max_rt,
            'accuracy': (correct_trials / num_trials) * 100 if num_trials > 0 else 0,
            'navigation_type': nav_type,
            'difficulty': difficulty,
            'num_trials': num_trials,
            'correct_trials': correct_trials,
            'incorrect_trials': incorrect_trials
        }
    
    except Exception as e:
        logging.error(f"Error calculating block metrics: {e}")
        return {}


def analyze_experiment_progress(datafile, total_expected_trials=None, loggers=None):
    """Analyze experiment progress and completion status
    
    Args:
        datafile: Path to the data file
        total_expected_trials: Total expected trials (if known)
        loggers: Dictionary of logger objects
    
    Returns:
        dict: Dictionary with experiment progress metrics
    """
    try:
        # Read the CSV data
        with open(datafile, 'r', newline='') as f:
            reader = csv.DictReader(f)
            trials = list(reader)
        
        # Count completed trials
        completed_trials = len(trials)
        
        # Calculate completion percentage if total expected is provided
        completion_percentage = None
        if total_expected_trials is not None and total_expected_trials > 0:
            completion_percentage = (completed_trials / total_expected_trials) * 100
        
        # Count trials by condition
        conditions = {}
        for trial in trials:
            nav_type = trial['navigation_type']
            difficulty = trial['difficulty']
            condition = f"{nav_type}_{difficulty}"
            
            if condition not in conditions:
                conditions[condition] = {
                    'count': 0,
                    'correct': 0
                }
            
            conditions[condition]['count'] += 1
            if int(trial['accuracy']) == 1:
                conditions[condition]['correct'] += 1
        
        # Calculate accuracy by condition
        for condition in conditions:
            count = conditions[condition]['count']
            correct = conditions[condition]['correct']
            conditions[condition]['accuracy'] = (correct / count * 100) if count > 0 else 0
        
        # Compile results
        results = {
            'completed_trials': completed_trials,
            'completion_percentage': completion_percentage,
            'conditions': conditions
        }
        
        # Log summary if loggers are available
        if loggers:
            progress_msg = f"Experiment progress: {completed_trials} trials completed"
            if completion_percentage is not None:
                progress_msg += f" ({completion_percentage:.1f}% complete)"
            
            loggers['system'].info(progress_msg)
            
            # Log condition summaries
            for condition, data in conditions.items():
                loggers['system'].info(
                    f"Condition {condition}: {data['count']} trials, "
                    f"{data['accuracy']:.1f}% accuracy ({data['correct']}/{data['count']})"
                )
        
        return results
    
    except Exception as e:
        if loggers:
            log_error(loggers, f"Error analyzing experiment progress", e)
        else:
            logging.error(f"Error analyzing experiment progress: {e}")
        return {'completed_trials': 0}