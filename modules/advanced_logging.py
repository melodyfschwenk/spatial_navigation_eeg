#!C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe
# -*- coding: utf-8 -*-

"""
Advanced Logging Configuration for Spatial Navigation EEG Experiment
===================================================================

This module configures an advanced multi-logger system with separate streams for:
1. System logs - Technical details about the experiment execution
2. Behavioral logs - Participant performance data
3. EEG event logs - Detailed event markers for EEG analysis
4. Error logs - Warnings and errors

Each log type is directed to its own file with appropriate formatting.
"""

import os
import logging
import logging.handlers
from datetime import datetime


class LoggerNames:
    """Constants for logger names to ensure consistency"""
    SYSTEM = "system"
    BEHAVIOR = "behavior"
    EEG = "eeg"
    ERROR = "error"
    

def setup_logging(config, participant_id):
    """Set up advanced logging system with multiple specialized loggers
    
    Args:
        config: Experiment configuration object
        participant_id: Participant identifier for filename
        
    Returns:
        dict: Dictionary of logger objects
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log directory if it doesn't exist
    log_dir = config.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create behavior data directory if it doesn't exist
    behavior_dir = os.path.join(config.data_dir, 'behavior')
    if not os.path.exists(behavior_dir):
        os.makedirs(behavior_dir)
        
    # Create EEG event directory if it doesn't exist
    eeg_dir = os.path.join(config.data_dir, 'eeg_events')
    if not os.path.exists(eeg_dir):
        os.makedirs(eeg_dir)
    
    # Base filename with participant ID and timestamp
    base_filename = f"sub-{participant_id}_{timestamp}"
    
    # File paths for each log type
    system_log_path = os.path.join(log_dir, f"{base_filename}_system.log")
    behavior_log_path = os.path.join(behavior_dir, f"{base_filename}_behavior.log")
    eeg_log_path = os.path.join(eeg_dir, f"{base_filename}_eeg_events.log")
    error_log_path = os.path.join(log_dir, f"{base_filename}_error.log")
    
    # Create and configure loggers
    loggers = {
        LoggerNames.SYSTEM: _create_system_logger(system_log_path),
        LoggerNames.BEHAVIOR: _create_behavior_logger(behavior_log_path),
        LoggerNames.EEG: _create_eeg_logger(eeg_log_path),
        LoggerNames.ERROR: _create_error_logger(error_log_path)
    }
    
    # Log paths for reference
    loggers[LoggerNames.SYSTEM].info(f"System log path: {system_log_path}")
    loggers[LoggerNames.SYSTEM].info(f"Behavior log path: {behavior_log_path}")
    loggers[LoggerNames.SYSTEM].info(f"EEG events log path: {eeg_log_path}")
    loggers[LoggerNames.SYSTEM].info(f"Error log path: {error_log_path}")
    
    # Set up console output for warnings and errors
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console.setFormatter(console_formatter)
    
    # Add console handler to error logger
    loggers[LoggerNames.ERROR].addHandler(console)
    
    return loggers


def _create_system_logger(log_path):
    """Create logger for system events
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Logger: Configured logger object
    """
    logger = logging.getLogger(LoggerNames.SYSTEM)
    logger.setLevel(logging.INFO)
    
    # Reset handlers if already configured
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


def _create_behavior_logger(log_path):
    """Create a logger for behavioral data"""
    logger = logging.getLogger('behavior')
    logger.setLevel(logging.INFO)
    
    # Reset handlers if already configured
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # Create a formatter that includes timestamp
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
    fh.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    
    # Write header row
    logger.info("block\ttrial\tnav_type\tdifficulty\tstimulus_id\tresponse\tdirection\tcorrect\taccuracy\trt")
    
    return logger


def _create_eeg_logger(log_path):
    """Create logger for EEG event markers
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Logger: Configured logger object
    """
    logger = logging.getLogger(LoggerNames.EEG)
    logger.setLevel(logging.INFO)
    
    # Reset handlers if already configured
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create precise timestamp formatter for EEG analysis
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Add header to EEG log
    logger.info("timestamp\ttrigger_code\tevent_type\tblock\ttrial\tnav_type\tdifficulty\tdetails")
    
    return logger


def _create_error_logger(log_path):
    """Create logger for errors and warnings
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Logger: Configured logger object
    """
    logger = logging.getLogger(LoggerNames.ERROR)
    logger.setLevel(logging.WARNING)
    
    # Reset handlers if already configured
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.WARNING)
    
    # Create detailed formatter for debugging
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


def log_trial(loggers, block_num, trial_num, navigation_type, difficulty, 
             stimulus_id, response, direction, correct_direction, accuracy, rt):
    """Log trial data to both behavioral and system logs
    
    Args:
        loggers: Dictionary of logger objects
        block_num: Current block number
        trial_num: Current trial number
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        stimulus_id: ID of the stimulus
        response: Participant's response key
        direction: Direction of movement
        correct_direction: Correct direction
        accuracy: 1 if correct, 0 if incorrect
        rt: Response time
    """
    try:
        # Log to behavioral data file (tab-separated for easy analysis)
        behavior_msg = f"{block_num}\t{trial_num}\t{navigation_type}\t{difficulty}\t" \
                      f"{stimulus_id}\t{response}\t{direction}\t{correct_direction}\t{accuracy}\t{rt:.4f}"
        
        # Check if behavior logger exists
        if LoggerNames.BEHAVIOR in loggers:
            loggers[LoggerNames.BEHAVIOR].info(behavior_msg)
        else:
            logging.warning(f"Behavior logger not found in loggers dictionary. Available loggers: {list(loggers.keys())}")
        
        # Log to system log (more human-readable)
        system_msg = f"Trial {trial_num} (Block {block_num}): {navigation_type}/{difficulty}, " \
                    f"Response: {response} ({direction}), Correct: {correct_direction}, Accuracy: {accuracy}, RT: {rt:.3f}s"
        
        if LoggerNames.SYSTEM in loggers:
            loggers[LoggerNames.SYSTEM].info(system_msg)
        else:
            # Fallback to standard logging if system logger not found
            logging.info(system_msg)
            
    except Exception as e:
        logging.error(f"Error in log_trial function: {e}")
        # Fallback to standard logging
        logging.info(f"BEHAVIOR: Block {block_num}, Trial {trial_num}, {navigation_type}/{difficulty}, " 
                    f"Response: {response}, Accuracy: {accuracy}, RT: {rt:.3f}s")


def log_eeg_event(loggers, trigger_code, event_type, block_num, trial_num, 
                navigation_type, difficulty, additional_info=None):
    """Log EEG event markers
    
    Args:
        loggers: Dictionary of logger objects
        trigger_code: Numeric trigger code sent to EEG system
        event_type: Description of the event (e.g., 'stimulus_onset')
        block_num: Current block number
        trial_num: Current trial number
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        additional_info: Optional additional information (string)
    """
    details = additional_info or ""
    
    # Log to EEG events file (tab-separated for easy analysis)
    eeg_msg = f"{trigger_code}\t{event_type}\t{block_num}\t{trial_num}\t" \
             f"{navigation_type}\t{difficulty}\t{details}"
    loggers[LoggerNames.EEG].info(eeg_msg)
    
    # Also log to system log at debug level
    system_msg = f"EEG Trigger {trigger_code}: {event_type} (Block {block_num}, Trial {trial_num})"
    loggers[LoggerNames.SYSTEM].debug(system_msg)


def log_block_summary(loggers, block_num, navigation_type, difficulty, 
                    num_trials, correct_trials, mean_rt, additional_metrics=None):
    """Log block summary to system log
    
    Args:
        loggers: Dictionary of logger objects
        block_num: Block number
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        num_trials: Number of trials in the block
        correct_trials: Number of correct trials
        mean_rt: Mean response time
        additional_metrics: Optional dictionary of additional metrics
    """
    accuracy = (correct_trials / num_trials * 100) if num_trials > 0 else 0
    
    # Create combined condition identifier
    condition = f"{navigation_type}_{difficulty}"
    
    system_msg = f"Block {block_num} summary: {condition}, " \
                f"Accuracy: {accuracy:.1f}% ({correct_trials}/{num_trials}), Mean RT: {mean_rt:.3f}s"
    
    # Add additional metrics if provided
    if additional_metrics:
        for metric, value in additional_metrics.items():
            if isinstance(value, float):
                system_msg += f", {metric}: {value:.3f}"
            else:
                system_msg += f", {metric}: {value}"
    
    loggers[LoggerNames.SYSTEM].info(system_msg)
    
    # Also log specific condition count information if available
    if 'condition_count' in additional_metrics:
        condition_msg = f"Condition {condition} block #{additional_metrics['condition_count']} of {additional_metrics.get('total_condition_blocks', '?')}"
        loggers[LoggerNames.SYSTEM].info(condition_msg)


def log_error(loggers, message, exception=None):
    """Log error messages
    
    Args:
        loggers: Dictionary of logger objects
        message: Error message
        exception: Optional exception object
    """
    if exception:
        loggers[LoggerNames.ERROR].error(f"{message}: {str(exception)}", exc_info=True)
    else:
        loggers[LoggerNames.ERROR].error(message)
    
    # Also log to system log
    loggers[LoggerNames.SYSTEM].error(message)
