#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities Module for Spatial Navigation EEG Experiment
=====================================================

This module contains utility functions for the experiment.
"""

import os
import logging
import random
import time
import socket
import numpy as np
from psychopy import event, core, visual, monitors, gui

from modules.controller import get_controller_input, controller_available, initialize_controller, update_controller_globals, controller


def check_system_compatibility():
    """Check if the system is compatible with the experiment requirements
    
    Returns:
        bool: True if system is compatible, False otherwise
    """
    try:
        # Check PsychoPy version
        from psychopy import __version__ as psychopy_version
        min_version = '2022.1.0'
        if psychopy_version < min_version:
            logging.warning(f"PsychoPy version {psychopy_version} may be outdated. Recommended: {min_version}+")
        
        # Check for required modules
        missing_modules = []
        try:
            import pandas
        except ImportError:
            missing_modules.append('pandas')
        
        try:
            import numpy
        except ImportError:
            missing_modules.append('numpy')
        
        try:
            from pylsl import StreamInfo
        except ImportError:
            missing_modules.append('pylsl')
        
        if missing_modules:
            logging.warning(f"Missing recommended modules: {', '.join(missing_modules)}")
            print(f"Warning: Missing recommended modules: {', '.join(missing_modules)}")
            if 'pylsl' in missing_modules:
                print("pylsl module not found. EEG triggers will be disabled.")
        
        return True
    
    except Exception as e:
        logging.error(f"System compatibility check failed: {e}")
        return False


def seed_random(seed=None):
    """Seed the random number generators for reproducibility
    
    Args:
        seed: Random seed (default: None, uses system time)
        
    Returns:
        int: The seed that was used
    """
    if seed is None:
        # Use current time as default seed
        seed = int(time.time())
    
    # Seed Python's random module
    random.seed(seed)
    
    # Seed NumPy's random module
    np.random.seed(seed)
    
    logging.info(f"Random number generators seeded with: {seed}")
    return seed


def verify_stimuli_paths(config):
    """Verify that all stimulus paths exist
    
    Args:
        config: The experiment configuration object
        
    Returns:
        bool: True if all paths exist, False otherwise
    """
    path_errors = []
    
    # Check stimulus mapping file
    if not os.path.exists(config.stimulus_map_path):
        path_errors.append(f"Stimulus mapping file not found: {config.stimulus_map_path}")
    
    # Check stimulus directories
    for difficulty, path in config.stim_paths.items():
        if not os.path.exists(path):
            path_errors.append(f"{difficulty.capitalize()} stimulus directory not found: {path}")
    
    # Log and report errors
    if path_errors:
        for error in path_errors:
            logging.error(error)
            print(f"Error: {error}")
        return False
    
    return True


def convert_response_to_direction(response, navigation_type):
    """Convert a key response to a movement direction
    
    Args:
        response: Key press ('up', 'down', 'left', 'right')
        navigation_type: 'egocentric', 'allocentric', or 'control'
        
    Returns:
        str: Direction as text
    """
    # For allocentric navigation, keys map directly to compass directions
    allo_map = {
        'up': 'north',
        'down': 'south',
        'left': 'west',
        'right': 'east'
    }
    
    # For egocentric navigation, keys map to relative directions
    ego_map = {
        'up': 'forward',
        'down': 'backward',
        'left': 'left',
        'right': 'right'
    }
    
    # For control navigation, keys directly map to visible arrows
    control_map = {
        'up': 'up',
        'down': 'down',
        'left': 'left',
        'right': 'right'
    }
    
    # Return the appropriate mapping
    if navigation_type == 'egocentric':
        return ego_map.get(response, 'unknown')
    elif navigation_type == 'control':
        return control_map.get(response, 'unknown')
    else:  # allocentric
        return allo_map.get(response, 'unknown')


def create_experiment_directory_structure(base_dir=None):
    """Create the experiment directory structure if it doesn't exist
    
    Args:
        base_dir: Base directory for the experiment (default: current directory)
        
    Returns:
        str: Base directory path
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    # Directories to create
    directories = [
        'data',
        'logs',
        'stimuli/easy',
        'stimuli/hard',
        'stimuli/control',
        'config',
        'modules',
        'tests'
    ]
    
    # Create each directory
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
    
    return base_dir


def verify_and_fix_config(config):
    """Verify and fix common configuration issues
    
    Args:
        config: The experiment configuration object
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Check for required attributes and add defaults if missing
    required_attributes = {
        'data_dir': os.path.join(os.getcwd(), 'data'),
        'log_dir': os.path.join(os.getcwd(), 'logs'),
        'screen_resolution': (1920, 1080),
        'background_color': (1, 1, 1),  # White
        'text_color': (-0.7, -0.7, -0.7),  # Dark gray for white background
        'intertrial_interval': (0.8, 1.2),
        'total_trials': 0
    }
    
    for attr, default_value in required_attributes.items():
        if not hasattr(config, attr):
            setattr(config, attr, default_value)
            logging.info(f"Added missing config attribute: {attr} = {default_value}")
    
    # Verify and fix paths
    if not os.path.exists(config.data_dir):
        try:
            os.makedirs(config.data_dir)
            logging.info(f"Created missing data directory: {config.data_dir}")
        except Exception as e:
            logging.error(f"Failed to create data directory: {e}")
            return False
    
    if not os.path.exists(config.log_dir):
        try:
            os.makedirs(config.log_dir)
            logging.info(f"Created missing log directory: {config.log_dir}")
        except Exception as e:
            logging.error(f"Failed to create log directory: {e}")
            return False
    
    # Initialize loggers attribute if not already present
    if not hasattr(config, 'loggers'):
        config.loggers = None
    
    # Calculate total trials if not already done
    if config.total_trials == 0:
        total_blocks = len(config.navigation_types) * len(config.difficulty_levels) * config.repetitions
        config.total_trials = total_blocks * config.trials_per_block
        logging.info(f"Calculated total trials: {config.total_trials}")
    
    # If BrainVision RCS failed to connect, disable it
    if hasattr(config, 'use_brainvision_rcs') and config.use_brainvision_rcs:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(2.0)  # Increased timeout for more reliable check
            try:
                test_socket.connect((config.brainvision_rcs_host, config.brainvision_rcs_port))
                test_socket.close()
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                logging.warning(f"BrainVision RCS not available ({str(e)}). Continuing without BrainVision triggers.")
                config.use_brainvision_rcs = False
            except Exception as e:
                logging.warning(f"Unexpected error when connecting to BrainVision RCS: {str(e)}. Continuing without BrainVision triggers.")
                config.use_brainvision_rcs = False
        except Exception as e:
            logging.warning(f"Failed to check BrainVision RCS: {str(e)}. Continuing without BrainVision triggers.")
            config.use_brainvision_rcs = False
    
    return True


def validate_response_mapping(stimulus_map):
    """Validate that response mappings in the stimulus file are consistent
    
    Args:
        stimulus_map: DataFrame containing stimulus mapping
        
    Returns:
        bool: True if mappings are valid, False otherwise
    """
    valid = True
    
    # Check egocentric responses
    ego_responses = set(stimulus_map['egocentric_correct_response'])
    invalid_ego = ego_responses - {'up', 'down', 'left', 'right'}
    if invalid_ego:
        logging.error(f"Invalid egocentric responses in stimulus mapping: {invalid_ego}")
        valid = False
    
    # Check allocentric responses
    allo_responses = set(stimulus_map['allocentric_correct_response'])
    invalid_allo = allo_responses - {'up', 'down', 'left', 'right'}
    if invalid_allo:
        logging.error(f"Invalid allocentric responses in stimulus mapping: {invalid_allo}")
        valid = False
    
    # Also check if there are any unexpected difficulties
    difficulties = set(stimulus_map['difficulty'])
    expected_difficulties = {'easy', 'hard', 'control'}
    invalid_diff = difficulties - expected_difficulties
    if invalid_diff:
        logging.warning(f"Unexpected difficulties in stimulus mapping: {invalid_diff}")
    
    return valid


def select_monitor(config):
    """Allow the user to select which monitor to use for the experiment
    
    Args:
        config: The experiment configuration object
        
    Returns:
        int: Selected monitor index
    """
    try:
        # Get information about connected monitors
        monitor_info = []
        
        try:
            # Try to get detailed monitor info using the monitors module
            for i in range(10):  # Check up to 10 monitors
                try:
                    mon = monitors.Monitor(f"monitor{i}")
                    if mon.getSizePix() is not None:
                        monitor_info.append({
                            'index': i,
                            'name': f"Monitor {i}",
                            'resolution': mon.getSizePix(),
                            'size': mon.getWidth()
                        })
                except:
                    pass
        except:
            # Fallback if detailed info is unavailable
            pass
            
        # If no monitors detected, provide default options
        if not monitor_info:
            monitor_info = [
                {'index': 0, 'name': "Primary monitor"},
                {'index': 1, 'name': "Secondary monitor"}
            ]
            # Add a few more potential monitors
            for i in range(2, 4):
                monitor_info.append({'index': i, 'name': f"Monitor {i}"})
        
        # Create a dialog to select the monitor
        monitor_dict = {
            'Monitor': [m['name'] for m in monitor_info]
        }
        
        # Add options for controller/gamepad
        monitor_dict['Use controller/gamepad'] = True
        
        # Add a note about which is likely the primary
        monitor_dict['Note'] = "Monitor 0 is typically the primary display"
        
        dlg = gui.DlgFromDict(
            dictionary=monitor_dict,
            title='Select Display Monitor',
            fixed=['Note']
        )
        
        if dlg.OK:
            selected_name = monitor_dict['Monitor']
            for m in monitor_info:
                if m['name'] == selected_name:
                    selected_index = m['index']
                    
                    # Update config
                    config.monitor_index = selected_index
                    
                    # Set controller preference
                    config.use_controller = monitor_dict['Use controller/gamepad']
                    
                    # Also update screen resolution if available
                    if 'resolution' in m:
                        config.screen_resolution = m['resolution']
                    
                    logging.info(f"Selected monitor {selected_index}: {selected_name}")
                    logging.info(f"Using controller/gamepad: {config.use_controller}")
                    return selected_index
            
            # If not found (shouldn't happen), default to 0
            config.monitor_index = 0
            logging.warning("Could not find selected monitor, defaulting to monitor 0")
            return 0
        else:
            # User cancelled, default to 0
            config.monitor_index = 0
            logging.info("Monitor selection cancelled by user, defaulting to monitor 0")
            return 0
    except Exception as e:
        # In case of any error, default to monitor 0
        config.monitor_index = 0
        logging.error(f"Error selecting monitor: {e}. Defaulting to monitor 0")
        return 0

def get_input(config, max_wait=None, allowed_keys=None, return_rt=False, clear_events=True):
    """Consolidated input handler that works with both keyboard and controller
    
    This function replaces all previous input handlers with a single, reliable
    implementation that can be used throughout the experiment.
    
    Args:
        config: The experiment configuration object
        max_wait: Maximum wait time in seconds (None for infinite wait)
        allowed_keys: List of allowed keys (default: space, escape)
        return_rt: Whether to return reaction time (default: False)
        clear_events: Whether to clear events before waiting (default: True)
        
    Returns:
        If return_rt is False: str or None - Key pressed or None if timeout
        If return_rt is True: tuple (key, rt) - Key pressed and reaction time, or (None, None) if timeout
    """
    import time
    from psychopy import event, core
    
    # Set default allowed keys if none provided
    if allowed_keys is None:
        allowed_keys = ['space', 'escape']
    
    # Create a clock for tracking reaction time
    response_clock = core.Clock()
    
    # Important: Clear any pending events if requested
    if clear_events:
        event.clearEvents()
        # Small delay to ensure events are fully cleared
        core.wait(0.2)
        # Clear again after delay
        event.clearEvents()
    
    # Initialize controller input components if needed
    controller_enabled = (hasattr(config, 'use_controller') and 
                         config.use_controller)
    
    if controller_enabled:
        try:
            from modules.controller import controller_available, get_controller_input, ensure_controller_processing
            has_controller = controller_available
        except:
            has_controller = False
    else:
        has_controller = False
    
    # Check if pygame is being used as the backend (this happens when controller is enabled)
    using_pygame_backend = False
    try:
        from psychopy.hardware import joystick
        using_pygame_backend = hasattr(joystick, 'backend') and joystick.backend == 'pygame'
    except:
        pass
    
    # Reset the clock for RT measurement
    response_clock.reset()
    start_time = time.time()
    
    # Main input loop
    while max_wait is None or time.time() - start_time < max_wait:
        # Check controller input if available
        if controller_enabled and has_controller:
            try:
                # Process controller events
                ensure_controller_processing()
                
                # Get controller input
                controller_input = get_controller_input()
                
                # Check D-pad input
                if controller_input['dpad'] in allowed_keys:
                    rt = response_clock.getTime()
                    if return_rt:
                        return controller_input['dpad'], rt
                    else:
                        return controller_input['dpad']
                
                # Check common button mappings
                if 'space' in allowed_keys and 'x' in controller_input['buttons']:
                    rt = response_clock.getTime()
                    if return_rt:
                        return 'space', rt
                    else:
                        return 'space'
                
                if 'return' in allowed_keys and 'a' in controller_input['buttons']:
                    rt = response_clock.getTime()
                    if return_rt:
                        return 'return', rt
                    else:
                        return 'return'
                
                if 'escape' in allowed_keys and 'b' in controller_input['buttons']:
                    rt = response_clock.getTime()
                    if return_rt:
                        return 'escape', rt
                    else:
                        return 'escape'
            except Exception as e:
                import logging
                logging.debug(f"Controller input error: {e}")
        
        # Check keyboard input - handle differently based on backend
        if using_pygame_backend:
            # With pygame backend, we need to avoid using timeStamped
            keys = event.getKeys()
            if keys:
                for key in keys:
                    if key in allowed_keys:
                        if return_rt:
                            # Calculate RT manually using our clock
                            return key, response_clock.getTime()
                        else:
                            return key
        else:
            # For other backends, we can use timeStamped parameter
            keys = event.getKeys(timeStamped=return_rt)
            if keys:
                for key_data in keys:
                    key = key_data[0] if return_rt else key_data
                    if key in allowed_keys:
                        if return_rt:
                            # Return key and RT from timeStamped
                            return key, key_data[1]
                        else:
                            return key
        
        # Short pause to prevent CPU hogging
        core.wait(0.001)
    
    # If we reached here, it's a timeout
    if return_rt:
        return None, None
    else:
        return None


# Wrapper functions for backward compatibility

def get_response(config, max_wait=None, allowed_keys=None):
    """Backward compatibility wrapper for get_input with RT
    
    Args:
        config: The experiment configuration object
        max_wait: Maximum wait time in seconds
        allowed_keys: List of allowed keys
        
    Returns:
        tuple: (response, rt) or (None, None) if no response
    """
    return get_input(config, max_wait=max_wait, allowed_keys=allowed_keys, return_rt=True)


def get_input_with_rt(config, max_wait=None, allowed_keys=None):
    """Backward compatibility wrapper for get_input with RT
    
    Args:
        config: The experiment configuration object
        max_wait: Maximum wait time in seconds
        allowed_keys: List of allowed keys
        
    Returns:
        tuple: (response, rt) or (None, None) if no response
    """
    return get_input(config, max_wait=max_wait, allowed_keys=allowed_keys, return_rt=True, clear_events=True)

def emergency_escape(message="User initiated emergency exit", window=None, eeg=None, loggers=None):
    """Immediately terminate the experiment with proper cleanup
    
    This function provides a universal way to exit the experiment from 
    anywhere in the code, with proper cleanup of resources.
    
    Args:
        message: Message to log about why the experiment is terminating
        window: PsychoPy window object to close (optional)
        eeg: EEG system to shut down cleanly (optional)
        loggers: Logger objects to record the exit (optional)
    """
    # Log the emergency exit
    logging.warning(f"EMERGENCY EXIT: {message}")
    
    # If we have advanced loggers, use them
    if loggers and 'error' in loggers:
        from modules.advanced_logging import log_error
        log_error(loggers, f"EMERGENCY EXIT: {message}")
    
    # Show a message to the participant if window exists
    if window and hasattr(window, 'flip'):
        try:
            exit_text = visual.TextStim(
                window,
                text="Exiting experiment...\nPlease wait.",
                color=(-0.7, -0.7, -0.7),
                height=0.07,
                wrapWidth=1.8
            )
            exit_text.draw()
            window.flip()
        except Exception as e:
            logging.error(f"Failed to show exit message: {e}")
    
    # Close EEG system if it exists
    if eeg:
        try:
            # Send a final marker indicating emergency termination
            eeg.send_trigger(999, "Emergency experiment termination", 
                            {'reason': message, 'timestamp': time.time()})
            eeg.close()
            logging.info("EEG system closed successfully")
        except Exception as e:
            logging.error(f"Failed to close EEG system cleanly: {e}")
    
    # Close the window if it exists
    if window:
        try:
            window.close()
            logging.info("Window closed successfully")
        except Exception as e:
            logging.error(f"Failed to close window: {e}")
    
    # Exit immediately
    logging.info("Terminating experiment")
    core.quit()


def check_controller_escape():
    """Check if controller escape combination is pressed
    
    Returns:
        bool: True if escape combination detected, False otherwise
    """
    if not controller_available:
        return False
    
    try:
        # Import to avoid circular imports
        from modules.controller import check_emergency_exit_combo, ensure_controller_processing
        
        # Ensure controller events are processed
        ensure_controller_processing()
        
        # Use the controller function to check for emergency exit
        return check_emergency_exit_combo()
    except Exception as e:
        logging.error(f"Error checking controller escape: {e}")
        return False


def setup_escape_handler(window=None, eeg=None, loggers=None):
    """Set up a global key handler to check for the escape key and controller escape commands
    
    This function sets up a background thread that continuously checks
    for the escape key and controller escape combinations, and calls the 
    emergency_escape function if detected.
    
    Escape commands:
    - Keyboard: ESCAPE key
    - Controller: START+B or SELECT+B button combinations
    
    Args:
        window: PsychoPy window object to close (optional)
        eeg: EEG system to shut down cleanly (optional)
        loggers: Logger objects to record the exit (optional)
    """
    from psychopy import event
    import threading
    
    # Define a function to check for escape key
    # FIXED: Added *args, **kwargs to accept arguments passed by PsychoPy's event system
    def check_escape(*args, **kwargs):
        keys = event.getKeys(['escape'])
        if 'escape' in keys:
            emergency_escape("Escape key pressed", window, eeg, loggers)
    
    # FIX FOR DUPLICATE KEY ISSUE - Check if escape is already registered
    try:
        # Check if escape key is already registered
        if 'escape' not in event.globalKeys:
            # Register the escape checker function with the psychopy event module
            event.globalKeys.add('escape', check_escape, "Escape experiment")
            logging.info("Global escape key handler registered")
        else:
            logging.info("Escape key already registered, skipping duplicate registration")
    except Exception as e:
        logging.warning(f"Could not register escape key handler: {e}")
    
    # Create a thread to monitor controller escape combinations
    if controller_available:
        def controller_monitor():
            while True:
                if check_controller_escape():
                    emergency_escape("Controller escape combination pressed", window, eeg, loggers)
                    break
                # Check every 100ms to reduce CPU usage
                time.sleep(0.1)
        
        # Start the controller monitoring thread as daemon (will exit when main thread exits)
        controller_thread = threading.Thread(target=controller_monitor, daemon=True)
        controller_thread.start()
        logging.info("Controller escape monitor started (press START+B or SELECT+B to exit)")
    
    logging.info("Global escape handler set up (press ESCAPE key to terminate experiment)")


def visualize_block_sequence(config, participant_id=None, counterbalance=None):
    """Visualize possible block sequences for experiment planning
    
    Args:
        config: The experiment configuration object
        participant_id: Optional participant ID for reproducible visualization
        counterbalance: Optional counterbalance condition; if None, shows all conditions
    
    Returns:
        str: ASCII visualization of block sequence(s)
    """
    # If no specific counterbalance is provided, show all 4 Latin Square conditions
    if counterbalance is None:
        conditions = [1, 2, 3, 4]
    else:
        conditions = [counterbalance]
    
    results = []
    
    for cond in conditions:
        # Get the sequence for this counterbalance condition
        sequence = config.create_block_sequence(cond, participant_id)
        
        # Create header for this condition
        header = f"COUNTERBALANCE {cond} (Latin Square Design)"
        
        # Create separator
        sep = "-" * len(header)
        
        # Add headers to results
        results.append(header)
        results.append(sep)
        
        # Create formatted sequence representation
        block_viz = []
        for i, (nav_type, diff, block_num) in enumerate(sequence):
            # Create short codes for navigation and difficulty
            nav_code = nav_type[:4].upper()
            diff_code = diff[0].upper()
            
            # Create block representation with navigation and difficulty
            block_repr = f"[{nav_code}/{diff_code}]"
            
            # Add block number
            block_viz.append(f"{block_num:2d}: {block_repr}")
        
        # Add block visualization to results
        # Format as a table with 5 blocks per row
        for i in range(0, len(block_viz), 5):
            row = "  ".join(block_viz[i:i+5])
            results.append(row)
        
        # Add empty line between conditions
        results.append("")
    
    # Join all results and return
    return "\n".join(results)


def print_block_sequence(config, participant_id=None, counterbalance=None):
    """Print a visualization of the block sequence
    
    Args:
        config: Experiment configuration object
        participant_id: Optional participant ID for reproducible sequence
        counterbalance: Counterbalance condition (1-4)
    """
    # Use default counterbalance if none provided
    if counterbalance is None:
        counterbalance = 1
        print(f"No counterbalance specified, using default: {counterbalance}")
    
    # Generate block sequence
    block_sequence = config.create_block_sequence(counterbalance, participant_id)
    
    # Calculate total blocks
    total_blocks = len(block_sequence)
    
    # Print header
    print("\n" + "="*80)
    print(f"BLOCK SEQUENCE (Counterbalance: {counterbalance}, Participant: {participant_id or 'Not specified'})")
    print("="*80)
    
    # Display blocks with color coding for navigation type
    prev_nav_type = None
    for i, (nav_type, difficulty, block_num) in enumerate(block_sequence):
        # Check if navigation type changed
        nav_changed = prev_nav_type is not None and nav_type != prev_nav_type
        
        # Format the block info
        block_info = f"Block {block_num}/{total_blocks}: {nav_type.upper()} - {difficulty}"
        
        # Add transition indicator if navigation type changed
        if nav_changed:
            print(f"  {'↓'*30} Navigation Change {'↓'*30}")
        
        # Print the block info
        print(f"  {block_info}")
        
        # Update previous navigation type
        prev_nav_type = nav_type
    
    print("="*80)
    
    # Summarize navigation transitions
    transitions = sum(1 for i in range(1, len(block_sequence)) 
                     if block_sequence[i][0] != block_sequence[i-1][0])
    print(f"Navigation transitions: {transitions}")
    
    # Summarize by condition
    print("\nBlock counts by condition:")
    condition_counts = {}
    for nav_type, difficulty, _ in block_sequence:
        key = f"{nav_type}/{difficulty}"
        if key not in condition_counts:
            condition_counts[key] = 0
        condition_counts[key] += 1
    
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} blocks")
    
    print()

def setup_universal_escape(window, eeg=None):
    """Set up a universal escape mechanism that works throughout the experiment
    
    This function sets up a global event hook for escape key and controller escape combo
    
    Args:
        window: PsychoPy window object
        eeg: Optional EEG system for clean shutdown
    """
    from psychopy import event
    import threading
    import time
    from modules.controller import get_controller_input, controller_available
    
    # Define the escape function
    def escape_function():
        emergency_escape("Universal escape activated", window, eeg)
    
    # Register global escape key handler - FIXED SYNTAX for backward compatibility
    event.globalKeys.clear()  # Clear any existing global keys
    try:
        # Use the older syntax for PsychoPy (positional arguments)
        event.globalKeys.add('escape', escape_function, name='shutdown')
        logging.info("Universal escape key registered successfully")
    except TypeError as e:
        # If that fails, try an even more basic approach
        logging.warning(f"Couldn't register escape key with standard method: {e}")
        try:
            # Register with psychopy's event system directly
            event.globalKeys.clear()
            event.globalKeys.add('escape', func=emergency_escape, 
                             func_args=["User pressed escape key", window, eeg])
            logging.info("Universal escape key registered with alternate method")
        except Exception as e2:
            logging.error(f"Failed to register escape key: {e2}. Emergency exit will still work with manual handling.")
    
    # Set up controller escape mechanism
    if controller_available:
        def check_controller_escape():
            while True:
                try:
                    # Import inside the thread to avoid circular imports
                    from modules.controller import update_controller_globals, get_controller_input
                    
                    # Process controller events
                    update_controller_globals()
                    
                    # Get controller input
                    controller_input = get_controller_input()
                    
                    # Check for START+B or SELECT+B combinations
                    if (('start' in controller_input['buttons'] and 'b' in controller_input['buttons']) or
                        ('select' in controller_input['buttons'] and 'b' in controller_input['buttons'])):
                        emergency_escape("Controller escape combination activated", window, eeg)
                    
                    # Brief pause to prevent CPU hogging
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Error in controller escape thread: {e}")
                    time.sleep(1.0)  # Wait longer on error
                    
        # Start controller escape thread as daemon
        controller_thread = threading.Thread(target=check_controller_escape, daemon=True)
        controller_thread.start()
        logging.info("Controller escape monitor started - press START+B or SELECT+B to exit")
    
    logging.info("Universal escape mechanism set up - press ESC key to exit at any time")


def send_ttl_trigger(trigger_value, duration=0.001):
    """Send a direct TTL trigger via parallel port with minimal overhead
    
    This function provides a direct, low-level access to send TTL triggers
    when you need precise timing control outside the EEG object.
    
    Args:
        trigger_value: Integer trigger value (0-255)
        duration: How long to hold the trigger high in seconds
        
    Returns:
        bool: True if trigger was sent successfully
    """
    try:
        from psychopy import parallel
        
        # Ensure value is valid
        trigger_value = min(255, max(0, int(trigger_value)))
        
        # Send 0 to clear any previous triggers
        parallel.setData(0)
        
        # Very short delay before setting new value
        time.sleep(0.0001)  # 0.1ms
        
        # Send trigger
        parallel.setData(trigger_value)
        
        # Hold for specified duration
        time.sleep(duration)
        
        # Reset to 0
        parallel.setData(0)
        
        return True
    except Exception as e:
        logging.error(f"Failed to send TTL trigger {trigger_value}: {e}")
        return False
        
def test_brainvision_ttl_connection():
    """Test BrainVision TTL trigger connection using parallel port
    
    Returns:
        bool: True if connection test was successful
    """
    try:
        from modules.eeg import test_ttl_triggers
        
        print("Testing BrainVision TTL trigger connection...")
        print("You should see trigger markers (value 254) appear in BrainVision Recorder")
        print("Press Ctrl+C to cancel test if needed\n")
        
        # Run the test function
        return test_ttl_triggers(num_triggers=3, interval=1.0)
    except Exception as e:
        logging.error(f"BrainVision TTL test failed: {e}")
        print(f"TTL test failed: {e}")
        return False


def determine_counterbalance(participant_id):
    """Automatically determine counterbalance condition from participant ID
    
    This function ensures consistent counterbalancing across participants
    while distributing conditions evenly.
    
    Args:
        participant_id: Participant identifier (string or number)
        
    Returns:
        int: Counterbalance condition (1-4)
    """
    try:
        # Try to convert participant_id to integer
        numeric_id = int(participant_id)
    except (ValueError, TypeError):
        # If participant_id is not a valid integer,
        # create a numeric hash based on string characters
        numeric_id = sum(ord(c) for c in str(participant_id))
    
    # Use modulo to get a value between 0-3, then add 1 for 1-4 range
    counterbalance = (numeric_id % 4) + 1
    
    logging.info(f"Automatically determined counterbalance {counterbalance} for participant {participant_id}")
    return counterbalance


