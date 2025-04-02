#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Entry Point for Spatial Navigation EEG Experiment
======================================================

This script initializes and runs the Spatial Navigation EEG experiment.
Place this file in the root directory of your project, alongside the 'modules' folder.
"""

import os
import sys
import logging
import time
from psychopy import core

# Ensure modules directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Initialize pygame first to avoid "video system not initialized" errors
try:
    import pygame
    if not pygame.get_init():
        pygame.init()
        logging.info("Pygame initialized at startup")
except ImportError:
    logging.warning("Pygame not found - controller support may be limited")
except Exception as e:
    logging.warning(f"Error initializing pygame: {e}")

# Import required modules
from modules.config import Config
from modules.ui import get_participant_info, create_experiment_window, show_text_screen
from modules.data_handler import create_data_file
from modules.stimulus import load_stimulus_mapping
from modules.experiment import initialize_eeg_system, run_experiment_blocks
from modules.error_handler import setup_error_handling
from modules.utils import check_system_compatibility, verify_stimuli_paths, select_monitor
from modules.controller import (
    initialize_controller, 
    update_controller_globals, 
    ensure_controller_processing, 
    controller_available,
    setup_controller_monitoring
)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to run the experiment"""
    try:
        # Set up error handling
        log_file = os.path.join('logs', f'experiment_{core.getAbsTime()}.log')
        setup_error_handling(log_file)
        
        logging.info("Starting Spatial Navigation EEG Experiment")
        
        # Check system compatibility
        if not check_system_compatibility():
            logging.error("System compatibility check failed")
            print("System compatibility check failed. See log for details.")
            return
            
        # Initialize configuration
        config = Config()
        
        # Select monitor for display
        select_monitor(config)
        
        # Verify stimulus paths
        if not verify_stimuli_paths(config):
            logging.error("Stimulus path verification failed")
            print("Error: Stimulus files not found in the expected locations.")
            print("Please ensure all required files are in place before running the experiment.")
            return
        
        # Test TTL trigger system if parallel port is enabled
        if config.use_parallel:
            from modules.utils import test_brainvision_ttl_connection
            
            # Ask user if they want to test TTL triggers
            print("\nBrainVision TTL trigger system is enabled.")
            test_response = input("Would you like to test TTL triggers before starting? (y/n): ")
            
            if test_response.lower().startswith('y'):
                test_brainvision_ttl_connection()
                print("\nContinuing with experiment setup...")
        
        # Get participant information
        participant_info = get_participant_info()
        
        # Auto-assign counterbalance if "Auto" was selected
        if participant_info['counterbalance'] == 'Auto':
            from modules.utils import determine_counterbalance
            participant_info['counterbalance'] = determine_counterbalance(participant_info['participant_id'])
            print(f"Automatically assigned counterbalance: {participant_info['counterbalance']}")
        
        # Create the experiment window
        logging.info("Creating experiment window")
        window = create_experiment_window(config)
        
        # Store the config in the window for access by callbacks
        window.exp_handler = type('obj', (object,), {'config': config})
        
        # Setup universal escape mechanism EARLY in execution
        from modules.utils import setup_universal_escape
        setup_universal_escape(window)
        logging.info("Universal escape mechanism initialized - press ESC to exit anytime")
        
        # Initialize controller - UPDATED APPROACH
        try:
            # First create a loading message while initializing controller
            show_text_screen(
                window, 
                "Detecting gamepad controller...\n\nPlease wait.", 
                wait_for_key=False
            )
            
            # Initialize controller
            controller_result = initialize_controller()
            
            if controller_result:
                # Set controller preference in config
                config.use_controller = True
                
                # Show confirmation message without testing buttons
                show_text_screen(
                    window, 
                    "Controller detected and ready to use.\n\n"
                    "Press SPACE to continue.",
                    wait_for_key=True
                )
                
                # Skip the interactive test and directly set up controller monitoring
                logging.info("Controller detected, using controller for input")
                monitor_thread = setup_controller_monitoring(window)
            else:
                logging.warning("No controller detected, using keyboard input")
                config.use_controller = False
                
                # Show message
                show_text_screen(
                    window, 
                    "No gamepad controller detected.\n\nUsing keyboard controls instead.\n\n"
                    "Press SPACE to continue.",
                    wait_for_key=True
                )
        except Exception as e:
            logging.warning(f"Failed to initialize controller: {e}")
            config.use_controller = False
            
            # Show message
            show_text_screen(
                window, 
                f"Controller initialization failed: {str(e)}\n\n"
                "Using keyboard controls instead.\n\n"
                "Press SPACE to continue.",
                wait_for_key=True
            )
        
        # Create data file
        datafile = create_data_file(config, participant_info)
        
        # Load stimulus mapping
        stimulus_map = load_stimulus_mapping(config)
        
        if stimulus_map is None:
            logging.error("Failed to load stimulus mapping")
            print("Error: Could not load stimulus mapping. Please check the configuration.")
            window.close()
            return
        
        # Initialize EEG system
        eeg = initialize_eeg_system(config, participant_info)
        
        # Run experiment
        run_experiment_blocks(window, config, stimulus_map, participant_info, datafile, eeg)
        
        # Clean up and exit
        if window:
            window.close()
        
        # Clean up pygame
        try:
            pygame.quit()
        except:
            pass
        
        logging.info("Experiment finished successfully")
        print("Experiment complete. Thank you!")
    
    except Exception as e:
        logging.error(f"Experiment crashed: {e}", exc_info=True)
        print(f"Error: {e}")
        print("The experiment has encountered an error and must close.")
        print("Check the log files for details.")
    
    finally:
        # Final cleanup
        try:
            pygame.quit()
        except:
            pass
        core.quit()

if __name__ == "__main__":
    main()