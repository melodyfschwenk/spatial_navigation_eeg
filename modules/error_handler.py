"""
Error Handler Module for Spatial Navigation EEG Experiment
=========================================================

This module provides error handling functions to prevent crashes.
"""

import logging
import traceback
import sys
from psychopy import visual, core, event


def safe_run(func, *args, **kwargs):
    """Run a function with error handling to prevent crashes
    
    Args:
        func: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function or None if an error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        logging.error(traceback.format_exc())
        
        # Try to show error dialog if a window is provided
        window = kwargs.get('window')
        if window and isinstance(window, visual.Window):
            try:
                error_text = visual.TextStim(
                    window, 
                    text=f"ERROR: {str(e)}\n\nPress ESCAPE to quit or SPACE to try to continue",
                    color=(1, -0.6, -0.6),  # Red
                    height=0.05,
                    wrapWidth=1.8
                )
                error_text.draw()
                window.flip()
                
                keys = event.waitKeys(keyList=['escape', 'space'])
                if 'escape' in keys:
                    core.quit()
            except:
                pass  # If showing the error dialog itself fails, just continue
        
        return None


def setup_error_handling(log_file=None):
    """Set up detailed error logging
    
    Args:
        log_file: Path to the log file
    """
    if log_file:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Set up global exception handler
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Let the default handler also do its thing
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = global_exception_handler
