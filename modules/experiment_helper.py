#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment Helper Module for Spatial Navigation EEG Experiment
=============================================================

This module provides helper functions for the experiment execution,
particularly for handling controller input and processing during experiment blocks.
"""

import time
import logging
import pyglet
from psychopy import core

from modules.controller import check_controller_health, ensure_controller_processing, reset_controller

def process_events_during_experiment(window, interval=0.1, max_duration=None):
    """Process events during experiment phases to keep controller responsive
    
    Call this function during experiment phases where waiting happens,
    to ensure controller keeps receiving events.
    
    Args:
        window: PsychoPy window
        interval: How often to process events (seconds)
        max_duration: Maximum duration to process events (None for indefinite)
        
    Returns:
        bool: True if processing was successful without interruption
    """
    start_time = time.time()
    
    try:
        while max_duration is None or time.time() - start_time < max_duration:
            # Process pygame events to update controller state
            ensure_controller_processing()
            
            # Yield to allow other processing
            core.wait(interval)
            
            # Check for experiment exit keys, etc. here if needed
            
        return True
    except Exception as e:
        logging.error(f"Error during event processing: {e}")
        return False

def maintain_controller_health(window, check_interval=5.0):
    """Set up a background process to maintain controller health
    
    This function can be called at the start of experiment blocks to
    ensure controller stays responsive throughout.
    
    Args:
        window: PsychoPy window
        check_interval: How often to check controller health (seconds)
        
    Returns:
        dict: Timer info that can be used to stop the checking
    """
    last_check = time.time()
    
    def health_check():
        nonlocal last_check
        current_time = time.time()
        
        # Only check at specified intervals
        if current_time - last_check >= check_interval:
            # Check controller health
            if not check_controller_health():
                logging.warning("Controller health check failed, attempting reset...")
                reset_controller(window)
            last_check = current_time
    
    # Return the health check function that can be called periodically
    return {'func': health_check, 'last_check': last_check}
