#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Controller Module for Spatial Navigation EEG Experiment
======================================================

This module handles gamepad/controller input for the experiment,
including setup, detection, and button mapping.
Updated to use pygame backend for improved Logitech controller support.
"""

import logging
import time
import sys
from psychopy.hardware import joystick
from psychopy import visual, event, core

# Define global variables for controller
controller = None
controller_available = False

# Define button mappings for Logitech F310 controller with pygame backend
BUTTON_A = 0        # A (bottom button)
BUTTON_B = 1        # B (right button)
BUTTON_X = 2        # X (left button)
BUTTON_Y = 3        # Y (top button)
BUTTON_LEFT_SHOULDER = 4   # Left bumper/shoulder button
BUTTON_RIGHT_SHOULDER = 5  # Right bumper/shoulder button
BUTTON_BACK = 6     # Back/Select button
BUTTON_START = 7    # Start button
BUTTON_LEFT_STICK = 8      # Left stick press
BUTTON_RIGHT_STICK = 9     # Right stick press

# D-pad mapping for hat values (pygame uses a different format than pyglet)
# FIXED: Inverted the D-pad mappings to correctly interpret your controller
DPAD_UP = (0, 1)       # Inverted value for UP
DPAD_DOWN = (0, -1)    # Inverted value for DOWN
DPAD_LEFT = (-1, 0)    # Unchanged
DPAD_RIGHT = (1, 0)    # Unchanged

# Create expanded hat mapping with all possible D-pad values
hat_mapping = {
    DPAD_UP: "up",
    DPAD_DOWN: "down",
    DPAD_LEFT: "left",
    DPAD_RIGHT: "right",
    # Add these diagonal mappings
    (1, 1): "up-right",    
    (-1, 1): "up-left",    
    (1, -1): "down-right", 
    (-1, -1): "down-left", 
    (0, 0): None           # Center position (no direction)
}

# ADDED: Mapping for primary direction from diagonals
diagonal_to_primary = {
    "up-right": "up",
    "up-left": "up",
    "down-right": "down",
    "down-left": "down"
}

# Button name mappings for easier reference
button_mapping = {
    BUTTON_A: "a",
    BUTTON_B: "b",
    BUTTON_X: "x",
    BUTTON_Y: "y",
    BUTTON_BACK: "back",
    BUTTON_START: "start",
    BUTTON_LEFT_SHOULDER: "left_shoulder",
    BUTTON_RIGHT_SHOULDER: "right_shoulder",
    BUTTON_LEFT_STICK: "left_stick",
    BUTTON_RIGHT_STICK: "right_stick"
}

def initialize_controller():
    """Initialize gamepad/controller using pygame backend
    
    Returns:
        bool: True if controller is available, False otherwise
    """
    global controller, controller_available
    
    try:
        logging.info("Initializing controller with pygame backend")
        
        # Initialize pygame first to prevent "video system not initialized" errors
        import pygame
        if not pygame.get_init():
            pygame.init()
            logging.info("Pygame initialized for controller")
        
        # Set pygame backend for joystick
        joystick.backend = 'pygame'
        
        # Initialize pygame.joystick module
        pygame.joystick.init()
        
        # Get number of available controllers
        n_controllers = joystick.getNumJoysticks()
        
        if n_controllers > 0:
            # Initialize the first controller
            controller = joystick.Joystick(0)
            controller_name = controller.getName()
            num_buttons = controller.getNumButtons()
            num_axes = controller.getNumAxes()
            num_hats = controller.getNumHats()
            
            logging.info(f"Controller detected using pygame: {controller_name}")
            logging.info(f"Number of buttons: {num_buttons}")
            logging.info(f"Number of axes: {num_axes}")
            logging.info(f"Number of hats: {num_hats}")
            
            # Process pygame events immediately to establish connection
            pygame.event.pump()
            
            # Test if the controller is actually responding
            if test_controller_basic():
                controller_available = True
                logging.info("Controller successfully initialized and responding")
                
                # Print detailed controller info
                if num_buttons > 0:
                    for i in range(min(num_buttons, 15)):  # Limit to 15 buttons to avoid excessive logging
                        logging.debug(f"Button {i} available: {controller.getButton(i)}")
                
                return True
            else:
                logging.warning("Controller not responding to basic tests, falling back to keyboard")
                controller_available = False
                return False
        else:
            logging.warning("No gamepad/controller detected. Falling back to keyboard input.")
            controller_available = False
            return False
    
    except Exception as e:
        logging.error(f"Failed to initialize controller with pygame: {e}")
        controller_available = False
        return False
    
def test_controller_basic():
    """Basic test to ensure the controller is responding
    
    Returns:
        bool: True if controller is working, False otherwise
    """
    global controller
    
    if controller is None:
        return False
    
    try:
        # Process events to ensure controller state is updated
        ensure_controller_processing()
        
        # Try to read basic controller properties
        num_buttons = controller.getNumButtons()
        num_axes = controller.getNumAxes()
        num_hats = controller.getNumHats()
        
        # Make sure we can read button states
        for i in range(min(num_buttons, 10)):  # Test first 10 buttons or less
            _ = controller.getButton(i)
        
        # Test hat reading if available
        if num_hats > 0:
            _ = controller.getHat(0)
        
        return True
    except Exception as e:
        logging.error(f"Controller basic test failed: {e}")
        return False

def get_controller_input():
    """Get input from controller using pygame backend
    
    Returns:
        dict: Dictionary with controller input state
    """
    if not controller_available or controller is None:
        return {'dpad': None, 'buttons': []}
    
    result = {'dpad': None, 'buttons': []}
    
    try:
        # Ensure events are processed before checking input
        ensure_controller_processing()
        
        # Check hat (D-pad) inputs - FIXED for more reliable detection
        if controller.getNumHats() > 0:
            hat_value = controller.getHat(0)  # Get first hat (D-pad)
            
            # Log the raw hat value for debugging
            if hat_value != (0, 0):
                logging.debug(f"Raw D-pad/hat value: {hat_value}")
            
            # Convert hat value to direction name
            dpad_value = hat_mapping.get(hat_value)
            if dpad_value:
                # Handle diagonal directions
                if dpad_value in diagonal_to_primary:
                    result['dpad'] = diagonal_to_primary[dpad_value]
                    logging.debug(f"Diagonal D-pad input: {dpad_value} simplified to {result['dpad']}")
                else:
                    result['dpad'] = dpad_value
                    # Debug log when D-pad is used
                    logging.debug(f"D-pad input detected: {dpad_value} from hat value {hat_value}")
        
        # Check button inputs
        for button_idx, button_name in button_mapping.items():
            if button_idx < controller.getNumButtons() and controller.getButton(button_idx):
                result['buttons'].append(button_name)
                # Debug log when button is pressed
                logging.debug(f"Controller button pressed: {button_name} (index {button_idx})")
        
        # If D-pad not working, try to use axes for D-pad emulation on some controllers
        # IMPROVED: More aggressive axis checking with lower threshold
        if result['dpad'] is None and controller.getNumAxes() >= 2:
            # Use first two axes (typically the left stick)
            x_axis = controller.getAxis(0)
            y_axis = controller.getAxis(1)
            
            # Apply a threshold to handle small movements (REDUCED threshold)
            threshold = 0.3  # More sensitive threshold
            
            # Convert axis values to D-pad directions
            if y_axis < -threshold:
                result['dpad'] = 'up'
                logging.debug(f"D-pad UP emulated from axis: ({x_axis}, {y_axis})")
            elif y_axis > threshold:
                result['dpad'] = 'down'
                logging.debug(f"D-pad DOWN emulated from axis: ({x_axis}, {y_axis})")
            elif x_axis < -threshold:
                result['dpad'] = 'left'
                logging.debug(f"D-pad LEFT emulated from axis: ({x_axis}, {y_axis})")
            elif x_axis > threshold:
                result['dpad'] = 'right'
                logging.debug(f"D-pad RIGHT emulated from axis: ({x_axis}, {y_axis})")
        
        # ADDED: Emergency fallback - map shoulder buttons to directions if D-pad doesn't work
        if result['dpad'] is None and 'left_shoulder' in result['buttons']:
            result['dpad'] = 'left'
            logging.debug("D-pad LEFT emulated from LEFT SHOULDER button")
        elif result['dpad'] is None and 'right_shoulder' in result['buttons']:
            result['dpad'] = 'right'
            logging.debug("D-pad RIGHT emulated from RIGHT SHOULDER button")
        elif result['dpad'] is None and 'y' in result['buttons']:
            result['dpad'] = 'up'
            logging.debug("D-pad UP emulated from Y button")
        elif result['dpad'] is None and 'a' in result['buttons']:
            result['dpad'] = 'down'
            logging.debug("D-pad DOWN emulated from A button")
    
    except Exception as e:
        logging.error(f"Error getting controller input: {e}")
    
    return result

def wait_for_controller_press(allowed_buttons=None, allowed_dpad=None, timeout=None):
    """Wait for controller button press
    
    Args:
        allowed_buttons: List of allowed buttons (e.g., ['a', 'b', 'x', 'y'])
        allowed_dpad: List of allowed D-pad directions (e.g., ['up', 'down', 'left', 'right'])
        timeout: Timeout in seconds (None for no timeout)
        
    Returns:
        tuple: (input_type, input_value, time_waited) or (None, None, time_waited) if timeout
    """
    global controller, controller_available
    
    if not controller_available:
        return (None, None, 0)
    
    if allowed_buttons is None:
        allowed_buttons = ['a', 'b', 'x', 'y']
    
    if allowed_dpad is None:
        allowed_dpad = ['up', 'down', 'left', 'right']
    
    # Create mapping from button names to indices
    button_indices = {name: idx for idx, name in button_mapping.items()}
    
    # Start timer - FIXED: Define start_time
    start_time = time.time()
    
    # Main input detection loop
    while timeout is None or time.time() - start_time < timeout:
        # Ensure controller events are processed
        ensure_controller_processing()
        
        # Check D-pad
        if controller.getNumHats() > 0:
            hat_value = controller.getHat(0)
            direction = hat_mapping.get(hat_value)
            if direction in allowed_dpad:
                return ('dpad', direction, time.time() - start_time)
        
        # Check buttons directly
        for button_name in allowed_buttons:
            if button_name in button_indices:
                button_idx = button_indices[button_name]
                if button_idx < controller.getNumButtons() and controller.getButton(button_idx):
                    return ('button', button_name, time.time() - start_time)
        
        # Sleep to reduce CPU usage
        time.sleep(0.01)
        
        # Periodically check if controller is still connected
        if time.time() - start_time > 1.0 and not check_controller_health():
            controller_available = False
            return (None, None, time.time() - start_time)
    
    # Timeout
    return (None, None, time.time() - start_time)

def test_controller_connection(window):
    """Test controller connection and show button mapping
    
    Args:
        window: PsychoPy window object
        
    Returns:
        bool: True if controller is working, False otherwise
    """
    global controller_available
    
    if not controller_available:
        logging.info("No controller available, skipping connection test")
        return False
    
    # Create test screen
    test_text = visual.TextStim(
        window,
        text=(
            "Controller Detected!\n\n"
            "Please press each button to verify connections:\n\n"
            "1. Press X button (left face button)\n"
            "2. Press D-pad UP, DOWN, LEFT, RIGHT\n\n"
            "Press A button (bottom face button) when finished"
        ),
        height=0.05,
        wrapWidth=1.8
    )
    
    button_status = {
        'x_pressed': False,
        'dpad_up': False,
        'dpad_down': False,
        'dpad_left': False,
        'dpad_right': False,
        'a_pressed': False
    }
    
    feedback_text = visual.TextStim(
        window,
        text="",
        pos=(0, -0.4),
        height=0.04,
        color="yellow"
    )
    
    waiting = True
    start_time = time.time()
    timeout = 30.0  # 30 seconds timeout
    
    while waiting and time.time() - start_time < timeout:
        # Update button status text
        status = []
        if button_status['x_pressed']: status.append("✓ X button")
        if button_status['dpad_up']: status.append("✓ D-pad UP")
        if button_status['dpad_down']: status.append("✓ D-pad DOWN")
        if button_status['dpad_left']: status.append("✓ D-pad LEFT")
        if button_status['dpad_right']: status.append("✓ D-pad RIGHT")
        
        feedback_text.text = "\n".join(status)
        
        # Draw screen
        test_text.draw()
        feedback_text.draw()
        window.flip()
        
        # Check keyboard escape
        if event.getKeys(['escape']):
            logging.info("User pressed escape during controller test")
            return False
        
        # Allow continuing with space if controller doesn't work
        if event.getKeys(['space']):
            logging.info("User pressed space to skip controller test")
            return True
        
        # Process pygame events to update controller state
        ensure_controller_processing()
        
        # Check X button directly
        if controller.getButton(BUTTON_X):
            button_status['x_pressed'] = True
            time.sleep(0.2)  # Debounce
        
        # Check D-pad directions
        if controller.getNumHats() > 0:
            hat_value = controller.getHat(0)
            if hat_value == DPAD_UP:
                button_status['dpad_up'] = True
            elif hat_value == DPAD_DOWN:
                button_status['dpad_down'] = True
            elif hat_value == DPAD_LEFT:
                button_status['dpad_left'] = True
            elif hat_value == DPAD_RIGHT:
                button_status['dpad_right'] = True
        
        # Check A button to finish
        if controller.getButton(BUTTON_A):
            button_status['a_pressed'] = True
            waiting = False
            time.sleep(0.2)
        
        time.sleep(0.01)  # Prevent CPU hogging
    
    # Check if all required buttons were detected
    all_buttons = all([
        button_status['x_pressed'],
        button_status['dpad_up'], 
        button_status['dpad_down'], 
        button_status['dpad_left'], 
        button_status['dpad_right']
    ])
    
    # If we got a timeout, show message and return false
    if time.time() - start_time >= timeout:
        timeout_text = visual.TextStim(
            window,
            text="Controller test timed out. Falling back to keyboard controls.",
            height=0.05,
            color="red"
        )
        timeout_text.draw()
        window.flip()
        core.wait(2.0)
        logging.warning("Controller test timed out")
        return False
    
    # Show result
    result_text = visual.TextStim(
        window,
        text="Controller working correctly!" if all_buttons else "Some controller buttons not detected. Continuing anyway.",
        height=0.05,
        color="green" if all_buttons else "yellow"
    )
    result_text.draw()
    window.flip()
    core.wait(1.5)
    
    return True


def check_emergency_exit_combo(eeg=None, window=None, loggers=None):
    """Check if emergency exit button combination is pressed
    
    Args:
        eeg: EEG marker system for clean shutdown
        window: PsychoPy window for clean shutdown
        loggers: Logging system for recording the exit
        
    Returns:
        bool: True if emergency exit triggered, False otherwise
    """
    if not controller_available:
        return False
    
    try:
        # Make sure controller events are processed
        ensure_controller_processing()
        
        # Check for emergency exit combination (Back/Select + B buttons)
        if controller.getButton(BUTTON_BACK) and controller.getButton(BUTTON_B):
            # Small delay to ensure both buttons were intentionally pressed together
            time.sleep(0.3)
            
            # Double check that both are still pressed
            if controller.getButton(BUTTON_BACK) and controller.getButton(BUTTON_B):
                from modules.utils import emergency_escape
                emergency_escape("Controller emergency exit (Back + B buttons)", window, eeg, loggers)
                return True
        
        # Also check for Start + B combination
        if controller.getButton(BUTTON_START) and controller.getButton(BUTTON_B):
            # Small delay to ensure both buttons were intentionally pressed together
            time.sleep(0.3)
            
            # Double check that both are still pressed
            if controller.getButton(BUTTON_START) and controller.getButton(BUTTON_B):
                from modules.utils import emergency_escape
                emergency_escape("Controller emergency exit (Start + B buttons)", window, eeg, loggers)
                return True
    except Exception as e:
        logging.error(f"Error checking controller emergency exit: {e}")
            
    return False


def update_controller_globals():
    """Update controller globals after initialization
    
    Call this function if controller state seems stale to refresh the connection
    
    Returns:
        bool: Current controller availability
    """
    global controller_available
    
    if not controller_available:
        return False
        
    try:
        # Test if controller is still responsive
        if controller is not None:
            # Try to get basic information - will fail if controller disconnected
            controller.getNumButtons()
            return True
    except Exception as e:
        logging.warning(f"Controller connection issue detected: {e}")
        controller_available = False
    
    return controller_available


def check_controller_health():
    """Check if the controller is still working properly
    
    Returns:
        bool: True if controller is healthy, False otherwise
    """
    if not controller_available:
        return False
    
    try:
        # Process pygame events to update controller state
        ensure_controller_processing()
        
        # Test if controller is still responsive
        if controller is not None:
            # Try to get basic information - will fail if controller disconnected
            num_buttons = controller.getNumButtons()
            num_axes = controller.getNumAxes()
            return True
    except Exception as e:
        logging.error(f"Controller health check failed: {e}")
        return False

def reset_controller(window=None):
    """Attempt to reset the controller connection
    
    Args:
        window: PsychoPy window to display status (optional)
        
    Returns:
        bool: True if reset succeeded, False otherwise
    """
    global controller, controller_available
    
    try:
        # Close existing controller if possible
        if controller is not None:
            try:
                # No explicit close method in PsychoPy joystick, rely on garbage collection
                controller = None
            except:
                pass
            
        # Show reset message if window provided
        if window is not None:
            msg = visual.TextStim(window, text="Controller disconnected.\nAttempting to reconnect...", 
                                 color='yellow', height=0.05)
            msg.draw()
            window.flip()
            
        # Re-initialize pygame joystick subsystem
        import pygame
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
        except:
            pass
            
        # Re-initialize the controller
        controller_available = initialize_controller()
        
        # Show success/failure message
        if window is not None:
            if controller_available:
                msg = visual.TextStim(window, text="Controller reconnected successfully!", 
                                    color='green', height=0.05)
            else:
                msg = visual.TextStim(window, text="Failed to reconnect controller.\nFalling back to keyboard.", 
                                    color='red', height=0.05)
            msg.draw()
            window.flip()
            core.wait(1.5)
            
        return controller_available
        
    except Exception as e:
        logging.error(f"Controller reset failed: {e}")
        controller_available = False
        return False

def ensure_controller_processing():
    """Ensure that controller events are being processed
    
    This is important for pygame backend to work properly
    """
    try:
        # Process pygame events to keep controller state updated
        import pygame
        
        # Check if pygame is initialized
        if not pygame.get_init():
            # Initialize pygame if it's not already initialized
            pygame.init()
            logging.info("Pygame initialized for controller processing")
        
        # Now pump events safely
        pygame.event.pump()
    except Exception as e:
        # Lower the log level to debug to avoid filling logs with these errors
        logging.debug(f"Failed to process controller events: {e}")

def setup_controller_monitoring(window):
    """Set up background monitoring for controller health
    
    This function starts a background thread that periodically checks the
    controller connection and attempts to reset it if needed.
    
    Args:
        window: PsychoPy window for status display
    """
    import threading
    
    def monitor_controller():
        last_check = time.time()
        check_interval = 5.0  # Check every 5 seconds
        
        while True:
            current_time = time.time()
            
            # Only check at specified intervals
            if current_time - last_check >= check_interval:
                # Check controller health
                if controller_available and not check_controller_health():
                    logging.warning("Controller health check failed, attempting reset...")
                    reset_controller(window)
                
                # Also try to initialize controller if not available
                elif not controller_available and current_time % 30 < 5:  # Try every 30 seconds
                    logging.info("Attempting to detect controller...")
                    reset_controller(window)
                
                last_check = current_time
            
            # Sleep to reduce CPU usage
            time.sleep(0.5)
    
    # Create and start the monitoring thread as a daemon
    monitor_thread = threading.Thread(target=monitor_controller, daemon=True)
    monitor_thread.start()
    logging.info("Controller monitoring thread started")
    
    return monitor_thread