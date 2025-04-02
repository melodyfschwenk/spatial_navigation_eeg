#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Controller Test Script for Spatial Navigation EEG Experiment
===========================================================

This script tests the Logitech F310 gamepad controller and its integration
with PsychoPy. Run this script to verify controller detection and input.
"""

import os
import sys
import time
import logging

# Ensure modules directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Set up basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from psychopy import visual, core, event
from psychopy.hardware import joystick

# Define global variables
controller = None
controller_available = False

# Define button mappings for Logitech F310 controller
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

# D-pad mapping for hat values
DPAD_UP = (0, 1)
DPAD_DOWN = (0, -1)
DPAD_LEFT = (-1, 0)
DPAD_RIGHT = (1, 0)
DPAD_CENTER = (0, 0)

# Create a mapping of hat values to directions
hat_mapping = {
    DPAD_UP: "UP",
    DPAD_DOWN: "DOWN",
    DPAD_LEFT: "LEFT",
    DPAD_RIGHT: "RIGHT",
    DPAD_CENTER: "CENTER"
}

# Button name mappings for display
button_display_names = {
    BUTTON_A: "A (Bottom)",
    BUTTON_B: "B (Right)",
    BUTTON_X: "X (Left)",
    BUTTON_Y: "Y (Top)",
    BUTTON_LEFT_SHOULDER: "Left Shoulder",
    BUTTON_RIGHT_SHOULDER: "Right Shoulder",
    BUTTON_BACK: "Back/Select",
    BUTTON_START: "Start",
    BUTTON_LEFT_STICK: "Left Stick Press",
    BUTTON_RIGHT_STICK: "Right Stick Press"
}


def initialize_controller():
    """Initialize gamepad/controller using pygame backend"""
    global controller, controller_available
    
    try:
        logging.info("Initializing controller with pygame backend")
        
        # Set pygame backend for joystick
        joystick.backend = 'pygame'
        
        # Get number of available controllers
        n_controllers = joystick.getNumJoysticks()
        
        if n_controllers > 0:
            logging.info(f"Found {n_controllers} controllers")
            
            # List all controllers
            for i in range(n_controllers):
                controller = joystick.Joystick(i)
                name = controller.getName()
                buttons = controller.getNumButtons()
                axes = controller.getNumAxes()
                hats = controller.getNumHats()
                logging.info(f"Controller {i}: {name} - Buttons: {buttons}, Axes: {axes}, Hats: {hats}")
            
            # Initialize the first controller
            controller = joystick.Joystick(0)
            controller_name = controller.getName()
            
            # Process pygame events immediately
            ensure_controller_processing()
            
            logging.info(f"Selected controller: {controller_name}")
            controller_available = True
            return True
        else:
            logging.warning("No gamepad/controller detected.")
            controller_available = False
            return False
    
    except Exception as e:
        logging.error(f"Failed to initialize controller: {e}")
        import traceback
        traceback.print_exc()
        controller_available = False
        return False


def ensure_controller_processing():
    """Ensure that controller events are being processed"""
    try:
        # Process pygame events to keep controller state updated
        import pygame
        pygame.event.pump()
    except Exception as e:
        logging.error(f"Failed to process controller events: {e}")


def run_controller_test():
    """Run interactive test of the controller"""
    # Create window
    win = visual.Window(
        size=(800, 600),
        fullscr=False,
        screen=0,
        winType='pyglet',
        allowGUI=True,
        color=(0, 0, 0),
        units='norm'
    )
    
    # Create text for instructions
    instructions = visual.TextStim(
        win,
        text="Controller Test Mode\n\n"
             "Press buttons on the controller to see input\n"
             "D-pad and buttons will be displayed here\n\n"
             "Press ESCAPE key to exit",
        pos=(0, 0.8),
        color='white',
        height=0.06,
        wrapWidth=1.8
    )
    
    # Create text for button display
    button_text = visual.TextStim(
        win,
        text="Waiting for input...",
        pos=(0, 0),
        color='white',
        height=0.05,
        wrapWidth=1.8
    )
    
    # Create text for controller status
    status_text = visual.TextStim(
        win,
        text=f"Controller: {'CONNECTED' if controller_available else 'NOT CONNECTED'}",
        pos=(0, -0.8),
        color='green' if controller_available else 'red',
        height=0.05
    )
    
    # Run the test loop
    try:
        last_input = "No input yet"
        running = True
        
        while running:
            # Check keyboard for escape
            keys = event.getKeys()
            if 'escape' in keys:
                running = False
                break
            
            # Process controller events
            ensure_controller_processing()
            
            # Check controller input
            if controller_available:
                # Check buttons
                buttons_pressed = []
                for i in range(min(15, controller.getNumButtons())):
                    if controller.getButton(i):
                        display_name = button_display_names.get(i, f"Button {i}")
                        buttons_pressed.append(display_name)
                
                # Check D-pad (hat)
                hat_value = None
                dpad_name = "None"
                if controller.getNumHats() > 0:
                    hat_value = controller.getHat(0)
                    dpad_name = hat_mapping.get(hat_value, str(hat_value))
                
                # Check axes
                axes_values = []
                for i in range(min(8, controller.getNumAxes())):
                    value = controller.getAxis(i)
                    if abs(value) > 0.1:  # Only show significant axis movement
                        axes_values.append(f"Axis {i}: {value:.2f}")
                
                # Format input display
                input_lines = []
                if buttons_pressed:
                    input_lines.append(f"Buttons: {', '.join(buttons_pressed)}")
                
                if hat_value is not None and hat_value != DPAD_CENTER:
                    input_lines.append(f"D-pad: {dpad_name}")
                
                if axes_values:
                    input_lines.append("Axes: " + ", ".join(axes_values))
                
                if input_lines:
                    last_input = "\n".join(input_lines)
                
                # Update status text (in case controller was disconnected)
                if not controller.getNumButtons() > 0:
                    status_text.text = "Controller: DISCONNECTED"
                    status_text.color = 'red'
                    last_input = "Controller disconnected"
            else:
                status_text.text = "Controller: NOT CONNECTED"
                status_text.color = 'red'
                
                # Try to initialize again periodically
                if int(time.time()) % 5 == 0:  # Every 5 seconds
                    if initialize_controller():
                        status_text.text = "Controller: CONNECTED"
                        status_text.color = 'green'
                        last_input = "Controller connected"
            
            # Update display
            button_text.text = last_input
            
            # Draw everything
            instructions.draw()
            button_text.draw()
            status_text.draw()
            win.flip()
            
            # Pause briefly
            time.sleep(0.01)
        
    finally:
        # Clean up
        win.close()


if __name__ == "__main__":
    print("Starting Controller Test Script")
    print("==============================")
    print("This script will test your Logitech F310 controller with PsychoPy")
    print("Press buttons and move the D-pad to see input detection")
    print("Press ESCAPE to exit")
    print("==============================")
    
    # Try to initialize controller
    if initialize_controller():
        print(f"Controller detected: {controller.getName()}")
    else:
        print("No controller detected - will keep trying during the test")
    
    # Run interactive test
    run_controller_test()
    
    print("Controller test completed")