#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input Diagnostic Tool
====================

This script helps diagnose potential conflicts between keyboard and controller inputs
by displaying which input method is being detected in real-time.
"""

import time
import logging
from psychopy import visual, event, core
from psychopy.hardware import joystick

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_input_diagnostic():
    """Run a diagnostic test to check for input conflicts"""
    
    print("=== Input Conflict Diagnostic Tool ===")
    print("\nThis tool will help diagnose if keyboard inputs are overriding controller inputs.")
    
    # Create window
    win = visual.Window(
        size=(800, 600),
        fullscr=False,
        screen=0,
        winType='pyglet',
        allowGUI=True,
        monitor='testMonitor',
        color='black',
        colorSpace='rgb',
        units='norm'
    )
    
    # Create text elements
    title = visual.TextStim(
        win=win,
        text="INPUT DIAGNOSTIC TOOL",
        pos=(0, 0.8),
        height=0.08,
        color='white'
    )
    
    instructions = visual.TextStim(
        win=win,
        text="Press buttons on your controller or keyboard.\nWatch which inputs are detected.\nPress ESC to exit.",
        pos=(0, 0.6),
        height=0.05,
        wrapWidth=1.8,
        color='white'
    )
    
    controller_text = visual.TextStim(
        win=win,
        text="No controller input detected",
        pos=(0, 0.2),
        height=0.06,
        color='yellow'
    )
    
    keyboard_text = visual.TextStim(
        win=win,
        text="No keyboard input detected",
        pos=(0, -0.2),
        height=0.06,
        color='green'
    )
    
    status_text = visual.TextStim(
        win=win,
        text="",
        pos=(0, -0.6),
        height=0.05,
        color='red'
    )
    
    # Initialize controller if available
    try:
        # Try pygame first
        joystick.backend = 'pygame'
        pygame_controllers = joystick.getNumJoysticks()
        
        if pygame_controllers > 0:
            joy = joystick.Joystick(0)
            controller_name = joy.getName()
            controller_available = True
            controller_backend = 'pygame'
            
            # Import pygame for event processing
            import pygame
            
            def process_controller_events():
                pygame.event.pump()
        else:
            # Try pyglet
            joystick.backend = 'pyglet'
            pyglet_controllers = joystick.getNumJoysticks()
            
            if pyglet_controllers > 0:
                joy = joystick.Joystick(0)
                controller_name = joy.getName()
                controller_available = True
                controller_backend = 'pyglet'
                
                # Import pyglet for event processing
                import pyglet
                
                def process_controller_events():
                    pyglet.clock.tick()
            else:
                controller_available = False
                controller_name = "No controller detected"
                controller_backend = None
                
                def process_controller_events():
                    pass
    except Exception as e:
        controller_available = False
        controller_name = f"Controller error: {e}"
        controller_backend = None
        
        def process_controller_events():
            pass
    
    # Show controller status
    status_text.setText(f"Controller: {controller_name}\nBackend: {controller_backend}")
    
    # Variables to track input
    last_controller_time = 0
    last_keyboard_time = 0
    controller_count = 0
    keyboard_count = 0
    should_exit = False
    
    # Main test loop
    while not should_exit:
        # Process controller events
        process_controller_events()
        
        # Check controller input if available
        controller_input_detected = False
        if controller_available:
            # Check buttons
            for i in range(joy.getNumButtons()):
                if joy.getButton(i):
                    controller_input_detected = True
                    controller_count += 1
                    last_controller_time = time.time()
                    controller_text.setText(f"CONTROLLER INPUT: Button {i} pressed\nTotal: {controller_count}")
                    break
            
            # Check D-pad if no button pressed
            if not controller_input_detected and joy.getNumHats() > 0:
                hat = joy.getHat(0)
                if hat != (0, 0):
                    controller_input_detected = True
                    controller_count += 1
                    last_controller_time = time.time()
                    controller_text.setText(f"CONTROLLER INPUT: D-pad {hat}\nTotal: {controller_count}")
        
        # Fade controller text over time
        if not controller_input_detected and time.time() - last_controller_time > 1.0:
            controller_text.setText(f"No recent controller input\nTotal detected: {controller_count}")
        
        # Check keyboard input
        keys = event.getKeys()
        if keys:
            if 'escape' in keys:
                should_exit = True
            else:
                keyboard_count += 1
                last_keyboard_time = time.time()
                keyboard_text.setText(f"KEYBOARD INPUT: {', '.join(keys)}\nTotal: {keyboard_count}")
        
        # Fade keyboard text over time
        if time.time() - last_keyboard_time > 1.0:
            keyboard_text.setText(f"No recent keyboard input\nTotal detected: {keyboard_count}")
        
        # Draw everything
        title.draw()
        instructions.draw()
        controller_text.draw()
        keyboard_text.draw()
        status_text.draw()
        win.flip()
        
        # Brief pause
        core.wait(0.01)
    
    # Clean up
    win.close()
    print("\nInput diagnostic test completed")
    print(f"Controller inputs detected: {controller_count}")
    print(f"Keyboard inputs detected: {keyboard_count}")

if __name__ == "__main__":
    run_input_diagnostic()
    core.quit()
