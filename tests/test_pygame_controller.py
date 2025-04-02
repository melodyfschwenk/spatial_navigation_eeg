#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pygame Controller Diagnostic Tool
================================

This script tests the controller connection using the pygame backend
and provides detailed feedback about button presses and mappings.
"""

import os
import sys
import time
import logging
from psychopy import visual, core, event
from psychopy.hardware import joystick
import pygame  # Import pygame directly

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_pygame_controller():
    """Test controller connection using pygame backend"""
    
    print("=== Pygame Controller Diagnostic Tool ===")
    print("Checking for pygame installation...")
    
    try:
        pygame_version = pygame.version.ver
        print(f"✓ Pygame detected (version {pygame_version})")
    except Exception as e:
        print(f"✗ Error with pygame: {e}")
        print("  Try installing pygame: pip install pygame")
        return
    
    print("\nInitializing joystick module...")
    try:
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Set pygame backend for PsychoPy
        joystick.backend = 'pygame'
        print(f"✓ Using {joystick.backend} backend")
    except Exception as e:
        print(f"✗ Error setting joystick backend: {e}")
        return
    
    # Create window
    print("\nCreating test window...")
    try:
        win = visual.Window(
            size=(800, 600),
            fullscr=False,
            screen=0,
            winType='pyglet',  # Keep window as pyglet
            allowGUI=True,
            allowStencil=False,
            monitor='testMonitor',
            color='black',
            colorSpace='rgb',
            units='norm'
        )
        print("✓ Test window created successfully")
    except Exception as e:
        print(f"✗ Error creating window: {e}")
        return
    
    print("\nSearching for controllers...")
    try:
        # Process pygame events before checking for joysticks
        pygame.event.pump()
        
        # Get number of joysticks
        n_joysticks = joystick.getNumJoysticks()
        print(f"Found {n_joysticks} controller(s)")
        
        if n_joysticks == 0:
            print("\n✗ No controllers detected!")
            print("  • Make sure your controller is connected")
            print("  • Try unplugging and reconnecting the controller")
            print("  • On some systems, you may need to install additional drivers")
            win.close()
            return
            
        # Get first joystick
        print("\nAttempting to initialize controller 0...")
        joy = joystick.Joystick(0)
        
        # Get controller details
        name = joy.getName()
        num_buttons = joy.getNumButtons()
        num_axes = joy.getNumAxes()
        num_hats = joy.getNumHats()
        
        print(f"\n✓ Controller initialized successfully!")
        print(f"  • Name: {name}")
        print(f"  • Buttons: {num_buttons}")
        print(f"  • Axes: {num_axes}")
        print(f"  • Hats: {num_hats}")
        
        # Create text stimuli
        instructions = visual.TextStim(
            win=win,
            text="Press buttons on your controller to test\n\n"
                 "The button/hat states will be displayed below\n\n"
                 "Press ESC key to exit",
            pos=(0, 0.7),
            height=0.07,
            wrapWidth=1.8
        )
        
        button_text = visual.TextStim(
            win=win, 
            text="", 
            pos=(0, 0), 
            height=0.05,
            color='yellow'
        )
        
        axes_text = visual.TextStim(
            win=win,
            text="",
            pos=(0, -0.7),
            height=0.05,
            color='cyan'
        )
        
        print("\nStarting controller test loop...")
        print("Press buttons to see their states. Press ESC to exit.")
        
        # Main test loop
        should_exit = False
        while not should_exit:
            # Process pygame events
            pygame.event.pump()
            
            # Get controller states
            btn_states = joy.getAllButtons()
            hat_states = joy.getAllHats() if num_hats > 0 else []
            
            # Format button states
            btn_status = "Buttons: "
            for i, pressed in enumerate(btn_states):
                if pressed:
                    btn_status += f"{i} "
            
            # Format hat states
            hat_status = "D-pad/Hats: "
            for i, hat in enumerate(hat_states):
                if hat != (0, 0):
                    hat_status += f"Hat {i}: {hat} "
            
            # Get axis values (first 6 only)
            axis_status = "Axis values:\n"
            for i in range(min(6, num_axes)):
                try:
                    val = joy.getAxis(i)
                    axis_status += f"Axis {i}: {val:.2f}\n"
                except:
                    axis_status += f"Axis {i}: ERROR\n"
            
            # Update display
            button_text.setText(f"{btn_status}\n\n{hat_status}")
            axes_text.setText(axis_status)
            
            # Draw everything
            instructions.draw()
            button_text.draw()
            axes_text.draw()
            win.flip()
            
            # Check for escape key using PsychoPy's event module
            keys = event.getKeys()
            if 'escape' in keys:
                should_exit = True
            
            # Brief pause
            core.wait(0.01)
        
        # Clean up
        win.close()
        print("\nController test completed")
        
    except Exception as e:
        print(f"\n✗ Error during controller test: {e}")
        import traceback
        traceback.print_exc()
        if 'win' in locals() and win is not None:
            win.close()

if __name__ == "__main__":
    test_pygame_controller()
    print("\nExiting pygame controller diagnostic tool...")
    pygame.quit()
    core.quit()
