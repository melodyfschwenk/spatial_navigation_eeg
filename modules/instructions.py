#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Instructions Module for Spatial Navigation EEG Experiment
========================================================

This module handles instruction text and presentation.
"""

import logging
from psychopy import core, event, visual

from modules.ui import show_text_screen


def get_instructions(navigation_type, difficulty='all', brief=False):
    """Return instructions based on the navigation type and difficulty
    
    Args:
        navigation_type: 'egocentric', 'allocentric', or 'control'
        difficulty: 'easy', 'hard', 'control', or 'all' for base navigation instructions
        brief: If True, return brief instructions for the main experiment
        
    Returns:
        str: The instruction text
    """
    # Brief instructions for main experiment blocks
    if brief:
        brief_text = {
            'egocentric': """
PLAYER VIEW:
UP = forward (direction you're facing)
DOWN = backward
LEFT = to your left
RIGHT = to your right
""",
            'allocentric': """
MAP VIEW:
UP = toward the top of the screen
DOWN = toward the bottom of the screen
LEFT = toward the left of the screen
RIGHT = toward the right of the screen
""",
            'control': """
ARROW FOLLOWING:
Simply press the arrow key that matches the direction of the first arrow.
UP = first arrow points up
DOWN = first arrow points down
LEFT = first arrow points left
RIGHT = first arrow points right
"""
        }
        
        return brief_text.get(navigation_type, "")
    
    # Detailed instructions for practice and first exposure - without headlines
    instruction_text = {
        'egocentric': """
In this task, you will move from the gray player to the red stop sign while avoiding blue walls.
The gray triangle shows which way the player is facing.
Your job is to choose the first step the player should take. Make your choice as if you are the player looking in the direction of the triangle.
Use these keys:
UP arrow: Move forward (in the direction the player is facing)
DOWN arrow: Move backward
LEFT arrow: Move to the player's left
RIGHT arrow: Move to the player's right
Example: UP moves you forward in whatever direction you're facing.
Choose the first step needed to reach the stop sign. Try to respond quickly and correctly.
""",
        'allocentric': """
In this task, you will move from the gray player to the red stop sign while avoiding blue walls.
Your job is to choose the first step the player should take. Make your choice based on screen directions (like using a map).
Use these keys:
UP arrow: Move toward the top of the screen
DOWN arrow: Move toward the bottom of the screen
LEFT arrow: Move toward the left side of the screen
RIGHT arrow: Move toward the right side of the screen
No matter which way the player is facing, pressing UP always moves toward the top of the screen.
Choose the first step needed to reach the target. Try to respond quickly and correctly.
""",
        'control': """
In this task, you will see arrows showing the path from the player to the target.
Your job is to follow the first arrow from the player's position.
Use these keys:
UP arrow: When the first arrow points up
DOWN arrow: When the first arrow points down
LEFT arrow: When the first arrow points left
RIGHT arrow: When the first arrow points right
Example: Press the RIGHT arrow key if the first arrow points right.
Try to respond quickly and correctly.
"""
    }
    
    # Return the appropriate instructions
    return instruction_text.get(navigation_type, "Instructions not available.")

def show_instructions(window, navigation_type, difficulty, config, block_num=None):
    """Display instructions for the given navigation type and difficulty
    
    Args:
        window: PsychoPy window object
        navigation_type: 'egocentric' or 'allocentric'
        difficulty: 'easy', 'hard', or 'control'
        config: The experiment configuration object
        block_num: If provided and > 0, show brief instructions for main experiment
        
    Returns:
        str: Key pressed to exit the instructions
    """
    # Import the consolidated input handler
    from modules.utils import get_input, emergency_escape
    import logging
    from psychopy import core, visual, event
    
    # Determine if we should show brief or full instructions
    use_brief = block_num is not None and block_num > 0
    
    # Get instructions for this condition (without headline)
    instruction_text = get_instructions(navigation_type, difficulty, brief=use_brief)
    
    # Define headline based on navigation type (only for visual display)
    headline = ""
    if navigation_type == 'egocentric':
        headline = "PLAYER VIEW"
    elif navigation_type == 'allocentric':
        headline = "MAP VIEW"
    elif navigation_type == 'control':
        headline = "ARROW FOLLOWING"
    
    # Add controller instructions ONLY if not already included in show_text_screen
    continue_message = "\n\nREAD CAREFULLY, THEN press SPACE to continue."
    
    # Create colored rectangle background for the headline (to make condition more obvious)
    if navigation_type == 'egocentric':
        rect_color = (0.2, 0.6, 0.2)  # Green for egocentric
    elif navigation_type == 'allocentric':
        rect_color = (0.6, 0.2, 0.6)  # Purple for allocentric
    else:
        rect_color = (0.6, 0.6, 0.2)  # Yellow for control
    
    # Create background rectangle for the headline
    headline_rect = visual.Rect(
        window, 
        width=1.8, 
        height=0.15,
        fillColor=rect_color,
        pos=(0, 0.8)
    )
    
    # Create instruction text stimulus (without headline duplication)
    instructions = visual.TextStim(
        window,
        text=instruction_text + continue_message,
        color=(-0.7, -0.7, -0.7),
        height=0.05,
        wrapWidth=1.8
    )
    
    # Create headline text to overlay on the rectangle
    headline_text = visual.TextStim(
        window,
        text=headline,
        color=(1, 1, 1),
        height=0.06,
        pos=(0, 0.8),
        bold=True
    )
    
    # Create a "take your time" message
    time_msg = visual.TextStim(
        window,
        text="Take your time to read the instructions",
        color=(-0.7, -0.7, -0.7),
        height=0.04,
        pos=(0, -0.85)
    )
    
    # IMPORTANT: Clear any pending events before showing instructions
    event.clearEvents()
    
    # Display instructions with the colored headline
    headline_rect.draw()
    headline_text.draw()
    instructions.draw()
    time_msg.draw()
    window.flip()
    
    # Log the instruction display
    logging.info(f"Displayed {'brief' if use_brief else 'full'} instructions for {navigation_type} navigation, {difficulty} difficulty")
    
    # Wait for response using a cross-platform compatible approach
    try:
        # First try our consolidated input handler
        response = get_input(config, allowed_keys=['space', 'escape'], clear_events=True)
        
        # If that fails, fall back to direct event waiting
        if response is None:
            keys = event.waitKeys(keyList=['space', 'escape'])
            response = keys[0] if keys else 'space'  # Default to space
    except Exception as e:
        logging.error(f"Error getting input: {e}")
        # Emergency fallback
        event.clearEvents()
        core.wait(0.5)
        keys = event.waitKeys(keyList=['space', 'escape'])
        response = keys[0] if keys else 'space'  # Default to space
    
    # Force return if we get no response (prevents hanging)
    if response is None:
        response = 'space'  # Default to continue if something went wrong
    
    if response == 'escape':
        logging.info("User pressed escape during instructions, quitting")
        emergency_escape("User pressed escape during instructions", window)
    
    # Display a 5-second countdown before starting trials
    for i in range(5, 0, -1):
        # Use participant-friendly labels for navigation type
        display_nav_type = "PLAYER VIEW" if navigation_type == 'egocentric' else "MAP VIEW" if navigation_type == 'allocentric' else "ARROW FOLLOWING"
        
        ready_text = visual.TextStim(
            window,
            text=f"Starting in {i}...\n\nRemember: This is a {display_nav_type} task",
            color=(-0.7, -0.7, -0.7),
            height=0.07,
            wrapWidth=1.8
        )
        
        # Draw the colored background again
        headline_rect.draw()
        headline_text.draw()
        ready_text.draw()
        window.flip()
        core.wait(1.0)  # Wait for 1 second
        
        # IMPORTANT: Check for escape during countdown with better error handling
        keys = event.getKeys(['escape'])
        if 'escape' in keys:
            emergency_escape("User pressed escape during countdown", window)
    
    # FIXED: Always return a value to prevent None return
    return response or 'space'