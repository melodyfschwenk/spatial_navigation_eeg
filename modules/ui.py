#!C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe
# -*- coding: utf-8 -*-

"""
UI Module for Spatial Navigation EEG Experiment
===============================================

This module handles the user interface components of the experiment.
Updated with white background and improved UI elements for EEG recording.
"""

import logging
import os
from psychopy import visual, core, event


def get_participant_info():
    """Display dialog to collect participant information"""
    from psychopy import gui
    
    exp_info = {
        'participant_id': '',
        'age': '',
        'gender': ['Male', 'Female', 'Non-binary', 'Prefer not to say'],
        'handedness': ['Right', 'Left', 'Ambidextrous'],
        'session': 1,
        'counterbalance': ['Auto', 1, 2, 3, 4],
        'experimenter_initials': ''
    }
    
    # Add descriptions for counterbalance conditions
    counterbalance_info = {
        'CB Description': "'Auto' = Determined from participant ID, or 1=ego_easy→ego_hard→allo_easy→allo_hard, 2=ego_hard→allo_hard→ego_easy→allo_easy, 3=allo_easy→ego_easy→allo_hard→ego_hard, 4=allo_hard→allo_easy→ego_hard→ego_easy"
    }
    
    # Create a dictionary with all fields
    combined_info = {**exp_info, **counterbalance_info}
    
    dlg = gui.DlgFromDict(dictionary=combined_info, title='Spatial Navigation EEG Experiment', 
                          fixed=['CB Description'])  # Make the description non-editable
    
    if dlg.OK:
        # Remove the description key before returning
        if 'CB Description' in combined_info:
            del combined_info['CB Description']
        
        logging.info(f"Participant info collected: ID={combined_info['participant_id']}, Session={combined_info['session']}, CB={combined_info['counterbalance']}")
        return combined_info
    else:
        logging.info("Experiment cancelled by user during participant info collection")
        core.quit()


def create_experiment_window(config):
    """Create the main experiment window on the chosen monitor.
    
    The window will appear on the monitor specified by config.monitor_index 
    (e.g., 0 for the primary monitor, 1 for a secondary monitor). 
    Make sure this attribute is set in your configuration.
    """
    try:
        window = visual.Window(
            size=config.screen_resolution,
            fullscr=config.fullscreen,
            screen=config.monitor_index,  # using monitor index from config (0, 1, etc.)
            color=config.background_color,
            units='norm'
        )
        logging.info("PsychoPy window created successfully")
        return window
    except Exception as e:
        logging.error(f"Failed to create PsychoPy window: {e}")
        raise


def show_text_screen(window, text, text_color=(-0.7, -0.7, -0.7), text_height=0.07, wait_for_key=True, 
                    key_list=None, max_wait=None):
    """Display a text screen and wait for a key press or timeout"""
    from modules.utils import get_input, emergency_escape
    from psychopy import visual, core
    import logging
    
    if key_list is None:
        key_list = ['space', 'escape']
    
    # Check if controller is being used
    try:
        using_controller = hasattr(window, 'exp_handler') and hasattr(window.exp_handler, 'config') and hasattr(window.exp_handler.config, 'use_controller') and window.exp_handler.config.use_controller
        config = window.exp_handler.config if using_controller else None
    except:
        using_controller = False
        config = None
    
    # Add controller instruction if using controller and instructions not already present
    if using_controller and wait_for_key:
        try:
            from modules.controller import controller_available
            if controller_available and "button" not in text.lower() and "d-pad" not in text.lower():
                # Separate the text with a line break if it doesn't already end with one
                if not text.endswith('\n\n'):
                    if not text.endswith('\n'):
                        text += '\n\n'
                    else:
                        text += '\n'
                
                # Add controller-specific instructions only if not already present
                if 'space' in key_list:
                    text += "Press X button to continue"
                elif 'up' in key_list or 'down' in key_list or 'left' in key_list or 'right' in key_list:
                    text += "Use D-pad for navigation"
                else:
                    # Generic controller instruction
                    text += "Use controller for response"
        except:
            pass
    
    # Create text stimulus
    text_stim = visual.TextStim(
        window,
        text=text,
        color=text_color,
        height=text_height,
        wrapWidth=1.8
    )
    
    # Draw text to screen
    text_stim.draw()
    window.flip()
    
    # Log what's being displayed
    logging.info(f"Showing text screen: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    if wait_for_key:
        # If we have a config object, use our consolidated input handler
        if config:
            response = get_input(config, max_wait=max_wait, allowed_keys=key_list)
        else:
            # Create a minimal config object
            class MinimalConfig:
                use_controller = False
            response = get_input(MinimalConfig(), max_wait=max_wait, allowed_keys=key_list)
            
        if response == 'escape':
            # Handle escape
            logging.info("User pressed escape during text screen")
            emergency_escape("User pressed escape", window)
            return None
        return response
    else:
        # Just wait for the specified time without waiting for key
        if max_wait is not None:
            core.wait(max_wait)
        return None


def show_welcome_screen(window, config, total_trials, total_blocks):
    """Show welcome screen with experiment information"""
    from modules.controller import controller_available
    
    # Add controller-specific text if using controller
    controller_text = ""
    if hasattr(config, 'use_controller') and config.use_controller and controller_available:
        controller_text = "\n\nYou'll be using the gamepad controller:\n" \
                         "- D-pad: Navigate and respond (UP, DOWN, LEFT, RIGHT)\n" \
                         "- X button: Confirm/Continue (like SPACE key)\n" \
                         "- B button: Exit experiment (emergency only)\n"
    
    welcome_text = (
        "Welcome to the Navigation Experiment\n\n"
        f"You'll complete 9 sections.{controller_text}"
    )
    
    return show_text_screen(
        window, 
        welcome_text,
        text_color=config.text_color
    )


def show_block_end_screen(window, config, block_num, total_blocks, absolute_trial_counter, total_trials):
    """Show end of block screen with improved progress information for EEG study"""
    from modules.controller import controller_available
    
    # Fixed total values
    total_blocks = 9
    total_trials = 135
    progress_percent = round((absolute_trial_counter / total_trials) * 100)
    
    # Calculate blocks remaining
    blocks_remaining = total_blocks - block_num
    trials_remaining = total_trials - absolute_trial_counter
    
    # Create visual progress bar with block symbols
    blocks_completed = "■" * block_num + "□" * blocks_remaining
    
    # Create encouraging message based on progress
    if block_num < total_blocks / 3:
        encouragement = "You're doing great! Keep going."
    elif block_num < 2 * total_blocks / 3:
        encouragement = "Over halfway there! You're making excellent progress."
    else:
        encouragement = "Almost done! Just a few more sections left."
    
    # Adapt text based on input method
    continue_text = "Press SPACE when ready to continue."
    if hasattr(config, 'use_controller') and config.use_controller and controller_available:
        continue_text = "Press X button when ready to continue."
        
    # Create block completion message
    message = f"""
Section {block_num} of {total_blocks} complete!

Progress: {progress_percent}% 
{blocks_completed}

{encouragement}

{continue_text}
"""
    
    return show_text_screen(window, message, text_color=config.text_color)


def show_navigation_transition(window, current_nav_type, next_nav_type, config):
    """Show transition message when switching between navigation types
    
    This function displays an explicit notification when the navigation type 
    changes between blocks to ensure participants are aware of the switch.
    
    Args:
        window: PsychoPy window object
        current_nav_type: The previous navigation type ('egocentric', 'allocentric', or 'control')
        next_nav_type: The upcoming navigation type ('egocentric', 'allocentric', or 'control')
        config: Experiment configuration object
        
    Returns:
        bool: True if transition was shown, False if no transition needed
    """
    # No transition needed if navigation type hasn't changed or if this is the first block
    if current_nav_type == next_nav_type or current_nav_type is None:
        return False
    
    import logging
    from psychopy import visual, core
    from modules.controller import controller_available
    
    # Log the navigation change
    logging.info(f"Navigation transition: {current_nav_type} → {next_nav_type}")
    
    # Create friendly names for display
    friendly_names = {
        'egocentric': "PLAYER VIEW",
        'allocentric': "MAP VIEW",
        'control': "ARROW FOLLOWING"
    }
    
    current_name = friendly_names.get(current_nav_type, current_nav_type)
    next_name = friendly_names.get(next_nav_type, next_nav_type)
    
    # Create colored backgrounds to clearly distinguish navigation types
    if next_nav_type == 'egocentric':
        bg_color = (0.2, 0.6, 0.2)  # Green for egocentric
        bg_text = "PLAYER VIEW"
    elif next_nav_type == 'allocentric':
        bg_color = (0.6, 0.2, 0.6)  # Purple for allocentric
        bg_text = "MAP VIEW" 
    else:
        bg_color = (0.6, 0.6, 0.2)  # Yellow for control
        bg_text = "ARROW FOLLOWING"
    
    # Create visual elements
    header_rect = visual.Rect(
        window,
        width=1.8,
        height=0.15,
        fillColor=bg_color,
        pos=(0, 0.8)
    )
    
    header_text = visual.TextStim(
        window,
        text=bg_text,
        color=(1, 1, 1),  # White text
        height=0.06,
        pos=(0, 0.8),
        bold=True
    )
    
    # Create the main transition message
    transition_msg = f"""
ATTENTION: Navigation Type Change

You are switching from:
{current_name}
to:
{next_name}

The controls will now work differently!
"""
    
    # Add controller-specific text if using controller
    if hasattr(config, 'use_controller') and config.use_controller and controller_available:
        continue_text = "\n\nPress X button to continue"
    else:
        continue_text = "\n\nPress SPACE to continue"
    
    # Create text stimulus
    message = visual.TextStim(
        window,
        text=transition_msg + continue_text,
        color=config.text_color,
        height=0.06,
        wrapWidth=1.8
    )
    
    # Display the transition message
    header_rect.draw()
    header_text.draw()
    message.draw()
    window.flip()
    
    # Wait for key press
    from modules.utils import get_input
    get_input(config, allowed_keys=['space', 'escape'])
    
    # Show a 3 second countdown
    for i in range(3, 0, -1):
        header_rect.draw()
        header_text.draw()
        
        countdown_text = visual.TextStim(
            window,
            text=f"Get ready for {next_name}\n\nStarting in {i}...",
            color=config.text_color,
            height=0.06
        )
        
        countdown_text.draw()
        window.flip()
        core.wait(1.0)
    
    # Return success
    return True


def show_completion_screen(window, config, absolute_trial_counter):
    """Show experiment completion screen"""
    completion_text = (
        f"Thank you for participating!\n\n"
        f"You've completed all {absolute_trial_counter} trials.\n\n"
        "The experiment is now complete."
    )
    
    return show_text_screen(
        window, 
        completion_text, 
        text_color=config.text_color, 
        max_wait=5.0, 
        wait_for_key=False
    )


def display_stimulus(window, stimulus_file):
    """Display a stimulus image"""
    try:
        # Use pixel units for consistent sizing
        stimulus = visual.ImageStim(
            window,
            image=stimulus_file,
            units='pix',
            size=(500, 500)  # Explicitly set size
        )
        
        stimulus.draw()
        window.flip()
        return stimulus
        
    except Exception as e:
        logging.error(f"Error displaying stimulus {stimulus_file}: {e}")
        raise


def display_fixation(window, config):
    """Display fixation cross with dark color for white background"""
    fixation = visual.TextStim(
        window,
        text='+',
        height=0.08,
        color=(-0.7, -0.7, -0.7)  # Dark gray for white background
    )
    fixation.draw()
    window.flip()
    return fixation


def display_feedback(window, accuracy):
    """Display feedback based on response accuracy"""
    if accuracy:
        feedback = visual.TextStim(
            window,
            text="Correct",
            height=0.07,
            color=(0, 0.7, 0)  # Darker green for white background
        )
    else:
        feedback = visual.TextStim(
            window,
            text="Incorrect",
            height=0.07,
            color=(0.7, 0, 0)  # Darker red for white background
        )
    
    feedback.draw()
    window.flip()
    return feedback


def confirm_quit(window):
    """Display confirmation dialog for quitting the experiment"""
    from modules.utils import emergency_escape
    from modules.controller import controller_available
    
    # Check if controller is being used
    using_controller = hasattr(window, 'exp_handler') and hasattr(window.exp_handler, 'config') and hasattr(window.exp_handler.config, 'use_controller') and window.exp_handler.config.use_controller
    controller_text = ""
    if using_controller and controller_available:
        controller_text = "\n\nPress X button for YES or B button for NO."
    
    confirm_text = visual.TextStim(
        window,
        text=f"Are you sure you want to quit?\n\nPress 'y' for YES or 'n' for NO.{controller_text}",
        color=(-0.7, -0.7, -0.7),
        height=0.07,
        wrapWidth=1.8
    )
    confirm_text.draw()
    window.flip()
    
    # Get response (includes controller support)
    from modules.utils import get_response
    if using_controller and controller_available:
        response, _ = get_response(window.exp_handler.config, allowed_keys=['y', 'n', 'escape'])
    else:
        keys = event.waitKeys(keyList=['y', 'n', 'escape'])
        response = keys[0] if keys else None
    
    if response == 'y' or response == 'escape':
        logging.info("User confirmed quit")
        emergency_escape("User confirmed quit via dialog", window)
        return True  # Won't actually reach this due to emergency_escape
    else:
        logging.info("User cancelled quit")
        return False


def show_loading_screen(window, message="Loading experiment..."):
    """Display a loading screen while resources are being prepared"""
    loading_text = visual.TextStim(
        window,
        text=message,
        color=(-0.7, -0.7, -0.7),
        height=0.07
    )
    
    loading_text.draw()
    window.flip()


def create_progress_bar(window, progress, position=(0, -0.8), width=1.6, height=0.05):
    """Create a progress bar to show loading status"""
    # Calculate background and foreground dimensions
    background_width = width
    foreground_width = width * progress
    
    # Calculate positions
    background_pos = position
    foreground_pos = (position[0] - (width - foreground_width) / 2, position[1])
    
    # Create background rectangle
    background = visual.Rect(
        window,
        width=background_width,
        height=height,
        fillColor=(-0.2, -0.2, -0.2),  # Dark gray
        lineColor=None,
        pos=background_pos
    )
    
    # Create foreground rectangle
    foreground = visual.Rect(
        window,
        width=foreground_width,
        height=height,
        fillColor=(0, 0.5, 0.7),  # Blue
        lineColor=None,
        pos=foreground_pos
    )
    
    # Create percentage text
    percentage = visual.TextStim(
        window,
        text=f"{int(progress * 100)}%",
        color=(-0.7, -0.7, -0.7),  # Dark gray
        height=0.04,
        pos=(position[0], position[1])
    )
    
    # Return all components
    return [background, foreground, percentage]


def display_debug_info(window, info_dict, position=(-0.9, 0.9), line_height=0.05):
    """Display debug information during development
    
    Args:
        window: PsychoPy window object
        info_dict: Dictionary of debug information
        position: Starting position for text
        line_height: Vertical spacing between lines
    """
    # Only show in debug mode
    debug_mode = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
    
    if not debug_mode:
        return
    
    lines = []
    y_pos = position[1]
    
    for key, value in info_dict.items():
        text = visual.TextStim(
            window,
            text=f"{key}: {value}",
            color=(-0.5, -0.5, -0.5),  # Dark gray
            height=0.03,
            pos=(position[0], y_pos),
            alignHoriz='left'
        )
        lines.append(text)
        y_pos -= line_height
    
    # Draw all lines
    for line in lines:
        line.draw()


def show_practice_condition_screen(window, config, navigation_type):
    """Display a screen announcing which practice condition is next
    
    Args:
        window: PsychoPy window object
        config: Configuration object
        navigation_type: 'egocentric' or 'allocentric'
    
    Returns:
        response: Key pressed to continue
    """
    from modules.controller import controller_available
    from modules.utils import get_input, emergency_escape
    
    # Set headline and color based on navigation type
    if navigation_type == 'egocentric':
        headline = "PLAYER VIEW PRACTICE"
        rect_color = (0.2, 0.6, 0.2)  # Green for egocentric
        description = (
            "You will now practice navigating from the PLAYER'S perspective.\n\n"
            "- UP: Move forward in the direction the player is facing\n"
            "- DOWN: Move backward\n"
            "- LEFT: Move to the player's left\n"
            "- RIGHT: Move to the player's right"
        )
    else:  # allocentric
        headline = "MAP VIEW PRACTICE"
        rect_color = (0.6, 0.2, 0.6)  # Purple for allocentric
        description = (
            "You will now practice navigating using MAP directions.\n\n"
            "- UP: Move toward the top of the screen\n"
            "- DOWN: Move toward the bottom of the screen\n"
            "- LEFT: Move toward the left side of the screen\n"
            "- RIGHT: Move toward the right side of the screen"
        )
    
    # Create colored rectangle for headline background
    headline_rect = visual.Rect(
        window, 
        width=1.8, 
        height=0.15,
        fillColor=rect_color,
        pos=(0, 0.8)
    )
    
    # Create headline text
    headline_text = visual.TextStim(
        window,
        text=headline,
        color=(1, 1, 1),  # White text on colored background
        height=0.06,
        pos=(0, 0.8),
        bold=True
    )
    
    # Create description text
    description_text = visual.TextStim(
        window,
        text=description,
        color=(-0.7, -0.7, -0.7),
        height=0.05,
        pos=(0, 0.3),
        wrapWidth=1.8
    )
    
    # Create continue instruction
    if hasattr(config, 'use_controller') and config.use_controller and controller_available:
        continue_text = "Press X button to begin practice"
    else:
        continue_text = "Press SPACE to begin practice"
    
    continue_stim = visual.TextStim(
        window,
        text=continue_text,
        color=(-0.7, -0.7, -0.7),
        height=0.05,
        pos=(0, -0.7)
    )
    
    # Draw all elements
    headline_rect.draw()
    headline_text.draw()
    description_text.draw()
    continue_stim.draw()
    window.flip()
    
    # Log the display
    logging.info(f"Showing practice condition screen for {navigation_type}")
    
    # Use the consolidated input handler instead of direct event waiting
    response = get_input(config, allowed_keys=['space', 'escape'])
    
    if response == 'escape':
        logging.info("User pressed escape during practice condition screen")
        emergency_escape("User pressed escape", window)
        return None
        
    return response or 'space'  # Default to space if response is None


def show_practice_instructions(window, config):
    """Display instructions for the practice trials."""
    from modules.controller import controller_available
    
    # Modify instruction text based on input method
    input_method_text = ""
    if hasattr(config, 'use_controller') and config.use_controller and controller_available:
        input_method_text = (
            "You will use the gamepad controller for this task:\n"
            "- UP on D-pad: Move up/forward\n"
            "- DOWN on D-pad: Move down/backward\n"
            "- LEFT on D-pad: Move left\n"
            "- RIGHT on D-pad: Move right\n"
            "- X button: Confirm selection/Continue\n"
            "- B button: Emergency exit (only if needed)\n\n"
        )
    else:
        input_method_text = (
            "You will use the keyboard for this task:\n"
            "- UP arrow: Move up/forward\n"
            "- DOWN arrow: Move down/backward\n"
            "- LEFT arrow: Move left\n"
            "- RIGHT arrow: Move right\n"
            "- SPACE: Confirm selection/Continue\n"
            "- ESCAPE: Emergency exit (only if needed)\n\n"
        )
    
    practice_instructions = (
        "Welcome to the practice phase!\n\n"
        f"{input_method_text}"
        "You will now practice two different navigation modes:\n\n"
        "1. PLAYER VIEW (2 trials)\n"
        "2. MAP VIEW (2 trials)\n\n"
        f"Press SPACE{' (or X button)' if hasattr(config, 'use_controller') and config.use_controller and controller_available else ''} to continue."
    )

    return show_text_screen(window, practice_instructions, text_color=config.text_color)


def show_controller_status_screen(window, config, controller_status=True):
    """Show the current status of the controller
    
    Args:
        window: PsychoPy window
        config: Experiment configuration
        controller_status: Current status of the controller
    
    Returns:
        str: Response key
    """
    from modules.controller import controller_available
    
    if controller_status and controller_available:
        status_text = (
            "Controller Status: CONNECTED\n\n"
            "The gamepad controller is working correctly.\n\n"
            "- D-pad buttons: Use for navigation (UP, DOWN, LEFT, RIGHT)\n"
            "- X button: Confirm/Continue (like SPACE key)\n"
            "- B button: Emergency exit (when pressed with START or BACK)\n\n"
            "Press X button or SPACE to continue."
        )
    else:
        status_text = (
            "Controller Status: NOT CONNECTED\n\n"
            "No gamepad controller detected or connection lost.\n"
            "Using keyboard controls instead:\n\n"
            "- Arrow keys: Use for navigation (UP, DOWN, LEFT, RIGHT)\n"
            "- SPACE key: Confirm/Continue\n"
            "- ESCAPE key: Emergency exit\n\n"
            "Press SPACE to continue."
        )
    
    return show_text_screen(
        window,
        status_text,
        text_color=(-0.7, -0.7, -0.7),
        key_list=['space', 'escape']
    )