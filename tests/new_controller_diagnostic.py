#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modern PsychoPy Controller + Keyboard Input Diagnostic Tool
============================================================

Updated to use `psychopy.hardware.joystick.devices` instead of deprecated `getNumJoysticks`.
Compatible with PsychoPy ≥ 2021.x
"""

import time
import logging
from psychopy import visual, event, core
from psychopy.hardware import joystick

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_input_diagnostic():
    # Initialize PsychoPy joystick module
    joystick.backend = 'pygame'  # or 'pyglet' if preferred
    joystick.init()

    win = visual.Window(
        size=(800, 600),
        fullscr=False,
        color='black',
        units='norm'
    )

    title = visual.TextStim(win, text="INPUT DIAGNOSTIC TOOL", pos=(0, 0.8), height=0.08, color='white')
    instructions = visual.TextStim(win, text="Press controller buttons or keyboard keys.\nPress ESC to exit.",
                                   pos=(0, 0.6), height=0.05, wrapWidth=1.8, color='white')
    controller_text = visual.TextStim(win, text="No controller input detected", pos=(0, 0.2), height=0.06, color='yellow')
    keyboard_text = visual.TextStim(win, text="No keyboard input detected", pos=(0, -0.2), height=0.06, color='green')
    status_text = visual.TextStim(win, text="", pos=(0, -0.6), height=0.05, color='red')

    devices = joystick.devices
    controller_available = len(devices) > 0
    controller_count = 0
    keyboard_count = 0
    last_controller_time = 0
    last_keyboard_time = 0
    should_exit = False

    if controller_available:
        joy = devices[0]
        status_text.text = f"Controller: {joy.name}\nBackend: {joystick.backend}"
    else:
        joy = None
        status_text.text = "No controller detected"

    while not should_exit:
        # Check controller
        controller_input_detected = False
        if joy:
            buttons = joy.getAllButtons()
            hats = joy.getHats()

            if any(buttons):
                controller_input_detected = True
                controller_count += 1
                last_controller_time = time.time()
                pressed = [i for i, val in enumerate(buttons) if val]
                controller_text.text = f"CONTROLLER INPUT: Buttons {pressed}\nTotal: {controller_count}"

            elif hats and hats[0] != (0, 0):
                controller_input_detected = True
                controller_count += 1
                last_controller_time = time.time()
                controller_text.text = f"CONTROLLER INPUT: Hat {hats[0]}\nTotal: {controller_count}"

        if not controller_input_detected and time.time() - last_controller_time > 1.0:
            controller_text.text = f"No recent controller input\nTotal detected: {controller_count}"

        # Check keyboard
        keys = event.getKeys()
        if keys:
            if 'escape' in keys:
                should_exit = True
            else:
                keyboard_count += 1
                last_keyboard_time = time.time()
                keyboard_text.text = f"KEYBOARD INPUT: {', '.join(keys)}\nTotal: {keyboard_count}"

        if time.time() - last_keyboard_time > 1.0:
            keyboard_text.text = f"No recent keyboard input\nTotal detected: {keyboard_count}"

        # Draw
        title.draw()
        instructions.draw()
        controller_text.draw()
        keyboard_text.draw()
        status_text.draw()
        win.flip()
        core.wait(0.01)

    win.close()
    print("\nInput diagnostic test completed.")
    print(f"Controller inputs detected: {controller_count}")
    print(f"Keyboard inputs detected: {keyboard_count}")

if __name__ == "__main__":
    run_input_diagnostic()
    core.quit()
