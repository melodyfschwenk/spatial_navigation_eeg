#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Controller Debug Tool for Logitech F310
=======================================

This tool provides a detailed view of all button presses and axis movements
to help diagnose controller mapping issues on your specific machine.
"""

import os
import sys
import time
import pygame

# Initialize pygame
pygame.init()
pygame.joystick.init()

# Set up display
pygame.display.set_caption("Controller Debug Tool")
screen = pygame.display.set_mode((800, 600))
font = pygame.font.SysFont('Arial', 16)
large_font = pygame.font.SysFont('Arial', 24, bold=True)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (150, 150, 150)

def draw_text(text, font, color, x, y, align="left"):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if align == "center":
        text_rect.center = (x, y)
    elif align == "right":
        text_rect.right = x
        text_rect.top = y
    else:
        text_rect.left = x
        text_rect.top = y
    screen.blit(text_surface, text_rect)

def main():
    # Check for joysticks
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No controllers found!")
        draw_text("No controllers detected! Please connect a controller.", large_font, RED, 400, 300, "center")
        pygame.display.flip()
        time.sleep(3)
        return
    
    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    # Get controller info
    name = joystick.get_name()
    axes = joystick.get_numaxes()
    buttons = joystick.get_numbuttons()
    hats = joystick.get_numhats()
    
    print(f"Controller: {name}")
    print(f"Axes: {axes}")
    print(f"Buttons: {buttons}")
    print(f"Hats: {hats}")
    
    # Create empty lists to store button and axis states
    button_states = [False] * buttons
    axis_states = [0.0] * axes
    hat_states = [(0, 0)] * hats
    
    # Map of button names (for Logitech F310 - may vary on other controllers)
    button_names = {
        0: "A", 
        1: "B", 
        2: "X", 
        3: "Y",
        4: "Left Shoulder",
        5: "Right Shoulder",
        6: "Back",
        7: "Start",
        8: "Left Stick Press",
        9: "Right Stick Press"
    }
    
    # Map of hat directions
    hat_directions = {
        (0, 0): "Center",
        (0, 1): "Up",
        (0, -1): "Down",
        (-1, 0): "Left",
        (1, 0): "Right",
        (-1, 1): "Up-Left",
        (1, 1): "Up-Right",
        (-1, -1): "Down-Left",
        (1, -1): "Down-Right"
    }
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Process controller input
        pygame.event.pump()
        
        # Get button states
        for i in range(buttons):
            button_states[i] = joystick.get_button(i)
        
        # Get axis states
        for i in range(axes):
            axis_states[i] = joystick.get_axis(i)
        
        # Get hat states
        for i in range(hats):
            hat_states[i] = joystick.get_hat(i)
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw controller info
        draw_text(f"Controller: {name}", large_font, WHITE, 400, 30, "center")
        draw_text(f"Press ESC to exit", font, GRAY, 400, 60, "center")
        
        # Draw button states
        draw_text("Button States:", large_font, WHITE, 50, 100)
        for i in range(buttons):
            color = GREEN if button_states[i] else WHITE
            button_name = button_names.get(i, f"Button {i}")
            draw_text(f"{button_name} (#{i}): {button_states[i]}", font, color, 50, 130 + i * 20)
        
        # Draw hat states
        y_offset = 130 + buttons * 20 + 20
        draw_text("D-pad/Hat States:", large_font, WHITE, 50, y_offset)
        for i in range(hats):
            hat_value = hat_states[i]
            direction = hat_directions.get(hat_value, str(hat_value))
            color = YELLOW if hat_value != (0, 0) else WHITE
            draw_text(f"Hat {i}: {hat_value} ({direction})", font, color, 50, y_offset + 30 + i * 20)
        
        # Draw axis states
        draw_text("Axis States:", large_font, WHITE, 450, 100)
        for i in range(axes):
            # Color gradient from white to blue based on axis value
            value = axis_states[i]
            color = BLUE if abs(value) > 0.1 else WHITE
            draw_text(f"Axis {i}: {value:.2f}", font, color, 450, 130 + i * 20)
        
        # Draw PsychoPy mapping information
        mapping_y = 450
        draw_text("PsychoPy Controller Mapping:", large_font, WHITE, 400, mapping_y, "center")
        draw_text("- A (0): Bottom button", font, WHITE, 400, mapping_y + 30, "center")
        draw_text("- B (1): Right button", font, WHITE, 400, mapping_y + 50, "center")
        draw_text("- X (2): Left button", font, WHITE, 400, mapping_y + 70, "center")
        draw_text("- Y (3): Top button", font, WHITE, 400, mapping_y + 90, "center")
        draw_text("- If D-pad is not working, check if it uses HAT or AXIS", font, YELLOW, 400, mapping_y + 120, "center")
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(30)
    
    # Clean up
    pygame.quit()

if __name__ == "__main__":
    main()