#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spatial Navigation EEG Experiment - Stimulus Setup
=================================================

This script sets up the stimulus directory structure and creates sample stimuli
for testing the experiment.
"""

import os
import sys
import csv
import random
import shutil
from pathlib import Path

# Import config
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
try:
    from modules.config import Config
except ImportError:
    print("Error importing Config. Make sure the 'modules' directory exists.")
    sys.exit(1)

def create_sample_stimulus_mapping(config):
    """Create a sample stimulus mapping CSV file for testing"""
    mapping_file = config.stimulus_map_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    
    # Check if mapping file already exists
    if os.path.exists(mapping_file):
        print(f"Stimulus mapping file already exists: {mapping_file}")
        overwrite = input("Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            return False
    
    # Sample stimulus IDs
    stimulus_ids = [f"stim_{i:03d}" for i in range(1, 51)]
    
    # Create CSV file
    with open(mapping_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'stimulus_id', 'navigation_type', 'difficulty', 
            'egocentric_correct_response', 'allocentric_correct_response'
        ])
        
        # Create sample mappings
        for stim_id in stimulus_ids:
            # Randomly assign navigation type and difficulty
            nav_type = random.choice(['egocentric', 'allocentric'])
            difficulty = random.choice(['easy', 'hard', 'control'])
            
            # Randomly assign correct responses for each navigation type
            ego_response = random.choice(['up', 'down', 'left', 'right'])
            allo_response = random.choice(['up', 'down', 'left', 'right'])
            
            writer.writerow([
                stim_id, nav_type, difficulty, ego_response, allo_response
            ])
    
    print(f"Created sample stimulus mapping file: {mapping_file}")
    return True

def create_sample_stimuli(config):
    """Create sample stimulus files for testing"""
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create directories if they don't exist
        for difficulty, path in config.stim_paths.items():
            os.makedirs(path, exist_ok=True)
            
            print(f"Creating sample stimuli for {difficulty} difficulty in {path}")
            
            # Create 5 sample stimuli for each difficulty
            for i in range(1, 6):
                stimulus_id = f"stim_{i:03d}"
                file_path = os.path.join(path, f"{stimulus_id}.png")
                
                if os.path.exists(file_path):
                    continue  # Skip if file already exists
                
                # Create a simple image with text
                img = Image.new('RGB', (500, 500), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)
                
                # Try to use a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 32)
                except IOError:
                    font = ImageFont.load_default()
                
                # Draw text
                draw.text(
                    (250, 100), 
                    f"Sample Stimulus", 
                    fill=(0, 0, 0), 
                    font=font, 
                    anchor="mm"
                )
                draw.text(
                    (250, 250), 
                    f"ID: {stimulus_id}", 
                    fill=(0, 0, 0), 
                    font=font, 
                    anchor="mm"
                )
                draw.text(
                    (250, 400), 
                    f"Difficulty: {difficulty}", 
                    fill=(0, 0, 0), 
                    font=font, 
                    anchor="mm"
                )
                
                # Save the image
                img.save(file_path)
                print(f"  Created {file_path}")
    
    except ImportError:
        print("Could not create sample stimuli. PIL (Pillow) package required.")
        print("Please install it using: pip install Pillow")
        print("Or create stimulus images manually.")
        return False
    
    except Exception as e:
        print(f"Error creating sample stimuli: {e}")
        return False
    
    return True

def main():
    """Setup stimulus structure and create sample files"""
    print("Spatial Navigation EEG Experiment - Stimulus Setup")
    print("=================================================")
    
    try:
        config = Config()
        print("Configuration loaded.")
        
        # Create stimulus directories
        print("\nChecking stimulus directories...")
        for difficulty, path in config.stim_paths.items():
            if os.path.exists(path):
                print(f"  ✓ {difficulty} directory exists: {path}")
            else:
                print(f"  ✗ {difficulty} directory doesn't exist, creating: {path}")
                os.makedirs(path, exist_ok=True)
        
        # Check stimulus mapping file
        print("\nChecking stimulus mapping...")
        if os.path.exists(config.stimulus_map_path):
            print(f"  ✓ Stimulus mapping file exists: {config.stimulus_map_path}")
        else:
            print(f"  ✗ Stimulus mapping file doesn't exist: {config.stimulus_map_path}")
            print("    Creating sample mapping file...")
            create_sample_stimulus_mapping(config)
        
        # Check for stimulus files
        print("\nChecking for stimulus files...")
        any_stimuli_found = False
        for difficulty, path in config.stim_paths.items():
            stimuli = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.bmp', '.gif'))]
            if stimuli:
                any_stimuli_found = True
                print(f"  ✓ Found {len(stimuli)} stimulus files in {difficulty} directory")
            else:
                print(f"  ✗ No stimulus files found in {difficulty} directory: {path}")
        
        # Create sample stimuli if needed
        if not any_stimuli_found:
            print("\nNo stimulus files found. Creating sample stimuli for testing...")
            create_sample_stimuli(config)
            
        print("\nSetup complete! You can now run the experiment.")
        print("Make sure to replace the sample stimuli with your actual experimental stimuli.")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
