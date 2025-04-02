#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Block Sequence Visualization Tool
================================

This script displays the possible block sequences for the spatial navigation
experiment based on counterbalancing conditions.
"""

import sys
import os
import argparse

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import Config
from modules.utils import print_block_sequence


def main():
    """Main function to display block sequences"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize block sequences for the experiment')
    parser.add_argument('-p', '--participant', help='Participant ID for reproducible sequence')
    parser.add_argument('-c', '--counterbalance', type=int, choices=[1, 2, 3, 4], 
                        help='Counterbalance condition (1-4)')
    parser.add_argument('-s', '--single', action='store_true', 
                        help='Show only the specified counterbalance (or default if none specified)')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Display block sequences
    if args.counterbalance and args.single:
        # Just show the specifically requested counterbalance
        print_block_sequence(config, args.participant, args.counterbalance)
    else:
        # By default, show all counterbalance conditions
        for cb in range(1, 5):
            if args.counterbalance and args.counterbalance != cb:
                # Skip conditions that don't match the requested counterbalance
                continue
                
            print(f"\n\n*** COUNTERBALANCE CONDITION {cb} ***")
            print_block_sequence(config, args.participant, cb)
    
    # Print experiment summary
    print("\nEXPERIMENT SUMMARY:")
    print(f"Total blocks: {sum(config.condition_blocks.values())}")
    print(f"Trials per block: {config.trials_per_block}")
    print(f"Total trials: {config.total_trials}")
    print("\nCondition distribution:")
    
    # Format the condition_blocks dictionary for display
    for (nav_type, diff), count in config.condition_blocks.items():
        print(f"  {nav_type.capitalize()}/{diff}: {count} blocks × {config.trials_per_block} trials = {count * config.trials_per_block} trials")


if __name__ == "__main__":
    main()
