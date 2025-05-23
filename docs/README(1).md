# Spatial Navigation EEG Experiment

This is a PsychoPy experiment for testing spatial navigation abilities while recording EEG data.

## Installation

### Requirements
- Python 3.6+ (Python 3.8 or 3.9 recommended)
- PsychoPy
- Required packages (see requirements.txt)

### Setup Instructions

1. **Install PsychoPy**

   Option 1: Install PsychoPy standalone application from [psychopy.org](https://www.psychopy.org/download.html)
   
   Option 2: Install via pip (recommended for advanced users):
   ```
   pip install psychopy
   ```

2. **Install additional dependencies**
   ```
   pip install -r requirements.txt
   ```
   
   This will install all required packages including numpy, pandas, pylsl, and pygame.

3. **Verify directory structure**
   
   Make sure your directory structure matches the one described in the "Project Structure" section below.
   The experiment will create necessary directories if they don't exist.

## Cross-Platform Compatibility

This experiment is designed to run on Windows, macOS, and Linux. The following features ensure portability:

- Uses relative paths for file access
- Uses platform-independent Python shebang (`#!/usr/bin/env python`)
- Avoids system-specific dependencies
- Provides fallbacks for controller/gamepad input
- Creates necessary directories automatically

## Usage

### Running the Full Experiment
To run the full experiment:
```
python main.py
```

### Testing Controller Support
To test if your controller is working properly:
```
python main.py --test-controller
```

### Testing EEG Triggers
To test EEG trigger functionality:
```
python main.py --test-eeg
```

### Configuration
Edit the `modules/config.py` file to modify experiment parameters:
- `monitor_index`: Set to 0 for primary monitor, 1 for secondary monitor, etc.
- `use_controller`: Set to True/False to enable/disable controller support
- `use_parallel`: Enable/disable parallel port TTL triggers
- `use_lsl`: Enable/disable Lab Streaming Layer for EEG
- Other experimental parameters

## Git Repository

This project is version-controlled with Git. For instructions on cloning or contributing to the repository, see `git_setup.md`.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'X'**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Make sure you're using the correct Python environment

2. **EEG Connection Issues**
   - Check that your EEG system is properly connected
   - Verify LSL or parallel port settings in config.py match your system

3. **Controller Not Detected**
   - Make sure your controller is connected before starting the experiment
   - Try a different USB port
   - The experiment will automatically fall back to keyboard controls if no controller is detected

4. **Display Issues**
   - Use the monitor selection dialog to choose the appropriate monitor
   - Adjust screen resolution settings in config.py if needed

## Project Structure

```
spatial_navigation_eeg/
├── config/                     # Configuration files
│   └── stimulus_mappings.csv   # Maps stimuli to conditions and correct responses
├── data/                       # Experiment data is saved here
├── logs/                       # Log files
├── modules/                    # Python modules
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration parameters
│   ├── controller.py           # Controller/gamepad support
│   ├── data_handler.py         # Data saving and management
│   ├── eeg.py                  # EEG integration
│   ├── experiment.py           # Core experiment functionality
│   ├── experiment_helper.py    # Helper functions for the experiment
│   ├── instructions.py         # Instruction text and display
│   ├── stimulus.py             # Stimulus management
│   ├── ui.py                   # User interface components
│   └── utils.py                # Utility functions
├── stimuli/                    # Stimulus images
│   ├── easy/                   # Easy difficulty stimuli
│   ├── hard/                   # Hard difficulty stimuli
│   └── control/                # Control condition stimuli
├── tests/                      # Test scripts
├── docs/                       # Documentation
├── .gitignore                  # Git ignore file
├── requirements.txt            # Required Python packages
└── main.py                     # Main entry point
```

## Experimental Design

The experiment uses a 3×2 design:

- **Difficulty Levels**: Easy, Hard, and Control
- **Navigation Types**: Egocentric (relative to player orientation) and Allocentric (fixed compass directions)

Each condition (Easy-Ego, Easy-Allo, Hard-Ego, Hard-Allo, Control-Ego, Control-Allo) is presented multiple times during the experiment, with trials balanced across conditions.

## Task Instructions

### Egocentric Condition
Participants navigate based on the player's perspective (forward/backward/left/right relative to the direction the player is facing).

### Allocentric Condition
Participants navigate based on fixed compass directions (north/south/east/west), regardless of player orientation.

### Control Condition
This condition shows the path with arrows, and participants simply follow the indicated direction.

## EEG Integration

The experiment includes comprehensive EEG trigger codes for all events:
- Experiment start/end
- Block start/end
- Trial events (stimulus onset, response, feedback)
- Condition-specific triggers

The experiment supports multiple EEG integration methods:
- LSL (Lab Streaming Layer)
- Parallel port triggers (for BrainVision or other systems)
- BrainVision Remote Control Server
- TCP marker system (for custom EEG setups)

## Data Output

The experiment saves data in CSV format with comprehensive information:
- Participant information
- Trial details (condition, stimulus, response)
- Timing information
- Accuracy and response time
- Block summaries
- EEG trigger timing and codes