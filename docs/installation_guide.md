# Installation Guide for Spatial Navigation EEG Experiment

This guide provides step-by-step instructions for installing and running the Spatial Navigation EEG Experiment on a new computer.

## System Requirements

- Windows 10/11, macOS 10.14+, or Linux
- Python 3.7-3.9 (3.8 recommended)
- 8GB RAM minimum (16GB recommended)
- 500MB free disk space
- Monitor with resolution of at least 1024x768
- Internet connection for initial setup

## Installation Steps

### 1. Transfer the Experiment Files

There are two ways to transfer the experiment to a new computer:

**Option A: Copy the directory**
- Copy the entire `spatial_navigation_eeg` directory to the target computer
- Make sure to preserve the directory structure

**Option B: Clone from version control (if available)**
- If the experiment is in a Git repository, clone it:
  ```
  git clone <repository-url> spatial_navigation_eeg
  cd spatial_navigation_eeg
  ```

### 2. Install Python

If Python is not already installed:

1. Download Python 3.8 from [python.org](https://www.python.org/downloads/)
2. During installation:
   - Check "Add Python to PATH"
   - Choose "Customize installation" and ensure pip is selected
   - Install for all users if you have administrator access

Verify the installation by opening a command prompt/terminal:
```
python --version
```

### 3. Install PsychoPy

**Option A: PsychoPy Standalone (recommended for beginners)**
- Download PsychoPy standalone from [psychopy.org](https://www.psychopy.org/download.html)
- Install following the on-screen instructions
- Launch PsychoPy and ensure it runs correctly

**Option B: Install via pip (for advanced users)**
```
pip install psychopy
```

### 4. Create and Activate a Virtual Environment (Recommended)

Using a virtual environment prevents conflicts with other Python installations:

**On Windows:**
```
cd spatial_navigation_eeg
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```
cd spatial_navigation_eeg
python -m venv venv
source venv/bin/activate
```

### 5. Install Required Dependencies

Install all dependencies from the requirements.txt file:

```
pip install -r requirements.txt
```

This will install:
- psychopy
- numpy
- pandas
- pylsl (for EEG integration)
- pygame (for controller support)

### 6. Check for System-Specific Dependencies

**For Windows:**
- For EEG parallel port support (BrainVision TriggerBox):
  ```
  pip install pyparallel
  ```

**For macOS:**
- Install PyGame dependencies:
  ```
  brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
  ```

**For Linux:**
- Install required system packages:
  ```
  sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
  ```

### 7. Verify Directory Structure

Ensure your directories match this structure:
```
spatial_navigation_eeg/
├── config/
│   └── stimulus_mappings.csv
├── data/
├── logs/
├── modules/
├── stimuli/
│   ├── easy/
│   ├── hard/
│   └── control/
├── docs/
├── requirements.txt
└── main.py
```

The experiment will create missing directories at runtime, but it's best to verify they exist first.

### 8. Test the Installation

1. Run the experiment with basic testing:
```
python main.py --test
```

2. Test controller detection if using a gamepad:
```
python main.py --test-controller
```

3. Test EEG integration if applicable:
```
python main.py --test-eeg
```

## Common Issues and Solutions

### Missing Dependencies
If you see errors about missing modules:
```
pip install psychopy numpy pandas pylsl pygame
```

### PsychoPy Import Errors
If you get PsychoPy import errors:
1. Make sure you're using Python 3.7-3.9 (Python 3.10+ may have compatibility issues)
2. Try reinstalling PsychoPy: `pip uninstall psychopy && pip install psychopy`

### Controller Not Detected
1. Connect the controller before starting the experiment
2. Make sure pygame is properly installed: `pip install --upgrade pygame`
3. On Linux, you might need additional permissions: `sudo chmod a+rw /dev/input/js*`

### EEG Connection Issues
1. Check that BrainVision Recorder is running (if using BrainVision)
2. Verify LSL installations: `pip install pylsl`
3. For parallel port issues, check port address in config.py

### Display/Resolution Issues
1. Update your graphics drivers
2. In config.py, modify `screen_resolution` to match your display
3. Set `fullscreen = False` for windowed mode

## EEG-Specific Setup

If you're using the experiment with EEG equipment:

### BrainVision Setup
1. Start BrainVision Recorder before running the experiment
2. Enable Remote Control Server in BrainVision settings
3. Configure port settings to match those in config.py

### LSL Setup
1. Install LSL libraries: `pip install pylsl`
2. Configure your EEG system to receive LSL markers
3. Set `use_lsl = True` in config.py

### Parallel Port Setup (Windows only)
1. Install the parallel port driver appropriate for your system
2. Verify parallel port address in Device Manager (typically 0x378)
3. Update `parallel_port_address` in config.py

## Running Without EEG

To run the experiment without EEG (behavioral only):
1. Open modules/config.py
2. Set these options:
   ```python
   self.use_eeg = False
   self.use_brainvision_rcs = False
   self.use_lsl = False
   self.use_parallel = False
   self.use_tcp_markers = False
   ```
3. Run the experiment normally: `python main.py`

## For More Information

- See README.md for general information
- See controller_guide.md for gamepad setup
- See BrainVision_Integration.md for detailed EEG setup
