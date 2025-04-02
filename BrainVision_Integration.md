# BrainVision Integration Guide for Spatial Navigation EEG

This guide provides instructions for setting up and running the Spatial Navigation EEG experiment with BrainVision EEG systems, with a focus on detailed event marking for mu rhythm and frontal coherence analysis.

## Supported Integration Methods

The experiment supports multiple methods to send markers to BrainVision:

1. **Remote Control Server (RCS)** - Primary method
2. **Lab Streaming Layer (LSL)** - Secondary method
3. **Parallel Port** - For TriggerBox
4. **TCP Markers** - For custom setups

## Setup Instructions

### 1. Remote Control Server (Recommended)

BrainVision's Remote Control Server allows direct communication between PsychoPy and the BrainVision Recorder.

1. **In BrainVision Recorder**:
   - Go to `Configuration` → `Preferences` → `Remote Control Server`
   - Enable the Remote Control Server
   - Keep the default port (6700) or set a custom one
   - Click "Apply" and "OK"

2. **In the experiment configuration**:
   - Ensure `use_brainvision_rcs = True` in `config.py`
   - Set `brainvision_rcs_host` to the IP of the computer running BrainVision Recorder (use "127.0.0.1" if same computer)
   - Set `brainvision_rcs_port` to match the port in BrainVision Recorder

3. **Testing the connection**:
   - Run `test_brainvision_connection()` from the Python console to verify connectivity

### 2. Lab Streaming Layer

LSL provides an alternative method that works well with BrainVision's LSL integration.

1. **In BrainVision Recorder**:
   - Go to `Configuration` → `Preferences` → `Lab Streaming Layer`
   - Enable "Use Lab Streaming Layer"
   - Enable "Import Markers from Stream"
   - Set "Marker Stream Name" to match `brainvision_lsl_name` in `config.py`
   
2. **In the experiment configuration**:
   - Ensure `use_lsl = True` in `config.py`
   - Set `brainvision_lsl_name = "BrainVision RDA Markers"` (standard name)

### 3. Parallel Port (TriggerBox)

For hardware trigger synchronization:

1. **Hardware setup**:
   - Connect the BrainVision TriggerBox to a parallel port
   - Connect the TriggerBox to your amplifier

2. **In BrainVision Recorder**:
   - Go to `Configuration` → `Preferences` → `Digital Port Settings`
   - Select the appropriate port type (usually EEG or Aux)
  
3. **In the experiment configuration**:
   - Set `use_parallel = True` in `config.py`
   - Set `parallel_port_address` to your parallel port address (commonly 0x378 for LPT1)

## Marker Codes for Mu Rhythm Analysis

The experiment includes a comprehensive set of trigger codes optimized for mu rhythm and frontal coherence analysis:

### Core Event Markers
- `10`: Fixation onset (baseline period)
- `11`: Stimulus onset
- `12`: Stimulus offset
- `20-23`: Response markers
- `30-31`: Feedback markers

### Combined Condition Markers
- `100-102`: Egocentric conditions (easy/hard/control)
- `110-112`: Allocentric conditions (easy/hard/control)

### Response and Direction Markers
- `200-203`: Egocentric responses (up/down/left/right)
- `210-213`: Allocentric responses (up/down/left/right)
- `300-303`: Egocentric directions (forward/backward/left/right)
- `310-313`: Allocentric directions (north/south/west/east)

### Outcome Markers
- `400-471`: Combined condition and outcome markers

## Data Analysis for Mu Rhythm

The experiment generates several files to support mu rhythm analysis:

1. **Main trial data**: CSV with precise timing information
2. **Trigger log**: JSON with all triggers and timing
3. **BIDS-compatible events file**: For easy integration with EEGLAB/MNE-Python
4. **Block summary files**: For condition-based analysis

### Analysis Recommendations

For mu rhythm analysis:

1. Use the baseline period (fixation onset to stimulus onset) for baseline correction
2. Analyze mu suppression during:
   - Planning phase: Stimulus onset to response
   - Feedback phase: Response to feedback offset
3. For each condition, compare lateralized mu (C3/C4) to assess motor planning differences
4. Calculate frontal coherence between sensorimotor and frontal regions during different navigation strategies
5. Use event markers to precisely segment the EEG data into relevant epochs

## Troubleshooting

### BrainVision RCS Connection Issues

If unable to connect to BrainVision Recorder:
- Verify Recorder is running and the Remote Control Server is enabled
- Check firewall settings if using separate computers
- Try using `127.0.0.1` as host if running on same computer
- Verify the port number matches in both the experiment and Recorder

### Marker Synchronization

If markers aren't appearing in BrainVision Recorder:
- Check the console output for successful marker transmission
- Try an alternative marker method (switch between RCS and LSL)
- For LSL, ensure the stream name matches exactly
- For parallel port, check that the address is correct

### Missing Markers

If specific markers are missing:
- Check the `trigger_log` JSON file to verify if the marker was sent
- Look for specific error messages in the log files
- Verify timing between markers (markers sent too close together might be dropped)

## Contact

If you encounter issues specific to BrainVision integration, please create an issue in the repository with:
1. The specific error message
2. BrainVision product/version information
3. Integration method being used
4. Console output showing marker transmission attempts