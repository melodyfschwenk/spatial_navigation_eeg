# Spatial Navigation EEG Experiment - Research Assistant Checklist

## Pre-Session Preparation (1 hour before participant arrival)

### Equipment Check
- [ ] Turn on EEG amplifier and allow to warm up (minimum 30 minutes before use)
- [ ] Turn on experiment computer and monitor
- [ ] Check that trigger cable is securely connected between experiment computer and EEG amplifier
- [ ] Start BrainVision Recorder software and verify it opens without errors
- [ ] Ensure experiment software runs correctly with `python main.py --test-eeg` command
- [ ] Verify EEG trigger transmission is working between experiment computer and BrainVision
- [ ] Check that all 64 electrodes are present and undamaged
- [ ] Test impedance measurement functionality

### Consumables Preparation
- [ ] Fill minimum 3 syringes with electrolyte gel (ensure no air bubbles)
- [ ] Place all syringes in syringe holder with tips facing up
- [ ] Prepare alcohol wipes (minimum 10)
- [ ] Prepare cotton pads and skin preparation gel
- [ ] Prepare non-latex gloves in appropriate sizes
- [ ] Prepare face tissues/paper towels
- [ ] Prepare 2 clean towels for participant
- [ ] Set out disinfectant spray for equipment cleaning
- [ ] Ensure measuring tape is available for head measurements

### Workspace Preparation
- [ ] Ensure testing room is clean and organized
- [ ] Adjust participant chair height for comfort
- [ ] Set room temperature to comfortable level (20-22°C recommended)
- [ ] Dim lights to reduce electrical noise and eye strain
- [ ] Select appropriate EEG cap size based on scheduled participant information (if available)
- [ ] Position chin rest and set at appropriate height
- [ ] Remove any sources of electrical interference from the room
- [ ] Close all unnecessary software on recording computer
- [ ] Disable automatic updates and notifications on all computers

## Participant Arrival

### Welcome and Setup (10-15 minutes)
- [ ] Greet participant and explain the general procedure
- [ ] Have participant complete/sign informed consent form
- [ ] Review exclusion criteria (recent head injury, neurological conditions, etc.)
- [ ] Ask participant to silence and put away all electronic devices
- [ ] Ask about hair products used today (gel/hairspray can interfere with recordings)
- [ ] Direct participant to restroom if needed (should be done before cap application)
- [ ] Ask participant to remove earrings, necklaces, and other metal objects
- [ ] Measure participant's head circumference at widest point (above ears and eyebrows)
- [ ] Select appropriate cap size based on measurement:
  - Small: 54-56cm
  - Medium: 56-58cm
  - Large: 58-60cm
  - Extra Large: 60-62cm

## EEG Cap Application (20-30 minutes)

### Cap Placement
- [ ] Put on non-latex gloves
- [ ] Show cap to participant and explain application process
- [ ] Position cap on participant's head ensuring:
  - Cz electrode is at vertex (center point)
  - Fpz electrode is 10% of nasion-inion distance above nasion
  - Cap is symmetrical (equal distance from ears on both sides)
  - Front edge of cap is approximately 1cm above eyebrows
- [ ] Secure cap with chin strap, ensuring comfortable but firm fit
- [ ] Connect ground (GND) and reference (REF) electrodes first

### Electrode Preparation
- [ ] Using blunt syringe tip, part hair under each electrode site
- [ ] Apply gel to each electrode (~1.5ml per electrode):
  - Start with Cz and work outward symmetrically
  - Ensure gel makes contact with scalp but doesn't spread between electrodes
  - Use consistent, controlled pressure on syringe
  - Abrade skin gently through electrode opening using syringe tip
- [ ] Check GND and REF impedances first (should be below 5 kΩ)
- [ ] Apply gel to remaining electrodes systematically:
  - Central electrodes (C3, C4, etc.)
  - Frontal electrodes (F3, F4, etc.)
  - Parietal electrodes (P3, P4, etc.)
  - Temporal electrodes (T7, T8, etc.)
  - Occipital electrodes (O1, O2, etc.)

### External Electrodes
- [ ] Prepare skin for external electrodes using alcohol wipes
- [ ] Apply EOG (electro-oculogram) electrodes:
  - VEOG: One electrode above and one below the left eye
  - HEOG: One electrode at outer canthus of each eye
- [ ] Apply mastoid electrodes (behind each ear) if required by protocol
- [ ] Secure all external electrode wires to prevent pulling

### Impedance Check
- [ ] Connect electrode cable bundle to amplifier
- [ ] Open impedance measurement in BrainVision Recorder
- [ ] Adjust all electrodes until impedances are:
  - GND and REF: below 5 kΩ (critical)
  - All other electrodes: below 10 kΩ (preferably below 5 kΩ)
- [ ] Document any problematic electrodes that cannot reach target impedance
- [ ] Save impedance values as reference

## Recording Setup (5-10 minutes)

### BrainVision Configuration
- [ ] Create new participant file in BrainVision with naming convention: "SNAV_P[ID]_S[Session]_[Date]"
- [ ] Set sampling rate to 1000Hz
- [ ] Set appropriate filters:
  - Low cutoff: 0.1 Hz
  - High cutoff: 100 Hz
  - Notch: 50 Hz (or 60 Hz depending on country)
- [ ] Verify Remote Control Server is enabled (Configuration → Preferences → Remote Control Server)
- [ ] Verify marker settings if using LSL (Configuration → Preferences → Lab Streaming Layer)

### Signal Quality Assessment
- [ ] Start monitoring EEG without recording
- [ ] Ask participant to:
  - Sit still with eyes open (30 seconds) - verify alpha suppression
  - Close eyes (30 seconds) - verify alpha rhythm present
  - Blink 5 times - verify blink artifacts visible in frontal channels
  - Clench jaw - identify muscle artifacts
  - Move eyes left/right - verify HEOG captures movement
  - Move eyes up/down - verify VEOG captures movement
- [ ] Check for 60Hz (or 50Hz) line noise in signals
- [ ] Address any problematic electrodes or excessive artifacts
- [ ] If signals look good, stop monitoring (do not save this test)

## Experiment Execution (60-90 minutes)

### Final Preparations
- [ ] Position participant at correct distance from monitor
- [ ] Ensure comfortable seating position that can be maintained
- [ ] Remind participant to minimize head/body movement
- [ ] Remind participant about task instructions and response method
- [ ] Verify experiment computer and BrainVision computer are communicating
- [ ] Check trigger cable connection once more

### Starting Recording
- [ ] Start BrainVision recording FIRST
- [ ] Wait 30 seconds of baseline recording before beginning task
- [ ] Annotate recording with "Baseline Start" and "Baseline End" markers
- [ ] Start experiment with command: `python main.py`
- [ ] Verify first trigger markers appear in BrainVision recording
- [ ] Monitor for trigger reception throughout experiment

### During Recording
- [ ] Monitor EEG for major artifacts or signal issues
- [ ] Note any unusual events, technical issues or participant behaviors
- [ ] Avoid unnecessary movement or talking in the recording room
- [ ] Periodically check that triggers continue to be recorded
- [ ] Monitor participant comfort and provide breaks if needed
- [ ] During any breaks, annotate recording with "Break Start" and "Break End" markers

### Ending Recording
- [ ] Wait for experiment completion confirmation message
- [ ] Annotate recording with "Experiment Completed" marker
- [ ] Record 30 seconds of post-experiment baseline
- [ ] Stop and save BrainVision recording
- [ ] Verify experiment data files are saved correctly on experiment computer

## Post-Session Procedures (20-30 minutes)

### Participant Cleanup
- [ ] Put on fresh gloves
- [ ] Carefully remove EOG and external electrodes
- [ ] Gently remove EEG cap
- [ ] Offer participant towel and direct to sink/bathroom
- [ ] Provide hair washing facilities if available
- [ ] Complete participant payment/credit forms

### Equipment Cleanup
- [ ] Rinse cap thoroughly with warm water (no soap)
- [ ] Clean electrodes individually, ensuring all gel is removed
- [ ] Disinfect cap and external electrodes according to lab protocol
- [ ] Allow cap to air dry completely before storing
- [ ] Clean chin rest, participant chair, and other contacted surfaces
- [ ] Dispose of used consumables properly
- [ ] Check syringe supply and refill as needed for next session

### Data Management
- [ ] Verify all files have been saved with correct naming convention
- [ ] Check that the following files exist:
  - BrainVision data files (.eeg, .vhdr, .vmrk)
  - Experiment data files (CSV in data directory)
  - EEG trigger log file (JSON in data directory)
- [ ] Create backup of all data files to external storage/server
- [ ] Complete session documentation form with:
  - Participant ID and session number
  - Date and time of recording
  - Names of experimenters present
  - Equipment used (amplifier serial number, cap size)
  - Notes on any technical issues or artifacts
  - Electrodes with high impedance values
  - Deviations from standard protocol

### Quick Quality Check
- [ ] Open EEG file in viewer and verify:
  - Recording duration matches expected experiment length
  - All channels were recorded properly
  - Trigger markers are present throughout recording
  - No major continuous artifacts or flatlined channels

## Final Tasks

### Session Documentation
- [ ] Update participant tracking spreadsheet
- [ ] File physical consent forms in appropriate location
- [ ] Update lab calendar for completed session
- [ ] Report any technical issues or equipment needs to lab manager
- [ ] Prepare session summary for PI if required

### Room Reset
- [ ] Return room to neutral state for next session
- [ ] Turn off amplifier if no sessions remaining for the day
- [ ] Lock lab door if leaving room unattended

## Emergency Procedures

### Data Saving Issues
- [ ] If BrainVision crashes, check "tmp" directory for recovery files
- [ ] For experiment crashes, check logs directory for error information

### Equipment Failures
- [ ] Document exact nature of failure
- [ ] Contact technical support with error codes/messages
- [ ] If recording interrupted, save partial data before attempting fixes

### Participant Distress
- [ ] Stop recording immediately if participant reports significant discomfort
- [ ] Remove cap if participant reports pain or significant distress
- [ ] Contact supervisor if unsure how to proceed
