# Controller Guide for Spatial Navigation EEG Experiment

## Supported Controllers

This experiment supports Logitech and other standard game controllers via the PsychoPy joystick interface. For optimal performance, we recommend using a Logitech F310 or F710 controller, which are well-tested with this experiment.

## Button Mappings

### Navigation Controls

| Controller Button | Function | Keyboard Equivalent |
|------------------|----------|---------------------|
| D-pad UP         | Move forward/north | UP arrow key |
| D-pad DOWN       | Move backward/south | DOWN arrow key |
| D-pad LEFT       | Move left/west | LEFT arrow key |
| D-pad RIGHT      | Move right/east | RIGHT arrow key |

### Interface Controls

| Controller Button | Function | Keyboard Equivalent |
|------------------|----------|---------------------|
| A button (green) | Confirm/Accept | ENTER/RETURN key |
| X button (blue)  | Continue/Next | SPACEBAR |
| B button (red)   | Back/Cancel | Not implemented |
| START button     | Pause experiment | Not implemented |

### Emergency Exit

If you need to immediately exit the experiment:

| Controller Combination | Function |
|----------------------|----------|
| START + B button     | Emergency exit |
| SELECT + B button    | Emergency exit (alternative) |

This is equivalent to pressing the ESCAPE key on the keyboard.

### Trial Navigation

During trials, use the D-pad to make your directional choices:

- **Egocentric (PLAYER VIEW) navigation**: 
  - UP = Forward
  - DOWN = Backward
  - LEFT = Left
  - RIGHT = Right

- **Allocentric (MAP VIEW) navigation**:
  - UP = North
  - DOWN = South
  - LEFT = West
  - RIGHT = East

- **Control navigation**:
  - Simply press the direction that matches the visible arrow

### Between Trials and Blocks

- Use the **X button** (blue) to advance through instructions and between trials
- Use the **A button** (green) to confirm choices and continue to the next block

## Controller Setup

1. Connect your controller before starting the experiment
2. When prompted, select "Use controller/gamepad" in the setup dialog
3. The experiment will automatically detect your controller

## Troubleshooting

If your controller isn't working properly:

1. Ensure it's properly connected to your computer
2. Check that it's recognized in your operating system's device settings
3. Try disconnecting and reconnecting the controller
4. If problems persist, use the keyboard controls instead

The experiment will automatically fall back to keyboard controls if a controller isn't detected or if controller input fails.
