# Spatial Navigation EEG Experiment

## Objective
The experiment investigates the cognitive and neural mechanisms of spatial navigation by comparing egocentric and allocentric strategies. A control condition is included to isolate navigation-specific processes. The task focuses on the initial decision-making process, where participants determine the first move they would make to navigate toward the target.

## Task Description
Participants navigate a 5x5 grid virtual environment. The grid is displayed on a computer screen, and participants must decide the first move they would make to reach the target. The task is divided into three navigation types:

**Egocentric Navigation:**
- Decisions are based on the participant's current perspective.
- Example: If facing north, "turn left" means moving west.

**Allocentric Navigation:**
- Decisions are based on a map-like understanding of the grid.
- Example: "Move north" refers to the grid's fixed orientation, regardless of the participant's current perspective.

**Control Condition:**
- Participants follow a sequence of arrows displayed on the screen, indicating the correct path.
- This condition does not require spatial reasoning.

## Stimuli
**Grid Layout:**
- A 5x5 grid is displayed on the screen.
- Each cell is visually distinct, with clear boundaries.

**Starting Position:**
- Represented by a gray triangle, indicating the participant's initial orientation (e.g., facing north).

**Target Position:**
- Represented by a red stop sign, marking the goal.

**Obstacles:**
- Blue walls block certain paths, requiring participants to plan alternative routes.

**Arrows (Control Condition):**
- Arrows appear sequentially, guiding participants step-by-step.

## Instructions to Participants

### Egocentric Navigation (PLAYER VIEW):
```
In this task, you will move from the gray player to the red stop sign while avoiding blue walls.
The gray triangle shows which way the player is facing.
Your job is to choose the first step the player should take. Make your choice as if you are the player looking in the direction of the triangle.

Use these keys:
UP arrow: Move forward (in the direction the player is facing)
DOWN arrow: Move backward
LEFT arrow: Move to the player's left
RIGHT arrow: Move to the player's right

Example: UP moves you forward in whatever direction you're facing.
Choose the first step needed to reach the stop sign. Try to respond quickly and correctly.
```

### Allocentric Navigation (MAP VIEW):
```
In this task, you will move from the gray player to the red stop sign while avoiding blue walls.
Your job is to choose the first step the player should take. Make your choice based on screen directions (like using a map).

Use these keys:
UP arrow: Move toward the top of the screen
DOWN arrow: Move toward the bottom of the screen
LEFT arrow: Move toward the left side of the screen
RIGHT arrow: Move toward the right side of the screen

No matter which way the player is facing, pressing UP always moves toward the top of the screen.
Choose the first step needed to reach the target. Try to respond quickly and correctly.
```

### Control Condition (ARROW FOLLOWING):
```
In this task, you will see arrows showing the path from the player to the target.
Your job is to follow the first arrow from the player's position.

Use these keys:
UP arrow: When the first arrow points up
DOWN arrow: When the first arrow points down
LEFT arrow: When the first arrow points left
RIGHT arrow: When the first arrow points right

Example: Press the RIGHT arrow key if the first arrow points right.
Try to respond quickly and correctly.
```

## Procedure
**Instructions:**
- Participants are briefed on the navigation types and shown examples of the grid and stimuli.

**Practice Trials:**
- Participants complete 4 practice trials (2 egocentric, 2 allocentric) before the main experiment.
- Each practice trial includes detailed feedback on accuracy.

**Experimental Structure:**
- The experiment consists of 9 total blocks:
  - 2 blocks of egocentric navigation with easy difficulty
  - 2 blocks of egocentric navigation with hard difficulty
  - 2 blocks of allocentric navigation with easy difficulty
  - 2 blocks of allocentric navigation with hard difficulty
  - 1 block of control condition
- Each block contains 15 trials, for a total of 135 trials.
- Block order follows a Latin square design with round-robin distribution to minimize order effects.
- Control blocks are inserted at approximately equidistant positions throughout the experiment.

**Trial Timing:**
- Each trial has a 0.5-second baseline period (fixation cross)
- Maximum response time is 3.0 seconds
- Feedback is displayed for 0.5 seconds
- Inter-trial interval varies between 0.5-0.8 seconds
- A 5-second countdown appears between blocks

**Counterbalancing:**
- Four counterbalancing conditions (1-4) determine the ordering of the main experimental conditions.
- Each participant is assigned to one of these counterbalancing conditions.

## Response Criteria
The correct response is based solely on the first move the participant would make to reach the target.

**Example:**
- If the starting position is at (2,2) and the target is at (2,4), the correct first move is "move up."
- If obstacles block the direct path, the correct first move accounts for the detour.

## Measurements
**Behavioral Data:**
- Response Time (RT): Time taken to decide the first move.
- Accuracy: Whether the first move matches the correct direction.
- Error Types: Categorized by navigation type and difficulty.

**Neural Data:**
- EEG Recordings: Brain activity is recorded to analyze:
  - Mu Rhythms: Motor cortex activity (8-13 Hz) during movement planning.
  - Beta Band: Motor activity (13-30 Hz).
  - Frontal Coherence: Connectivity between frontal regions during decision-making.
  - Event-Related Potentials (ERPs): Time-locked EEG responses to specific events.

**EEG Triggers:**
- Event markers are sent to the EEG system for synchronization, including:
  - Experiment/block/trial start and end markers
  - Stimulus onset/offset
  - Fixation onset (baseline period)
  - Response and response type
  - Accuracy (correct/incorrect)
  - Feedback onset/offset
  - Navigation condition and difficulty markers

## Data Analysis
**Behavioral Analysis:**
- Compare accuracy and RT across navigation types and difficulty levels.
- Analyze RT bins (fast: <500ms, medium: 500-1000ms, slow: >1000ms) by condition.
- Examine error patterns across conditions.

**EEG Analysis:**
- Analyze mu rhythms and frontal coherence for differences between navigation types.
- Compare neural activity during correct vs. incorrect trials.
- Examine ERP components related to decision-making and feedback.
- Investigate timing differences in neural processing between egocentric and allocentric navigation.

## Applications
This experiment provides insights into:
- Cognitive strategies in spatial navigation.
- Neural correlates of egocentric and allocentric navigation.
- The impact of task difficulty on performance and brain activity.
