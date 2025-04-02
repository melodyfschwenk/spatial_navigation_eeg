import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, PathPatch
import matplotlib.path as mpath
import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
ax.set_xlim(0, 6)
ax.set_ylim(0, 8)
ax.axis('off')

# Add title
ax.text(3, 7.8, "Spatial Navigation EEG Task", fontsize=14, fontweight='bold', ha='center')

# Function to draw a screen
def draw_screen(y_pos, label, timing, draw_func=None):
    # Draw screen box
    screen_width, screen_height = 2, 2
    screen_x = 1.5
    screen_y = y_pos
    
    # Draw a connecting line to the next screen (if not the last one)
    if y_pos > 1:
        ax.plot([screen_x + screen_width/2, screen_x + screen_width/2], 
                [screen_y, screen_y - 0.5], 'k-', lw=1.5)
    
    # Draw the screen
    screen = Rectangle((screen_x, screen_y), screen_width, screen_height, 
                     fill=True, color='black', edgecolor='k', lw=1.5)
    ax.add_patch(screen)
    
    # Add timing info
    ax.text(screen_x + screen_width + 0.3, screen_y + screen_height/2, timing, 
           fontsize=12, va='center', ha='left')
    
    # Add label
    ax.text(screen_x - 0.3, screen_y + screen_height/2, label, 
           fontsize=12, va='center', ha='right', fontweight='bold')
    
    # Call the drawing function if provided
    if draw_func:
        draw_func(screen_x, screen_y, screen_width, screen_height)
    
    return screen_y

# Draw fixation cross
def draw_fixation(x, y, width, height):
    ax.plot([x + width/2 - 0.3, x + width/2 + 0.3], [y + height/2, y + height/2], 'w-', lw=3)
    ax.plot([x + width/2, x + width/2], [y + height/2 - 0.3, y + height/2 + 0.3], 'w-', lw=3)

# Draw instruction
def draw_instruction(x, y, width, height):
    ax.text(x + width/2, y + height/2, "EGOCENTRIC", ha='center', va='center', 
            color='white', fontweight='bold', fontsize=12)

# Draw grid with player and target
def draw_grid(x, y, width, height):
    # Draw grid lines
    grid_size = 5
    cell_size = min(width, height) / grid_size
    start_x = x + (width - grid_size * cell_size) / 2
    start_y = y + (height - grid_size * cell_size) / 2
    
    # Draw grid lines
    for i in range(grid_size + 1):
        # Horizontal lines
        ax.plot([start_x, start_x + grid_size * cell_size], 
               [start_y + i * cell_size, start_y + i * cell_size], 
               'w-', lw=1, alpha=0.7)
        # Vertical lines
        ax.plot([start_x + i * cell_size, start_x + i * cell_size], 
               [start_y, start_y + grid_size * cell_size], 
               'w-', lw=1, alpha=0.7)
    
    # Fill some cells to represent path (blue semi-transparent)
    path_cells = [(1, 1), (1, 2), (2, 2), (2, 3)]
    for col, row in path_cells:
        cell_x = start_x + col * cell_size
        cell_y = start_y + row * cell_size
        ax.add_patch(Rectangle((cell_x, cell_y), cell_size, cell_size,
                              fill=True, color='#4C72B0', alpha=0.5))
    
    # Add player (gray circle)
    player_x = start_x + 1.5 * cell_size
    player_y = start_y + 1.5 * cell_size
    ax.add_patch(Circle((player_x, player_y), radius=cell_size/3,
                       fill=True, color='#AAAAAA'))
    
    # Add target (red circle)
    target_x = start_x + 3.5 * cell_size
    target_y = start_y + 3.5 * cell_size
    ax.add_patch(Circle((target_x, target_y), radius=cell_size/3,
                       fill=True, color='#C23B22'))

# Draw feedback (checkmark)
def draw_feedback(x, y, width, height):
    # Add a green checkmark
    check_verts = [
        (x + width/2 - 0.3, y + height/2),
        (x + width/2, y + height/2 - 0.3),
        (x + width/2 + 0.5, y + height/2 + 0.4)
    ]
    check_codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO]
    check_path = mpath.Path(check_verts, check_codes)
    ax.add_patch(PathPatch(check_path, edgecolor='lime', linewidth=3,
                          fill=False))

# Draw ITI (blank)
def draw_iti(x, y, width, height):
    # Nothing to draw for blank screen
    pass

# Draw the trial sequence from top to bottom
y_pos = 6
y_pos = draw_screen(y_pos, "Fixation", "500-700ms", draw_fixation)
y_pos = draw_screen(y_pos - 2.5, "Instruction", "1000ms", draw_instruction)
y_pos = draw_screen(y_pos - 2.5, "Navigation", "≤5000ms", draw_grid)
y_pos = draw_screen(y_pos - 2.5, "Feedback", "500ms", draw_feedback)
y_pos = draw_screen(y_pos - 2.5, "ITI", "800-1200ms", draw_iti)

# Add a caption or note
note = """Note: Participants use arrow keys to indicate navigation direction
• Egocentric condition: Arrow keys relative to avatar orientation
• Allocentric condition: Arrow keys mapped to compass directions
• Control condition: Arrow keys follow directional cue"""

ax.text(3, 0.5, note, ha='center', va='center', fontsize=10, 
       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('EEG_Task_Schematic_Vertical.png', dpi=300, bbox_inches='tight')
plt.show()