import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from datetime import timedelta
import matplotlib.dates as mdates

annotation_text_size = 11

class BraceAnnotation:
    def __init__(self, ax, text, x_date, y_pos, width_days, leg_height, head_height, line_width, text_size=annotation_text_size):
        self.ax = ax
        self.text = text
        self.x_date = x_date    
        self.y_pos = y_pos
        self.width = width_days
        self.leg_height = leg_height
        self.head_height = head_height
        self.line_width = line_width
        self.text_size = text_size
        
        # Convert date to matplotlib number for calculations
        x = mdates.date2num(x_date)
        width = mdates.date2num(x_date + timedelta(days=width_days)) - mdates.date2num(x_date - timedelta(days=width_days))

        # Define the bracket vertices
        verts = [
            (x - width/2, y_pos),           # Left end
            (x - width/2, y_pos + leg_height),  # Left top
            (x + width/2, y_pos + leg_height),  # Right top
            (x + width/2, y_pos),           # Right end
            (x, y_pos + leg_height),          # Center start (middle of horizontal line)
            (x, y_pos + leg_height + head_height)    # Center end (to top)
        ]
        
        # Define the path codes
        codes = [
            Path.MOVETO,      # Start at left end
            Path.LINETO,      # Draw to left top
            Path.LINETO,      # Draw to right top
            Path.LINETO,      # Draw to right end
            Path.MOVETO,      # Move to center bottom (without drawing)
            Path.LINETO       # Draw center line
        ]
        
        # Create and add the path
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor='black', lw=line_width)
        ax.add_patch(patch)
        
        # Add the text
        ax.text(x, y_pos + leg_height + head_height, text,
                horizontalalignment='center',
                verticalalignment='bottom',
                size=text_size)