import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from transformers import pipeline
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import matplotlib.patches as patches
import random
import numpy as np
import time
import matplotlib.image as mpimg
import os
from scipy import ndimage


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


SHAPES = ['circle', 'triangle', 'square']
COLORS = ['red', 'green', 'yellow']


try:

    robot_img = plt.imread('robot.png')

    if robot_img.shape[2] == 4:
      
        pass
    else:
     
        pass
except FileNotFoundError:
    robot_img = None
    print("Warning: robot.png not found. Using circle representation instead.")
except Exception as e:
    robot_img = None
    print(f"Warning: Could not load robot.png: {e}. Using circle representation instead.")

class GridObject:
    """Class to represent objects on the grid."""
    def __init__(self, shape, color, position):
        self.shape = shape
        self.color = color
        self.position = position  
        self.is_carried = False
        self.is_properly_sorted = False 

    def draw(self, ax):
        """Draw the object on the given axes."""
        if self.is_carried:
            return  
            
        x, y = self.position
        if self.shape == 'circle':
            patch = patches.Circle((x + 0.5, y + 0.5), radius=0.6, fc=self.color)
        elif self.shape == 'triangle':
            patch = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.6, fc=self.color)
        elif self.shape == 'square':
            patch = patches.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, fc=self.color)
        
      
        if self.is_properly_sorted:
            halo = patches.Circle((x + 0.5, y + 0.5), radius=0.9, fc='none', ec='gold', lw=4)
            ax.add_patch(halo)
            
        ax.add_patch(patch)

class SortingArea:
    """Class to represent a designated sorting area."""
    def __init__(self, category_type, category_value, position, size=(3, 3)):
        self.category_type = category_type 
        self.category_value = category_value 
        self.position = position 
        self.size = size 
        
    def contains_position(self, position):
        """Check if a position is within this sorting area."""
        x, y = position
        area_x, area_y = self.position
        area_width, area_height = self.size
        
        return (area_x <= x < area_x + area_width and 
                area_y <= y < area_y + area_height)
                
    def is_matching_object(self, obj):
        """Check if an object belongs in this sorting area."""
        if self.category_type == 'color':
            return obj.color == self.category_value
        elif self.category_type == 'shape':
            return obj.shape == self.category_value
        return False
        
    def draw(self, ax):
        """Draw the sorting area on the grid."""
        x, y = self.position
        width, height = self.size
        
        
        color = self.category_value if self.category_type == 'color' else 'lightgray'
        area_patch = patches.Rectangle(
            (x, y), width, height, 
            fc=color, ec='black', lw=4,
            alpha=0.3
        )
        ax.add_patch(area_patch)
        
        # Add a label for the area
        label_text = f"{self.category_value}"
        ax.text(x + width/2, y + height/2, label_text,
                ha='center', va='center', fontsize=18, fontweight='bold')

def generate_random_objects(num_objects, grid_size, sorting_areas):
    """Generate random objects on the grid, avoiding sorting areas for initial placement."""
    objects = []
    positions = set()
    

    forbidden_positions = {(0, 0)}  
    for area in sorting_areas:
        for x in range(area.position[0], area.position[0] + area.size[0]):
            for y in range(area.position[1], area.position[1] + area.size[1]):
                forbidden_positions.add((x, y))
    

    for shape in SHAPES:
        for color in COLORS:
          
            count = 0
            while count < 1:  
                x = random.randint(0, grid_size[0] - 1)
                y = random.randint(0, grid_size[1] - 1)
                
      
                if (x, y) not in positions and (x, y) not in forbidden_positions:
                    positions.add((x, y))
                    objects.append(GridObject(shape, color, (x, y)))
                    count += 1
    
    return objects

def create_sorting_areas(grid_size, sort_by):
    """Create the designated sorting areas based on sort criteria."""
    areas = []
    
    if sort_by == "SORT_BY_COLOR":
       
        area_width, area_height = 3, 3
        
       
        for i, color in enumerate(COLORS):
            x_start = 2 + i * (area_width + 1)
            y_start = 2
            areas.append(SortingArea('color', color, (x_start, y_start), (area_width, area_height)))
    
    elif sort_by == "SORT_BY_SHAPE":
       
        area_width, area_height = 3, 3
        
      
        for i, shape in enumerate(SHAPES):
            x_start = 2 + i * (area_width + 1)
            y_start = 2
            areas.append(SortingArea('shape', shape, (x_start, y_start), (area_width, area_height)))
    
    return areas

def draw_grid(robot_pos, robot_orientation, fig, ax, grid_objects, sorting_areas, carried_object, status_message, grid_size=(10, 10)):
    """
    Draw the grid, robot, sorting areas, and objects on the grid.
    """
    ax.clear()
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)
    ax.set_title('Robot Grid Environment', fontsize=18)
    

    for area in sorting_areas:
        area.draw(ax)
    

    for obj in grid_objects:
        obj.draw(ax)
    
    x, y = robot_pos
    

    # Get the robot color based on whether it's carrying an object
    robot_color = 'blue'
    if carried_object:
        robot_color = carried_object.color
    
    # Decide whether to use the image or fallback to a circle
    if robot_img is not None:
        # Calculate the rotation angle based on orientation
        angle = 0
        if robot_orientation == 'N':
            angle = 0
        elif robot_orientation == 'E':
            angle = 90
        elif robot_orientation == 'S':
            angle = 180
        elif robot_orientation == 'W':
            angle = 270
        
        # Calculate the extent of the image (width and height increased by 3x)
        img_extent = [x - 0.5, x + 1.5, y - 0.5, y + 1.5]
        
        # Display the robot image with rotation
        rotated_img = ndimage.rotate(robot_img, angle, reshape=False)
        ax.imshow(rotated_img, extent=img_extent, zorder=10)
        
        # If the robot is carrying an object, add an indicator matching the object's shape
        if carried_object:
            # Create an indicator with the same shape as the carried object
            if carried_object.shape == 'circle':
                indicator = patches.Circle((x + 0.5, y + 0.5), radius=0.6, 
                                         fc=carried_object.color, ec='black', zorder=11, alpha=0.7)
            elif carried_object.shape == 'triangle':
                indicator = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.6, 
                                                 fc=carried_object.color, ec='black', zorder=11, alpha=0.7)
            elif carried_object.shape == 'square':
                indicator = patches.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, 
                                            fc=carried_object.color, ec='black', zorder=11, alpha=0.7)
            ax.add_patch(indicator)
    else:
        # Fallback to circle representation for the robot
        robot = patches.Circle((x + 0.5, y + 0.5), radius=0.9, fc=robot_color, ec='black', lw=3)
        ax.add_patch(robot)
        
        # Draw arrow to indicate orientation
        dx, dy = 0, 0
        if robot_orientation == 'N':
            dx, dy = 0, 0.9
        elif robot_orientation == 'E':
            dx, dy = 0.9, 0
        elif robot_orientation == 'S':
            dx, dy = 0, -0.9
        elif robot_orientation == 'W':
            dx, dy = -0.9, 0
        
        ax.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.3, head_length=0.3, fc='white', ec='black', lw=2)
        
        # If the robot is carrying an object, represent it with the proper shape
        if carried_object:
            # Position the indicator slightly in front of the robot based on orientation
            indicator_x, indicator_y = x + 0.5, y + 0.5
            if robot_orientation == 'N':
                indicator_y += 0.4
            elif robot_orientation == 'E':
                indicator_x += 0.4
            elif robot_orientation == 'S':
                indicator_y -= 0.4
            elif robot_orientation == 'W':
                indicator_x -= 0.4
                
            # Create an indicator with the same shape as the carried object
            if carried_object.shape == 'circle':
                indicator = patches.Circle((indicator_x, indicator_y), radius=0.6, 
                                         fc=carried_object.color, ec='black', zorder=11, alpha=0.7)
            elif carried_object.shape == 'triangle':
                indicator = patches.RegularPolygon((indicator_x, indicator_y), numVertices=3, radius=0.6, 
                                                 fc=carried_object.color, ec='black', zorder=11, alpha=0.7)
            elif carried_object.shape == 'square':
                indicator = patches.Rectangle((indicator_x - 0.3, indicator_y - 0.3), 0.6, 0.6, 
                                            fc=carried_object.color, ec='black', zorder=11, alpha=0.7)
            ax.add_patch(indicator)
    
    if carried_object:
        carried_text = f"Carrying: {carried_object.color} {carried_object.shape}"
        ax.text(0.5, grid_size[1] - 0.5, carried_text, 
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7),
                fontsize=16)
    

    if status_message:
        ax.text(grid_size[0] / 2, -0.5, status_message,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='lightblue', alpha=0.7),
                fontsize=16)
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()  

def interpret_command(user_command):
    """
    Use the zero-shot classifier to convert the user's natural language command
    into one of the defined structured commands.
    """
    # For movement commands
    movement_labels = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    movement_result = classifier(user_command, movement_labels)
    
    # For sorting commands
    sorting_labels = ["SORT_BY_COLOR", "SORT_BY_SHAPE", "PICK_UP", "DROP"]
    sorting_result = classifier(user_command, sorting_labels)
    
    # Determine which category had the highest confidence
    if sorting_result["scores"][0] > movement_result["scores"][0]:
        return sorting_result["labels"][0]
    else:
        return movement_result["labels"][0]

def update_robot_state(robot_pos, robot_orientation, command, grid_size=(10, 10)):
    """Update the robot's state based on the interpreted command."""
    x, y = robot_pos
    orientations = ['N', 'E', 'S', 'W']
    idx = orientations.index(robot_orientation)

    if command == "MOVE_FORWARD":
        if robot_orientation == 'N' and y < grid_size[1] - 1:
            y += 1
        elif robot_orientation == 'E' and x < grid_size[0] - 1:
            x += 1
        elif robot_orientation == 'S' and y > 0:
            y -= 1
        elif robot_orientation == 'W' and x > 0:
            x -= 1
    elif command == "TURN_LEFT":
        robot_orientation = orientations[(idx - 1) % 4]
    elif command == "TURN_RIGHT":
        robot_orientation = orientations[(idx + 1) % 4]
    
    return (x, y), robot_orientation

def find_path(start_pos, target_pos, grid_objects, grid_size):
    """
    Find a path from start_pos to target_pos avoiding objects.
    Returns a list of positions to visit.
    """
  
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
  
    occupied = {obj.position for obj in grid_objects if obj.position != target_pos and not obj.is_carried}
    
    open_set = {start_pos}
    closed_set = set()
    
    came_from = {}
    
    g_score = {start_pos: 0}
    f_score = {start_pos: heuristic(start_pos, target_pos)}
    
    while open_set:
        current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
        
        if current == target_pos:
       
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        open_set.remove(current)
        closed_set.add(current)
        
      
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
           
            if (neighbor[0] < 0 or neighbor[0] >= grid_size[0] or
                neighbor[1] < 0 or neighbor[1] >= grid_size[1] or
                neighbor in closed_set or
                neighbor in occupied):
                continue
            
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target_pos)
    

    return [start_pos]

class AutonomousRobot:
    """Class to manage an autonomous robot that can sort objects."""
    def __init__(self, initial_pos, initial_orientation, grid_size, grid_objects, fig, ax):
        self.pos = initial_pos
        self.orientation = initial_orientation
        self.grid_size = grid_size
        self.grid_objects = grid_objects
        self.fig = fig
        self.ax = ax
        self.carried_object = None
        self.is_sorting = False
        self.sort_key_func = None
        self.sorting_areas = []
        self.current_action = ""
        self.sorting_complete = False
        self.target_object = None
        self.path = []
        self.target_orientation = None
    
    def update_display(self, status_message=""):
        """Update the display with the current state."""
        draw_grid(self.pos, self.orientation, self.fig, self.ax, 
                 self.grid_objects, self.sorting_areas, self.carried_object, 
                 status_message, self.grid_size)
        time.sleep(0.05)  # Reduced from 0.2 to 0.05 for faster animation
    
    def turn_to_orientation(self, target_orientation):
        """Turn the robot to face the target orientation."""
        if self.orientation == target_orientation:
            return True
            
        orientations = ['N', 'E', 'S', 'W']
        idx = orientations.index(self.orientation)
        target_idx = orientations.index(target_orientation)
        
       
        if (idx + 1) % 4 == target_idx:
            self.orientation = orientations[(idx + 1) % 4] 
            self.update_display(f"Turning right to face {self.orientation}")
        else:
            self.orientation = orientations[(idx - 1) % 4]  
            self.update_display(f"Turning left to face {self.orientation}")
            
        return self.orientation == target_orientation
    
    def move_along_path(self):
        """Move one step along the current path."""
        if not self.path or len(self.path) <= 1:
            return True 
            
        next_pos = self.path[1]
        dx, dy = next_pos[0] - self.pos[0], next_pos[1] - self.pos[1]
        
        
        target_orientation = self.orientation
        if dx == 1:
            target_orientation = 'E'
        elif dx == -1:
            target_orientation = 'W'
        elif dy == 1:
            target_orientation = 'N'
        elif dy == -1:
            target_orientation = 'S'
            
 
        if self.orientation != target_orientation:
            self.target_orientation = target_orientation
            return False
   
        self.pos = next_pos
        self.path.pop(0)  
        self.update_display(f"Moving to position {self.pos}")
        
        return len(self.path) <= 1
    
    def pick_up_object(self, obj):
        """Pick up the specified object."""
        if self.carried_object:
            return True
        if self.pos != obj.position:
            return False
        
    
        if obj.is_properly_sorted:
            self.update_display(f"Object is already sorted, leaving it alone")
            return True
            
        obj.is_carried = True
        self.carried_object = obj
        self.update_display(f"Picking up {obj.color} {obj.shape}")
        
        return True
    
    def drop_object(self):
        """Drop the carried object at the current position."""
        if not self.carried_object:
            return True  
        
      
        for area in self.sorting_areas:
            if area.contains_position(self.pos) and area.is_matching_object(self.carried_object):
                self.carried_object.is_properly_sorted = True
                self.update_display(f"Object is now properly sorted")
                break
            
        self.carried_object.position = self.pos
        self.carried_object.is_carried = False
        self.update_display(f"Dropping {self.carried_object.color} {self.carried_object.shape}")
        self.carried_object = None
        
        return True
    
    def find_nearest_unsorted_object(self):
        """Find the nearest object that isn't carried and isn't properly sorted."""
        available_objects = [obj for obj in self.grid_objects 
                             if not obj.is_carried and 
                             not obj.is_properly_sorted and 
                             not (self.carried_object and obj == self.carried_object)]
        
        if not available_objects:
            return None
            
  
        nearest_obj = None
        shortest_path_length = float('inf')
        
        for obj in available_objects:
            path = find_path(self.pos, obj.position, 
                             [o for o in self.grid_objects if o != obj and not o.is_carried], 
                             self.grid_size)
            if len(path) < shortest_path_length:
                shortest_path_length = len(path)
                nearest_obj = obj
                
        return nearest_obj
    
    def find_position_in_sorting_area(self, area):
        """Find an available position in the sorting area."""

        occupied_positions = {obj.position for obj in self.grid_objects 
                             if not obj.is_carried and obj != self.carried_object}
        

        for x in range(area.position[0], area.position[0] + area.size[0]):
            for y in range(area.position[1], area.position[1] + area.size[1]):
                if (x, y) not in occupied_positions:
                    return (x, y)

        center_x = area.position[0] + area.size[0] // 2
        center_y = area.position[1] + area.size[1] // 2
        return (center_x, center_y)
    
    def initialize_sorting(self, sort_by):
        """Initialize the sorting process."""
        self.is_sorting = True
        self.sorting_complete = False
        self.current_action = "FINDING_OBJECT"

        self.sorting_areas = create_sorting_areas(self.grid_size, sort_by)
        
        if sort_by == "SORT_BY_COLOR":
            self.sort_key_func = lambda obj: obj.color
            self.update_display(f"Starting to sort by color")
            
        elif sort_by == "SORT_BY_SHAPE":
            self.sort_key_func = lambda obj: obj.shape
            self.update_display(f"Starting to sort by shape")
    
    def perform_sorting_step(self):
        """Perform one step of the sorting process."""
        if not self.is_sorting or self.sorting_complete:
            return
            
        if self.current_action == "FINDING_OBJECT":
            if self.carried_object:
       
                object_key = self.sort_key_func(self.carried_object)
                
       
                target_area = None
                for area in self.sorting_areas:
                    if area.is_matching_object(self.carried_object):
                        target_area = area
                        break
                
                if target_area:
             
                    target_position = self.find_position_in_sorting_area(target_area)
                    
    
                    self.path = find_path(self.pos, target_position, 
                                         [obj for obj in self.grid_objects if not obj.is_carried], 
                                         self.grid_size)
                    self.current_action = "MOVING_TO_TARGET"
                    self.update_display(f"Moving to drop off point in {object_key} area")
            else:
          
                nearest_obj = self.find_nearest_unsorted_object()
                
                if nearest_obj:
                    self.target_object = nearest_obj
                    self.path = find_path(self.pos, nearest_obj.position, 
                                         [obj for obj in self.grid_objects if obj != nearest_obj and not obj.is_carried], 
                                         self.grid_size)
                    self.current_action = "MOVING_TO_OBJECT"
                    self.update_display(f"Moving to pick up {nearest_obj.color} {nearest_obj.shape}")
                else:
         
                    self.current_action = ""
                    self.sorting_complete = True
                    self.is_sorting = False
                    self.update_display("Sorting complete!")
        
        elif self.current_action == "MOVING_TO_OBJECT":
            if self.pos == self.target_object.position:
      
                self.pick_up_object(self.target_object)
                self.current_action = "FINDING_OBJECT"
            else:
     
                if self.target_orientation and self.orientation != self.target_orientation:
                    self.turn_to_orientation(self.target_orientation)
                    self.target_orientation = None
                else:
                    self.move_along_path()
                    
        elif self.current_action == "MOVING_TO_TARGET":
            if len(self.path) <= 1: 
                # Drop the object
                self.drop_object()
                self.current_action = "FINDING_OBJECT"
            else:
               
                if self.target_orientation and self.orientation != self.target_orientation:
                    self.turn_to_orientation(self.target_orientation)
                    self.target_orientation = None
                else:
                    self.move_along_path()
    
    def start_autonomous_task(self, task):
        """Start an autonomous task based on the user command."""
        command = interpret_command(task)
        
        if command in ["SORT_BY_COLOR", "SORT_BY_SHAPE"]:
            self.initialize_sorting(command)
            

            while self.is_sorting and not self.sorting_complete:
                self.perform_sorting_step()
                self.fig.canvas.draw_idle() 
                time.sleep(0.02)  # Reduced from 0.1 to 0.02 for faster animation
                
            return "Sorting complete!"
        else:
            return f"Command '{command}' not supported for autonomous operation."

def main():
  
    plt.ioff()
    
    
    root = tk.Tk()
    root.title('Autonomous Robot Controller')
    root.configure(bg='#f0f0f0')


    robot_pos = (0, 0)
    robot_orientation = 'N'
    grid_size = (15, 15)
    
    
    dummy_sorting_areas = create_sorting_areas(grid_size, "SORT_BY_COLOR")
    

    grid_objects = generate_random_objects(9, grid_size, dummy_sorting_areas) 
   
    fig = plt.figure(figsize=(10, 10))  
    ax = fig.add_subplot(111)
    
  
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
  
    robot = AutonomousRobot(robot_pos, robot_orientation, grid_size, grid_objects, fig, ax)
    robot.update_display("Waiting for command...")
    
 
    input_frame = tk.Frame(root, bg='#e0e0e0', pady=10)
    input_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
  
    instruction_label = tk.Label(input_frame, text="Enter a command for the robot:", 
                              font=('Arial', 12), bg='#e0e0e0')
    instruction_label.pack(side=tk.TOP, pady=(10, 5))
    

    command_entry = tk.Entry(input_frame, font=('Arial', 12), width=40)
    command_entry.pack(side=tk.LEFT, padx=10, pady=10)
    command_entry.focus_set()  

    def process_command():
        command_text = command_entry.get()
        if command_text.strip():  
            result_label.config(text="Processing command...")
            command_entry.delete(0, tk.END) 
            
 
            def run_task():
                result = robot.start_autonomous_task(command_text)
                result_label.config(text=result)
            

            root.after(100, run_task)
    

    submit_button = tk.Button(input_frame, text='Send Command', command=process_command,
                           bg='#4CAF50', fg='white', padx=20, pady=5, font=('Arial', 12, 'bold'))
    submit_button.pack(side=tk.LEFT, padx=10)
    

    command_entry.bind('<Return>', lambda event: process_command())
    

    result_label = tk.Label(input_frame, text="Enter a command above", 
                         font=('Arial', 10), bg='#e0e0e0')
    result_label.pack(side=tk.LEFT, padx=10)
    

    quit_frame = tk.Frame(root, bg='#e0e0e0', pady=5)
    quit_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    quit_button = tk.Button(quit_frame, text='Quit', command=root.quit, bg='#f44336', fg='white', 
                          padx=10, pady=5, font=('Arial', 10, 'bold'))
    quit_button.pack(side=tk.RIGHT, padx=10)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
