# LLM-AutonomousSorter

A simulation of an autonomous robot that can sort objects on a grid based on natural language commands. The robot uses a zero-shot classification model to interpret commands and can sort objects based on their shape or color.

![Robot Sorting Simulation](robot_demo.png)

## Features

- Interactive grid-based environment with a robot and colorful objects
- Natural language command interpretation using a transformer-based model
- Autonomous path finding and object sorting capabilities
- Sorting areas that can be configured by shape or color
- Visual representation of the robot and sorted objects
- Real-time animation of robot movement and object manipulation

## Prerequisites

To run this project, you'll need:

- Python 3.7+
- TkInter (usually comes with Python)
- The following Python packages:
  - matplotlib
  - transformers
  - numpy
  - scipy
  - PIL (Pillow)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/LLM-AutonomousSorter.git
   cd LLM-AutonomousSorter
   ```

2. Install the required packages:
   ```
   pip install matplotlib transformers numpy scipy pillow
   ```

3. (Optional) For a custom robot appearance, add a file named `robot.png` to the project directory.

## Usage

Run the simulation:
```
python robo.py
```

### Commands

The robot understands natural language commands through zero-shot classification. Here are some example commands:

- **Movement commands**:
  - "Move forward" - Moves the robot one step in the direction it's facing
  - "Turn left" - Rotates the robot 90 degrees counterclockwise
  - "Turn right" - Rotates the robot 90 degrees clockwise

- **Sorting commands**:
  - "Sort objects by color" - Initiates autonomous sorting of objects based on their color
  - "Sort objects by shape" - Initiates autonomous sorting of objects based on their shape
  - "Pick up this object" - Picks up an object at the robot's current location
  - "Drop the object" - Drops the currently carried object

For best results, use clear and simple language that closely matches these example commands.

## How It Works

1. The simulation creates a grid with randomly placed objects and designated sorting areas
2. Objects have different shapes (circle, triangle, square) and colors (red, green, blue)
3. When a sorting command is given, the robot:
   - Finds the nearest unsorted object
   - Navigates to the object using A* pathfinding
   - Picks up the object
   - Finds the appropriate sorting area based on the object's properties
   - Navigates to the sorting area
   - Drops the object in the sorting area
   - Repeats until all objects are sorted

## Configuration

The simulation has several configurable parameters in the code:

- `SHAPES` and `COLORS` - Defines the available shapes and colors for objects
- `grid_size` in the `main()` function - Sets the size of the grid
- The number of objects is set in the `main()` function with `generate_random_objects(4, grid_size, dummy_sorting_areas)`

## Extending the Project

You can extend this project in several ways:

1. Add more shapes and colors
2. Implement more complex sorting algorithms
3. Create obstacles in the environment
4. Add multiple robots for collaborative sorting
5. Implement a more sophisticated natural language processing model

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) model for zero-shot classification
- Built with matplotlib for visualization and tkinter for the GUI
