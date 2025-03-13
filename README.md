# LLM-AutonomousSorter

A simulation of an autonomous robot that can sort objects on a grid based on natural language commands. The robot uses a zero-shot classification model to interpret commands and can sort objects based on their shape or color.


https://github.com/user-attachments/assets/4170b81b-5fb1-4df2-9a26-dae74459b497


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


## Extending the Project

You can extend this project in several ways:

1. Add more shapes and colors
2. Implement more complex sorting algorithms
3. Create obstacles in the environment
4. Add multiple robots for collaborative sorting

## Acknowledgments

- This project uses the [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) model for zero-shot classification
- Built with matplotlib for visualization and tkinter for the GUI
