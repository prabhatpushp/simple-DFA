# Deterministic Finite Automaton (DFA)

## Introduction
This project provides a concrete implementation of a Deterministic Finite Automaton (DFA). The DFA is designed to manage states, handle transitions, and check string acceptance efficiently using numpy arrays.

## Features
- **State Management**: Easily add and manage states within the DFA.
- **Transition Handling**: Define transitions between states based on input symbols.
- **String Acceptance Checking**: Determine if a given input string is accepted by the DFA.
- **Efficient Implementation**: Utilizes numpy arrays for fast state transitions.

## Tech Stack
- **Python**: The implementation is written in Python.
- **Numpy**: Used for efficient array operations and state management.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/prabhatpushp/simple-DFA
   cd simple DFA
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install numpy
   ```

## Usage
To use the DFA implementation, you can create an instance of the `DFA` class and define states and transitions. Here is an example:
```python
from src.automata.dfa import DFA

# Create a DFA instance
my_dfa = DFA()

# Add states and transitions
my_dfa.add_transition(0, 'a', 1)
my_dfa.add_transition(1, 'b', 0)

# Set final states
my_dfa.set_final_states({1})

# Check if a string is accepted
result = my_dfa.accepts('ab')
print(f"String accepted: {result}")
```

## Development
- The main implementation is in `src/automata/dfa.py`. You can modify or extend the functionality as needed.
- Ensure to test your changes thoroughly to maintain the integrity of the DFA.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes and create a pull request.

## License
This project is licensed under the MIT License. Feel free to use this project for personal or commercial purposes. 
