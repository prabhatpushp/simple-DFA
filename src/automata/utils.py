import random
from typing import Set, List, Dict
import numpy as np
from .dfa import DFA

def create_random_dfa(num_states: int, alphabet: Set[str], num_final_states: int = 1, num_dead_states: int = 0) -> DFA:
    """Create a randomly generated DFA.
    
    Args:
        num_states: Number of states in the DFA
        alphabet: Set of input symbols
        num_final_states: Number of final states (default: 1)
        num_dead_states: Number of dead states (default: 0)
        
    Returns:
        DFA: A randomly generated DFA
        
    Raises:
        ValueError: If num_final_states + num_dead_states > num_states
    """
    if num_final_states + num_dead_states > num_states:
        raise ValueError("Sum of final and dead states cannot exceed total states")
    
    # Create DFA with specified states and alphabet
    dfa = DFA(num_states=num_states, alphabet=alphabet)
    
    # Get available states for final and dead state assignment
    available_states = set(range(num_states))
    
    # Randomly select dead states first
    if num_dead_states > 0:
        dead_states = set(random.sample(list(available_states), num_dead_states))
        available_states -= dead_states
        
        # For dead states, all transitions go back to the same state
        for state in dead_states:
            for symbol in alphabet:
                dfa.add_transition(state, symbol, state)
            dfa.states[state].is_dead = True
    
    # Randomly select final states from remaining available states
    final_states = set(random.sample(list(available_states), num_final_states))
    dfa.set_final_states(final_states)
    
    # Generate random transitions for non-dead states
    non_dead_states = set(range(num_states)) - set(state for state in range(num_states) if dfa.states[state].is_dead)
    for state in non_dead_states:
        for symbol in alphabet:
            if not dfa.has_transition(state, symbol):
                to_state = random.randint(0, num_states - 1)
                dfa.add_transition(state, symbol, to_state)
    
    return dfa

def generate_random_strings(alphabet: Set[str], num_strings: int, max_length: int) -> List[str]:
    """Generate random test strings using the given alphabet.
    
    Args:
        alphabet: Set of symbols to use
        num_strings: Number of strings to generate
        max_length: Maximum length of each string
        
    Returns:
        List[str]: List of randomly generated strings
    """
    alphabet_list = list(alphabet)
    strings = []
    
    for _ in range(num_strings):
        # Random length between 1 and max_length
        length = random.randint(1, max_length)
        # Generate random string
        test_str = ''.join(random.choice(alphabet_list) for _ in range(length))
        strings.append(test_str)
    
    return strings

def create_box_border(width: int, style: str = 'single') -> Dict[str, str]:
    """Create box drawing characters for borders.
    
    Args:
        width: Width of the box
        style: Border style ('single' or 'double')
        
    Returns:
        Dict[str, str]: Dictionary of border characters
    """
    if style == 'double':
        return {
            'top_left': '╔', 'top_right': '╗', 'bottom_left': '╚', 'bottom_right': '╝',
            'horizontal': '═', 'vertical': '║', 'left_t': '╠', 'right_t': '╣',
            'top_t': '╦', 'bottom_t': '╩', 'cross': '╬'
        }
    return {
        'top_left': '┌', 'top_right': '┐', 'bottom_left': '└', 'bottom_right': '┘',
        'horizontal': '─', 'vertical': '│', 'left_t': '├', 'right_t': '┤',
        'top_t': '┬', 'bottom_t': '┴', 'cross': '┼'
    }

def pretty_print_dfa(dfa: DFA) -> str:
    """Create a pretty string representation of the DFA.
    
    Args:
        dfa: The DFA to pretty print
        
    Returns:
        str: A formatted string representation of the DFA
    """
    # Terminal colors
    HEADER = '\033[95m'  # Purple
    BLUE = '\033[94m'    # Blue
    GREEN = '\033[92m'   # Green
    RED = '\033[91m'     # Red
    ENDC = '\033[0m'     # Reset
    BOLD = '\033[1m'     # Bold
    
    # Get box drawing characters
    box = create_box_border(width=50, style='double')
    
    # Calculate widths
    state_width = 6
    symbol_width = 5
    total_width = state_width + (symbol_width + 1) * len(dfa.alphabet) + 1
    
    output = []
    
    # Create top border
    output.append(f"{box['top_left']}{box['horizontal'] * total_width}{box['top_right']}")
    
    # Header
    title = f"{HEADER}{BOLD}DFA Information{ENDC}"
    padding = (total_width - len("DFA Information")) // 2
    output.append(f"{box['vertical']}{' ' * padding}{title}{' ' * (total_width - padding - len('DFA Information'))}{box['vertical']}")
    
    # Separator
    output.append(f"{box['left_t']}{box['horizontal'] * total_width}{box['right_t']}")
    
    # Basic information
    states_info = f"{BOLD}States:{ENDC} {dfa.num_states}"
    alphabet_info = f"{BOLD}Alphabet:{ENDC} {', '.join(sorted(dfa.alphabet))}"
    output.append(f"{box['vertical']} {states_info}{' ' * (total_width - len(states_info) - 1)}{box['vertical']}")
    output.append(f"{box['vertical']} {alphabet_info}{' ' * (total_width - len(alphabet_info) - 1)}{box['vertical']}")
    
    # Special states information
    output.append(f"{box['left_t']}{box['horizontal'] * total_width}{box['right_t']}")
    output.append(f"{box['vertical']} {BOLD}Special States:{ENDC}{' ' * (total_width - len('Special States:') - 1)}{box['vertical']}")
    
    # Initial state
    initial_state = 0  # DFA always starts at state 0
    initial_info = f" • {BLUE}Initial State:{ENDC} {initial_state}"
    output.append(f"{box['vertical']}{initial_info}{' ' * (total_width - len(initial_info))}{box['vertical']}")
    
    # Final states
    final_states = [state_id for state_id, state in dfa.states.items() if state.is_final]
    final_info = f" • {GREEN}Final States:{ENDC} {', '.join(map(str, final_states))}"
    output.append(f"{box['vertical']}{final_info}{' ' * (total_width - len(final_info))}{box['vertical']}")
    
    # Dead states
    dead_states = [state_id for state_id, state in dfa.states.items() if state.is_dead]
    dead_info = f" • {RED}Dead States:{ENDC} {', '.join(map(str, dead_states))}"
    output.append(f"{box['vertical']}{dead_info}{' ' * (total_width - len(dead_info))}{box['vertical']}")
    
    # Transition table
    output.append(f"{box['left_t']}{box['horizontal'] * total_width}{box['right_t']}")
    output.append(f"{box['vertical']} {BOLD}Transition Table:{ENDC}{' ' * (total_width - len('Transition Table:') - 1)}{box['vertical']}")
    
    # Header row for transition table
    header_row = "State".ljust(state_width)
    for symbol in sorted(dfa.alphabet):
        header_row += f"{box['vertical']}{str(symbol).center(symbol_width)}"
    output.append(f"{box['left_t']}{box['horizontal'] * state_width}{box['top_t']}{(box['horizontal'] * symbol_width + box['top_t']) * (len(dfa.alphabet) - 1)}{box['horizontal'] * symbol_width}{box['right_t']}")
    output.append(f"{box['vertical']}{header_row}{box['vertical']}")
    
    # Separator for transition table
    output.append(f"{box['left_t']}{box['horizontal'] * state_width}{box['cross']}{(box['horizontal'] * symbol_width + box['cross']) * (len(dfa.alphabet) - 1)}{box['horizontal'] * symbol_width}{box['right_t']}")
    
    # State rows
    for state in range(dfa.num_states):
        row = str(state).ljust(state_width)
        for symbol in sorted(dfa.alphabet):
            next_state = dfa.get_next_state(state, symbol)
            if next_state is not None:
                row += f"{box['vertical']}{str(next_state).center(symbol_width)}"
            else:
                row += f"{box['vertical']}{'-'.center(symbol_width)}"
        output.append(f"{box['vertical']}{row}{box['vertical']}")
    
    # Bottom border
    output.append(f"{box['bottom_left']}{box['horizontal'] * total_width}{box['bottom_right']}")
    
    return "\n".join(output)

def show_input_transitions(dfa: DFA, input_string: str) -> str:
    """Create a visual representation of state transitions for a given input string.
    
    Args:
        dfa: The DFA to analyze
        input_string: Input string to process
        
    Returns:
        str: A formatted string showing the step-by-step transitions
        
    Raises:
        InvalidSymbolError: If input contains invalid symbols
    """
    # Terminal colors
    HEADER = '\033[95m'  # Purple
    BLUE = '\033[94m'    # Blue
    GREEN = '\033[92m'   # Green
    RED = '\033[91m'     # Red
    ENDC = '\033[0m'     # Reset
    BOLD = '\033[1m'     # Bold
    
    # Get box drawing characters
    box = create_box_border(width=50, style='single')
    
    output = []
    current_state = 0  # Start from initial state
    
    # Create header
    output.append(f"{box['top_left']}{box['horizontal'] * 60}{box['top_right']}")
    title = f"{HEADER}{BOLD}DFA Transition Path{ENDC}"
    output.append(f"{box['vertical']} {title}{' ' * (59 - len('DFA Transition Path'))}{box['vertical']}")
    output.append(f"{box['left_t']}{box['horizontal'] * 60}{box['right_t']}")
    
    # Input string information
    output.append(f"{box['vertical']} {BOLD}Input String:{ENDC} {input_string}{' ' * (47 - len(input_string))}{box['vertical']}")
    output.append(f"{box['left_t']}{box['horizontal'] * 60}{box['right_t']}")
    
    # Show transitions
    output.append(f"{box['vertical']} {BOLD}Transitions:{ENDC}{' ' * 49}{box['vertical']}")
    
    # Initial state
    state_info = dfa.states[current_state]
    state_type = []
    if state_info.is_initial:
        state_type.append(f"{BLUE}Initial{ENDC}")
    if state_info.is_final:
        state_type.append(f"{GREEN}Final{ENDC}")
    if state_info.is_dead:
        state_type.append(f"{RED}Dead{ENDC}")
    
    state_desc = f"Start: State {current_state}"
    if state_type:
        state_desc += f" ({', '.join(state_type)})"
    output.append(f"{box['vertical']} {state_desc}{' ' * (59 - len(state_desc))}{box['vertical']}")
    
    # If starting state is dead, immediately reject
    if state_info.is_dead:
        output.append(f"{box['vertical']} {RED}Starting in dead state - immediate rejection{ENDC}{' ' * 20}{box['vertical']}")
        output.append(f"{box['left_t']}{box['horizontal'] * 60}{box['right_t']}")
        result_line = f"Final Result: String '{input_string}' is {RED}REJECTED{ENDC} (Dead State)"
        output.append(f"{box['vertical']} {result_line}{' ' * (59 - len(result_line))}{box['vertical']}")
        output.append(f"{box['bottom_left']}{box['horizontal'] * 60}{box['bottom_right']}")
        return "\n".join(output)
    
    # Process each symbol
    remaining_input = input_string
    for i, symbol in enumerate(input_string, 1):
        if symbol not in dfa.alphabet:
            raise InvalidSymbolError(f"Invalid symbol in input: {symbol}")
            
        next_state = dfa.get_next_state(current_state, symbol)
        if next_state is None:
            output.append(f"{box['vertical']} {RED}No transition found for symbol '{symbol}' from state {current_state}{ENDC}")
            break
            
        # Show transition
        transition = f"Step {i}: {current_state} --({symbol})--> {next_state}"
        output.append(f"{box['vertical']} {transition}{' ' * (59 - len(transition))}{box['vertical']}")
        
        current_state = next_state
        state_info = dfa.states[current_state]
        remaining_input = input_string[i:]
        
        # Show state properties if any
        state_type = []
        if state_info.is_final:
            state_type.append(f"{GREEN}Final{ENDC}")
        if state_info.is_dead:
            state_type.append(f"{RED}Dead{ENDC}")
            
        if state_type:
            state_desc = f"      State {current_state} is {', '.join(state_type)}"
            output.append(f"{box['vertical']} {state_desc}{' ' * (59 - len(state_desc))}{box['vertical']}")
        
        # If we reach a dead state, stop processing and show remaining input
        if state_info.is_dead:
            if remaining_input:
                output.append(f"{box['vertical']} {RED}Dead state reached - remaining input '{remaining_input}' ignored{ENDC}")
            else:
                output.append(f"{box['vertical']} {RED}Dead state reached at end of input{ENDC}")
            break
    
    # Show final result
    output.append(f"{box['left_t']}{box['horizontal'] * 60}{box['right_t']}")
    
    # Determine result based on final state and whether we hit a dead state
    if dfa.states[current_state].is_dead:
        result = f"{RED}REJECTED{ENDC} (Dead State)"
    else:
        is_accepted = dfa.accepts(input_string)
        result = f"{GREEN}ACCEPTED{ENDC}" if is_accepted else f"{RED}REJECTED{ENDC}"
    
    result_line = f"Final Result: String '{input_string}' is {result}"
    output.append(f"{box['vertical']} {result_line}{' ' * (59 - len(result_line))}{box['vertical']}")
    
    # Bottom border
    output.append(f"{box['bottom_left']}{box['horizontal'] * 60}{box['bottom_right']}")
    
    return "\n".join(output)