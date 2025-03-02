import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.automata.utils import create_random_dfa, generate_random_strings, pretty_print_dfa, show_input_transitions

def main():
    """Example usage of randomly generated DFA."""
    # Configuration
    NUM_STATES = 4
    ALPHABET = {'0', '1', '4', '3'}
    NUM_FINAL_STATES = 2
    NUM_DEAD_STATES = 1
    NUM_TEST_STRINGS = 10
    MAX_STRING_LENGTH = 5
    
    # Create random DFA
    dfa = create_random_dfa(
        num_states=NUM_STATES,
        alphabet=ALPHABET,
        num_final_states=NUM_FINAL_STATES,
        num_dead_states=NUM_DEAD_STATES
    )
    
    # Print DFA information using pretty print
    print(pretty_print_dfa(dfa))
    
    # Generate and test random strings
    test_strings = generate_random_strings(
        alphabet=ALPHABET,
        num_strings=NUM_TEST_STRINGS,
        max_length=MAX_STRING_LENGTH
    )
    
    print("\nTesting random strings:")
    print("-" * 50)
    
    accepted = 0
    for test_str in test_strings:
        # Show detailed transitions for each string
        print(show_input_transitions(dfa, test_str))
        print()  # Add blank line between strings
        accepted += 1 if dfa.accepts(test_str) else 0
    
    # Print acceptance statistics
    print("\nAcceptance Statistics:")
    print("-" * 50)
    acceptance_rate = (accepted / NUM_TEST_STRINGS) * 100
    print(f"Accepted: {accepted}/{NUM_TEST_STRINGS} ({acceptance_rate:.1f}%)")

if __name__ == "__main__":
    main() 