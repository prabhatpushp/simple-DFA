from abc import ABC, abstractmethod
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

class AutomataError(Exception):
    """Base exception class for all automata-related errors."""
    pass

class InvalidStateError(AutomataError):
    """Raised when an invalid state is referenced."""
    pass

class InvalidSymbolError(AutomataError):
    """Raised when an invalid symbol is used."""
    pass

@dataclass
class State:
    """Represents a state in the automaton."""
    id: int
    is_final: bool = False
    is_initial: bool = False
    is_dead: bool = False

class AbstractDFA(ABC):
    """Abstract base class for Deterministic Finite Automata (DFA).
    
    This class defines the interface for DFA implementations with proper type hints
    and abstract methods that must be implemented by concrete classes.
    """
    
    def __init__(self, num_states: int, alphabet: Set[str], initial_state: int = 0):
        """Initialize the DFA.
        
        Args:
            num_states: Total number of states in the automaton
            alphabet: Set of input symbols
            initial_state: ID of the initial state (default: 0)
            
        Raises:
            ValueError: If num_states <= 0 or alphabet is empty
        """
        if num_states <= 0:
            raise ValueError("Number of states must be positive")
        if not alphabet:
            raise ValueError("Alphabet cannot be empty")
            
        self.num_states = num_states
        self.alphabet = sorted(list(alphabet))  # Convert to sorted list for consistent indexing
        self.alphabet_size = len(self.alphabet)
        self.states: Dict[int, State] = {
            i: State(id=i, is_initial=(i == initial_state))
            for i in range(num_states)
        }
        
        # Initialize transition matrix with -1 (indicating no transition)
        # Shape: [num_states x alphabet_size]
        self.transition_matrix: npt.NDArray[np.int_] = np.full((num_states, self.alphabet_size), -1, dtype=np.int_)
        
        # Create symbol to index mapping
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(self.alphabet)}
        
    @abstractmethod
    def add_transition(self, from_state: int, symbol: str, to_state: int) -> None:
        """Add a transition from one state to another on a given symbol.
        
        Args:
            from_state: Source state ID
            symbol: Input symbol
            to_state: Destination state ID
            
        Raises:
            InvalidStateError: If either state is invalid
            InvalidSymbolError: If symbol is not in alphabet
        """
        pass
    
    @abstractmethod
    def set_final_states(self, states: Set[int]) -> None:
        """Set the final (accepting) states of the DFA.
        
        Args:
            states: Set of state IDs to mark as final
            
        Raises:
            InvalidStateError: If any state ID is invalid
        """
        pass
    
    @abstractmethod
    def accepts(self, input_string: str) -> bool:
        """Check if the DFA accepts the given input string.
        
        Args:
            input_string: String to check for acceptance
            
        Returns:
            bool: True if the string is accepted, False otherwise
            
        Raises:
            InvalidSymbolError: If input contains invalid symbols
        """
        pass
    
    def get_state(self, state_id: int) -> State:
        """Get the State object for a given state ID.
        
        Args:
            state_id: ID of the state to retrieve
            
        Returns:
            State: The State object
            
        Raises:
            InvalidStateError: If state_id is invalid
        """
        if state_id not in self.states:
            raise InvalidStateError(f"Invalid state ID: {state_id}")
        return self.states[state_id]
    
    def __str__(self) -> str:
        """Return a string representation of the DFA."""
        return (f"DFA with {self.num_states} states\n"
                f"Alphabet: {self.alphabet}\n"
                f"Final states: {[s.id for s in self.states.values() if s.is_final]}\n"
                f"Transition Matrix:\n{self.transition_matrix}")
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the DFA."""
        return (f"DFA(num_states={self.num_states}, "
                f"alphabet={self.alphabet}, "
                f"transition_matrix=\n{self.transition_matrix})") 