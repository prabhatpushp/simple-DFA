from typing import Set, Dict, Optional
from .base import AbstractDFA, InvalidStateError, InvalidSymbolError, State

class DFA(AbstractDFA):
    """Concrete implementation of a Deterministic Finite Automaton (DFA).
    
    This class provides a complete implementation of a DFA with proper state
    management, transition handling, and string acceptance checking using
    numpy arrays for efficient state transitions.
    """
    
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
        # Validate states and symbol
        if from_state not in self.states:
            raise InvalidStateError(f"Invalid source state: {from_state}")
        if to_state not in self.states:
            raise InvalidStateError(f"Invalid destination state: {to_state}")
        if symbol not in self.alphabet:
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")
        
        # Add the transition to the matrix
        symbol_idx = self.symbol_to_index[symbol]
        self.transition_matrix[from_state, symbol_idx] = to_state
    
    def has_transition(self, from_state: int, symbol: str) -> bool:
        """Check if a transition exists for the given state and symbol.
        
        Args:
            from_state: Source state ID
            symbol: Input symbol
            
        Returns:
            bool: True if transition exists, False otherwise
            
        Raises:
            InvalidStateError: If state is invalid
            InvalidSymbolError: If symbol is not in alphabet
        """
        if from_state not in self.states:
            raise InvalidStateError(f"Invalid state: {from_state}")
        if symbol not in self.alphabet:
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")
            
        symbol_idx = self.symbol_to_index[symbol]
        return self.transition_matrix[from_state, symbol_idx] != -1
    
    def get_next_state(self, from_state: int, symbol: str) -> Optional[int]:
        """Get the next state for the given state and symbol.
        
        Args:
            from_state: Source state ID
            symbol: Input symbol
            
        Returns:
            Optional[int]: Next state ID if transition exists, None otherwise
            
        Raises:
            InvalidStateError: If state is invalid
            InvalidSymbolError: If symbol is not in alphabet
        """
        if from_state not in self.states:
            raise InvalidStateError(f"Invalid state: {from_state}")
        if symbol not in self.alphabet:
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")
            
        symbol_idx = self.symbol_to_index[symbol]
        next_state = self.transition_matrix[from_state, symbol_idx]
        return next_state if next_state != -1 else None
    
    def set_final_states(self, states: Set[int]) -> None:
        """Set the final (accepting) states of the DFA.
        
        Args:
            states: Set of state IDs to mark as final
            
        Raises:
            InvalidStateError: If any state ID is invalid
        """
        for state_id in states:
            if state_id not in self.states:
                raise InvalidStateError(f"Invalid state ID: {state_id}")
            self.states[state_id].is_final = True
    
    def accepts(self, input_string: str) -> bool:
        """Check if the DFA accepts the given input string.
        
        Args:
            input_string: String to check for acceptance
            
        Returns:
            bool: True if the string is accepted, False otherwise
            
        Raises:
            InvalidSymbolError: If input contains invalid symbols
        """
        current_state = 0  # Start from initial state
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise InvalidSymbolError(f"Invalid symbol in input: {symbol}")
                
            symbol_idx = self.symbol_to_index[symbol]
            next_state = self.transition_matrix[current_state, symbol_idx]
            
            # Check if there's no valid transition
            if next_state == -1:
                return False
                
            current_state = next_state
        
        # Check if final state is reached
        return self.states[current_state].is_final
    
    def get_current_state_transitions(self, state_id: int) -> Dict[str, int]:
        """Get all transitions from a given state.
        
        Args:
            state_id: ID of the state to get transitions for
            
        Returns:
            Dict[str, int]: Dictionary mapping symbols to destination states
            
        Raises:
            InvalidStateError: If state_id is invalid
        """
        if state_id not in self.states:
            raise InvalidStateError(f"Invalid state ID: {state_id}")
            
        transitions = {}
        for symbol, idx in self.symbol_to_index.items():
            next_state = self.transition_matrix[state_id, idx]
            if next_state != -1:  # Only include valid transitions
                transitions[symbol] = next_state
                
        return transitions 