"""
Automata package for working with Deterministic Finite Automata (DFA).

This package provides a clean, type-safe implementation of DFAs with proper
error handling and state management.
"""

from .base import (
    AbstractDFA,
    State,
    AutomataError,
    InvalidStateError,
    InvalidSymbolError,
)
from .dfa import DFA
from .genetic import DFAGeneticAlgorithm

__all__ = [
    'AbstractDFA',
    'DFA',
    'State',
    'AutomataError',
    'InvalidStateError',
    'InvalidSymbolError',
    'DFAGeneticAlgorithm',
] 