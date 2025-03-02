from typing import List, Set, Tuple, Optional
import random
import numpy as np
from .dfa import DFA
from .utils import create_random_dfa

class DFAGeneticAlgorithm:
    """Genetic Algorithm for evolving DFAs to solve specific problems."""
    
    def __init__(self, 
                 population_size: int,
                 num_states: int,
                 alphabet: Set[str],
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        """Initialize the genetic algorithm.
        
        Args:
            population_size: Number of DFAs in the population
            num_states: Number of states for each DFA
            alphabet: Input alphabet
            mutation_rate: Probability of mutation (default: 0.1)
            crossover_rate: Probability of crossover (default: 0.7)
        """
        self.population_size = population_size
        self.num_states = num_states
        self.alphabet = alphabet
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[DFA] = []
        
    def initialize_population(self) -> None:
        """Initialize the population with random DFAs."""
        self.population = [
            create_random_dfa(
                num_states=self.num_states,
                alphabet=self.alphabet,
                num_final_states=random.randint(1, self.num_states // 2),
                num_dead_states=random.randint(0, self.num_states // 2)
            )
            for _ in range(self.population_size)
        ]
    
    def fitness_xor(self, dfa: DFA) -> float:
        """Calculate fitness for XOR problem.
        
        The fitness is the percentage of correct classifications
        for all possible 2-input combinations.
        
        Args:
            dfa: DFA to evaluate
            
        Returns:
            float: Fitness score between 0 and 1
        """
        test_cases = [
            ("00", False),  # 0 XOR 0 = 0
            ("01", True),   # 0 XOR 1 = 1
            ("10", True),   # 1 XOR 0 = 1
            ("11", False)   # 1 XOR 1 = 0
        ]
        
        correct = 0
        for input_str, expected in test_cases:
            try:
                if dfa.accepts(input_str) == expected:
                    correct += 1
            except Exception:
                continue
                
        return correct / len(test_cases)
    
    def select_parent(self) -> DFA:
        """Select a parent using tournament selection.
        
        Returns:
            DFA: Selected parent
        """
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=self.fitness_xor)
    
    def mutate(self, dfa: DFA) -> DFA:
        """Mutate a DFA by randomly changing transitions and final states.
        
        Args:
            dfa: DFA to mutate
            
        Returns:
            DFA: Mutated DFA
        """
        mutated = DFA(dfa.num_states, set(dfa.alphabet))
        
        # Copy transitions with possible mutations
        for state in range(dfa.num_states):
            for symbol in dfa.alphabet:
                if random.random() < self.mutation_rate:
                    # Random new transition
                    to_state = random.randint(0, dfa.num_states - 1)
                else:
                    # Keep existing transition
                    to_state = dfa.get_next_state(state, symbol) or 0
                mutated.add_transition(state, symbol, to_state)
        
        # Mutate final states
        final_states = {s.id for s in dfa.states.values() if s.is_final}
        for state in range(dfa.num_states):
            if random.random() < self.mutation_rate:
                if state in final_states:
                    final_states.remove(state)
                else:
                    final_states.add(state)
        
        mutated.set_final_states(final_states)
        return mutated
    
    def crossover(self, parent1: DFA, parent2: DFA) -> Tuple[DFA, DFA]:
        """Perform crossover between two parent DFAs.
        
        Args:
            parent1: First parent DFA
            parent2: Second parent DFA
            
        Returns:
            Tuple[DFA, DFA]: Two child DFAs
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        child1 = DFA(parent1.num_states, set(parent1.alphabet))
        child2 = DFA(parent2.num_states, set(parent2.alphabet))
        
        # Crossover point for transition matrix
        crossover_point = random.randint(1, parent1.num_states - 1)
        
        # Copy transitions
        for state in range(parent1.num_states):
            for symbol in parent1.alphabet:
                if state < crossover_point:
                    # First part from parent1
                    to_state1 = parent1.get_next_state(state, symbol) or 0
                    to_state2 = parent2.get_next_state(state, symbol) or 0
                else:
                    # Second part from parent2
                    to_state1 = parent2.get_next_state(state, symbol) or 0
                    to_state2 = parent1.get_next_state(state, symbol) or 0
                    
                child1.add_transition(state, symbol, to_state1)
                child2.add_transition(state, symbol, to_state2)
        
        # Crossover final states
        final_states1 = {s.id for s in parent1.states.values() if s.is_final}
        final_states2 = {s.id for s in parent2.states.values() if s.is_final}
        
        child1_finals = {s for s in final_states1 if s < crossover_point} | \
                       {s for s in final_states2 if s >= crossover_point}
        child2_finals = {s for s in final_states2 if s < crossover_point} | \
                       {s for s in final_states1 if s >= crossover_point}
                       
        child1.set_final_states(child1_finals)
        child2.set_final_states(child2_finals)
        
        return child1, child2
    
    def evolve(self, generations: int = 100) -> Tuple[DFA, float]:
        """Evolve the population to find a DFA that solves the XOR problem.
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            Tuple[DFA, float]: Best DFA found and its fitness score
        """
        if not self.population:
            self.initialize_population()
            
        best_fitness = 0.0
        best_dfa = None
        
        for generation in range(generations):
            # Evaluate current population
            fitness_scores = [(dfa, self.fitness_xor(dfa)) for dfa in self.population]
            current_best = max(fitness_scores, key=lambda x: x[1])
            
            if current_best[1] > best_fitness:
                best_fitness = current_best[1]
                best_dfa = current_best[0]
                
            if best_fitness == 1.0:
                break
                
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(current_best[0])
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim population to exact size
            self.population = new_population[:self.population_size]
        
        return best_dfa, best_fitness 