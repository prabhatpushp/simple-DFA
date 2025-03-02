"""Genetic algorithm implementation for evolving XOR DFAs."""
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

from src.automata.dfa import DFA
from src.automata.utils import create_random_dfa

@dataclass
class XOREvolutionMetrics:
    """Metrics for XOR DFA evolution."""
    generations: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    convergence_generation: int = 0
    time_taken: float = 0.0
    population_diversity: float = 0.0
    success_rate: float = 0.0
    best_individuals: List[DFA] = None
    
    def __post_init__(self):
        """Initialize lists."""
        self.best_individuals = []
    
    def update(self, generation: int, population_fitness: List[float]) -> None:
        """Update metrics with current generation data."""
        self.generations = generation + 1
        self.best_fitness = max(population_fitness)
        self.avg_fitness = sum(population_fitness) / len(population_fitness)
        self.worst_fitness = min(population_fitness)
        
        if self.best_fitness == 1.0 and self.convergence_generation == 0:
            self.convergence_generation = generation
            
        mean = self.avg_fitness
        variance = sum((f - mean) ** 2 for f in population_fitness) / len(population_fitness)
        self.population_diversity = variance ** 0.5
        self.success_rate = sum(1 for f in population_fitness if f > 0.5) / len(population_fitness)

class XORDFAGeneticAlgorithm:
    """Genetic algorithm for evolving DFAs to solve the XOR problem."""
    
    def __init__(self, config):
        """Initialize XOR DFA genetic algorithm.
        
        Args:
            config: Configuration object with required parameters
        """
        self.config = config
        self.population: List[DFA] = []
        self.metrics = XOREvolutionMetrics()
        self.generation = 0
        self.best_individual: Optional[DFA] = None
        
    def initialize_population(self) -> None:
        """Initialize population with random DFAs."""
        self.population = [
            create_random_dfa(
                num_states=self.config.num_states,
                alphabet=self.config.alphabet,
                num_final_states=random.randint(
                    self.config.min_final_states,
                    int(self.config.num_states * self.config.max_final_states_ratio)
                ),
                num_dead_states=self.config.num_dead_states or 0
            )
            for _ in range(self.config.population_size)
        ]
    
    def fitness(self, dfa: DFA) -> float:
        """Calculate fitness for XOR problem.
        
        Args:
            dfa: DFA to evaluate
            
        Returns:
            float: Fitness score between 0 and 1
        """
        test_cases = self._generate_test_cases()
        correct = 0
        
        for input_str, expected in test_cases:
            try:
                if dfa.accepts(input_str) == expected:
                    correct += 1
            except Exception:
                continue
                
        return correct / len(test_cases)
    
    def _generate_test_cases(self) -> List[Tuple[str, bool]]:
        """Generate test cases for n-input XOR."""
        test_cases = []
        num_combinations = 2 ** self.config.num_inputs
        
        for i in range(num_combinations):
            input_str = format(i, f'0{self.config.num_inputs}b')
            expected = bin(i).count('1') % 2 == 1
            test_cases.append((input_str, expected))
            
        return test_cases
    
    def select_parent(self) -> DFA:
        """Select a parent using tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=self.fitness)
    
    def mutate(self, dfa: DFA) -> DFA:
        """Mutate a DFA by randomly changing transitions and final states."""
        mutated = DFA(dfa.num_states, set(dfa.alphabet))
        
        # Copy transitions with possible mutations
        for state in range(dfa.num_states):
            for symbol in dfa.alphabet:
                if random.random() < self.config.mutation_rate:
                    # Random new transition
                    to_state = random.randint(0, dfa.num_states - 1)
                else:
                    # Keep existing transition
                    to_state = dfa.get_next_state(state, symbol) or 0
                mutated.add_transition(state, symbol, to_state)
        
        # Mutate final states
        final_states = {s.id for s in dfa.states.values() if s.is_final}
        for state in range(dfa.num_states):
            if random.random() < self.config.mutation_rate:
                if state in final_states:
                    final_states.remove(state)
                else:
                    final_states.add(state)
        
        mutated.set_final_states(final_states)
        return mutated
    
    def crossover(self, parent1: DFA, parent2: DFA) -> Tuple[DFA, DFA]:
        """Perform crossover between two parent DFAs."""
        if random.random() > self.config.crossover_rate:
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
    
    def evolve(self) -> Tuple[Optional[DFA], XOREvolutionMetrics]:
        """Evolve the population to find a DFA that solves the XOR problem."""
        if not self.population:
            self.initialize_population()
            
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate current population
            fitness_scores = [(dfa, self.fitness(dfa)) for dfa in self.population]
            current_best = max(fitness_scores, key=lambda x: x[1])
            
            # Update metrics
            population_fitness = [score for _, score in fitness_scores]
            self.metrics.update(generation, population_fitness)
            
            if current_best[1] > self.metrics.best_fitness:
                self.best_individual = current_best[0]
                if self.config.save_best:
                    self.metrics.best_individuals.append(current_best[0])
                    if len(self.metrics.best_individuals) > self.config.max_history_size:
                        self.metrics.best_individuals.pop(0)
            
            if self.metrics.best_fitness == 1.0:
                break
                
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            sorted_population = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
            new_population.extend(ind for ind, _ in sorted_population[:self.config.elite_size])
            
            # Generate rest of population
            while len(new_population) < self.config.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim population to exact size
            self.population = new_population[:self.config.population_size]
            
            # Print progress if enabled
            if self.config.print_progress:
                print(f"Generation {generation}: "
                      f"Best = {self.metrics.best_fitness:.2%}, "
                      f"Avg = {self.metrics.avg_fitness:.2%}, "
                      f"Diversity = {self.metrics.population_diversity:.4f}")
        
        return self.best_individual, self.metrics 