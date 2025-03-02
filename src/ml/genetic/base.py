"""Base classes for genetic algorithms."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple, Any
from src.config.genetic_config import GeneticConfig

T = TypeVar('T')  # Type of individual in population

class GeneticAlgorithm(Generic[T], ABC):
    """Abstract base class for genetic algorithms."""
    
    def __init__(self, config: GeneticConfig):
        """Initialize genetic algorithm.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.population: List[T] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.best_individual = None
        
    @abstractmethod
    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        pass
    
    @abstractmethod
    def fitness(self, individual: T) -> float:
        """Calculate fitness for an individual.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            float: Fitness score between 0 and 1
        """
        pass
    
    @abstractmethod
    def select_parent(self) -> T:
        """Select a parent using tournament selection.
        
        Returns:
            Selected parent individual
        """
        pass
    
    @abstractmethod
    def mutate(self, individual: T) -> T:
        """Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        pass
    
    @abstractmethod
    def crossover(self, parent1: T, parent2: T) -> Tuple[T, T]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two child individuals
        """
        pass
        
    def evolve(self) -> Tuple[T, float]:
        """Evolve the population.
        
        Returns:
            Tuple of (best individual, best fitness score)
        """
        if not self.population:
            self.initialize_population()
            
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate current population
            fitness_scores = [(ind, self.fitness(ind)) for ind in self.population]
            current_best = max(fitness_scores, key=lambda x: x[1])
            
            if current_best[1] > self.best_fitness:
                self.best_fitness = current_best[1]
                self.best_individual = current_best[0]
                
            if self.best_fitness == 1.0:
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
            
            # Optional: Call hook for custom per-generation logic
            self.on_generation_complete()
            
        return self.best_individual, self.best_fitness
    
    def on_generation_complete(self) -> None:
        """Hook called after each generation.
        
        Override this method to add custom logic after each generation.
        """
        pass 