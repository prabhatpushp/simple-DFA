"""Example of evolving a DFA to solve the n-input XOR problem."""
import sys
import os
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.genetic.dfa_xor import XORDFAGeneticAlgorithm
from src.automata.utils import pretty_print_dfa, show_input_transitions

@dataclass
class XOREvolutionConfig:
    """Configuration for XOR DFA evolution."""
    # DFA Structure
    num_states: int
    alphabet: Set[str]
    min_final_states: int
    max_final_states_ratio: float
    num_dead_states: Optional[int]
    
    # Genetic Algorithm Parameters
    population_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    max_generations: int
    elite_size: int
    
    # XOR Problem Specific
    num_inputs: int
    
    # Metrics and Logging
    track_metrics: bool = True
    print_progress: bool = True
    save_best: bool = True
    max_history_size: int = 10
    
    @classmethod
    def create_default(cls, num_inputs: int = 10) -> 'XOREvolutionConfig':
        """Create default configuration.
        
        Args:
            num_inputs: Number of inputs for XOR problem
            
        Returns:
            Default configuration
        """
        return cls(
            # DFA Structure
            num_states=max(4, 2 ** num_inputs),
            alphabet={'0', '1'},
            min_final_states=1,
            max_final_states_ratio=0.5,
            num_dead_states=None,
            
            # Genetic Algorithm
            population_size=100,
            mutation_rate=0.1,
            crossover_rate=0.7,
            tournament_size=3,
            max_generations=200,
            elite_size=2,
            
            # XOR Problem
            num_inputs=num_inputs
        )

def get_test_cases(num_inputs: int) -> List[Tuple[str, bool]]:
    """Generate test cases for n-input XOR.
    
    Args:
        num_inputs: Number of inputs for XOR
        
    Returns:
        List of tuples containing input string and expected output
    """
    test_cases = []
    num_combinations = 2 ** num_inputs
    
    for i in range(num_combinations):
        input_str = format(i, f'0{num_inputs}b')
        expected = bin(i).count('1') % 2 == 1
        test_cases.append((input_str, expected))
        
    return test_cases

def print_evolution_progress(generation: int, best_fitness: float, avg_fitness: float, 
                           population_diversity: float) -> None:
    """Print evolution progress.
    
    Args:
        generation: Current generation
        best_fitness: Best fitness in population
        avg_fitness: Average fitness in population
        population_diversity: Population diversity measure
    """
    print(f"Generation {generation}: "
          f"Best = {best_fitness:.2%}, "
          f"Avg = {avg_fitness:.2%}, "
          f"Diversity = {population_diversity:.4f}")

def print_final_metrics(generations: int, convergence_gen: int, time_taken: float,
                       best_fitness: float, avg_fitness: float, worst_fitness: float,
                       success_rate: float, diversity: float) -> None:
    """Print final evolution metrics.
    
    Args:
        generations: Total generations run
        convergence_gen: Generation where convergence occurred
        time_taken: Total time taken
        best_fitness: Best fitness achieved
        avg_fitness: Average fitness in final population
        worst_fitness: Worst fitness in final population
        success_rate: Success rate of final population
        diversity: Final population diversity
    """
    print("\n" + "="*50)
    print("Evolution Metrics:")
    print("="*50)
    
    print("\nPerformance Metrics:")
    print(f"  Best Fitness: {best_fitness:.2%}")
    print(f"  Average Fitness: {avg_fitness:.2%}")
    print(f"  Worst Fitness: {worst_fitness:.2%}")
    print(f"  Success Rate: {success_rate:.2%}")
    
    print("\nConvergence Metrics:")
    print(f"  Total Generations: {generations}")
    if convergence_gen > 0:
        print(f"  Converged at Generation: {convergence_gen}")
    print(f"  Time Taken: {time_taken:.2f} seconds")
    
    print("\nPopulation Metrics:")
    print(f"  Population Diversity: {diversity:.4f}")

def test_generalization(dfa, trained_inputs: int, test_range: range) -> Dict[int, float]:
    """Test the DFA's ability to generalize to different input sizes.
    
    Args:
        dfa: The trained DFA to test
        trained_inputs: Number of inputs the DFA was trained on
        test_range: Range of input sizes to test
        
    Returns:
        Dictionary mapping input sizes to accuracy scores
    """
    results = {}
    
    print("\nGeneralization Testing:")
    print("="*50)
    print(f"DFA trained on {trained_inputs}-input XOR")
    print("-"*50)
    print(f"{'Input Size':<15} {'Accuracy':<10} {'Status':<15}")
    print("-"*50)
    
    for num_inputs in test_range:
        test_cases = get_test_cases(num_inputs)
        correct = 0
        
        for test_input, expected in test_cases:
            if dfa.accepts(test_input) == expected:
                correct += 1
                
        accuracy = correct / len(test_cases)
        results[num_inputs] = accuracy
        
        # Determine generalization status
        if accuracy == 1.0:
            status = "Perfect"
        elif accuracy >= 0.95:
            status = "Excellent"
        elif accuracy >= 0.8:
            status = "Good"
        elif accuracy >= 0.6:
            status = "Fair"
        else:
            status = "Poor"
            
        print(f"{num_inputs:<15} {accuracy:,.2%}    {status:<15}")
    
    return results

def create_combined_dataset(max_input_size: int) -> List[Tuple[str, bool]]:
    """Create a combined dataset of all XOR cases up to max_input_size.
    
    Args:
        max_input_size: Maximum number of inputs to include
        
    Returns:
        List of tuples containing input string and expected output
    """
    combined_dataset = []
    
    # Generate test cases for each input size from 1 to max_input_size
    for size in range(1, max_input_size + 1):
        combined_dataset.extend(get_test_cases(size))
    
    return combined_dataset

def train_on_combined_dataset(max_train_size: int, max_test_size: int,
                            base_config: Optional[XOREvolutionConfig] = None) -> None:
    """Train a DFA on combined dataset of smaller inputs and test generalization.
    
    Args:
        max_train_size: Maximum input size to include in training
        max_test_size: Maximum input size to test generalization on
        base_config: Base configuration to use for training
    """
    print("\nCombined Dataset Training Experiment")
    print("="*50)
    print(f"Training on all XOR cases up to {max_train_size} inputs")
    print(f"Testing generalization up to {max_test_size} inputs")
    print("="*50)
    
    # Create combined training dataset
    training_dataset = create_combined_dataset(max_train_size)
    total_cases = len(training_dataset)
    print(f"\nTraining Dataset:")
    print(f"Total training cases: {total_cases}")
    
    # Group cases by input length for analysis
    cases_by_length = {}
    for input_str, expected in training_dataset:
        length = len(input_str)
        if length not in cases_by_length:
            cases_by_length[length] = 0
        cases_by_length[length] += 1
    
    print("\nCases by input length:")
    for length, count in sorted(cases_by_length.items()):
        print(f"  {length}-input cases: {count}")
    
    # Create configuration for training
    if base_config is None:
        config = XOREvolutionConfig.create_default(max_train_size)
    else:
        config = XOREvolutionConfig(
            num_states=max(4, 2 * max_train_size),
            alphabet=base_config.alphabet,
            min_final_states=base_config.min_final_states,
            max_final_states_ratio=base_config.max_final_states_ratio,
            num_dead_states=base_config.num_dead_states,
            population_size=base_config.population_size,
            mutation_rate=base_config.mutation_rate,
            crossover_rate=base_config.crossover_rate,
            tournament_size=base_config.tournament_size,
            max_generations=base_config.max_generations,
            elite_size=base_config.elite_size,
            num_inputs=max_train_size
        )
    
    # Train DFA
    print("\nTraining DFA on combined dataset...")
    ga = XORDFAGeneticAlgorithm(config)
    best_dfa, metrics = ga.evolve()
    
    if best_dfa:
        # Test training performance
        print("\nTraining Performance:")
        print("-"*50)
        correct_by_length = {}
        total_by_length = {}
        
        for input_str, expected in training_dataset:
            length = len(input_str)
            if length not in total_by_length:
                total_by_length[length] = 0
                correct_by_length[length] = 0
            
            total_by_length[length] += 1
            if best_dfa.accepts(input_str) == expected:
                correct_by_length[length] += 1
        
        print(f"{'Input Size':<12} {'Accuracy':<10} {'Cases':<10}")
        print("-"*50)
        for length in sorted(total_by_length.keys()):
            accuracy = correct_by_length[length] / total_by_length[length]
            print(f"{length:<12} {accuracy:,.2%}    {total_by_length[length]:<10}")
        
        # Test generalization on larger inputs
        print("\nGeneralization to Larger Inputs:")
        print("="*50)
        print(f"{'Input Size':<12} {'Accuracy':<10} {'Status':<10}")
        print("-"*50)
        
        generalization_results = {}
        for size in range(max_train_size + 1, max_test_size + 1):
            test_cases = get_test_cases(size)
            correct = 0
            
            for test_input, expected in test_cases:
                if best_dfa.accepts(test_input) == expected:
                    correct += 1
            
            accuracy = correct / len(test_cases)
            generalization_results[size] = accuracy
            
            # Determine status
            if accuracy == 1.0:
                status = "Perfect"
            elif accuracy >= 0.95:
                status = "Excellent"
            elif accuracy >= 0.8:
                status = "Good"
            elif accuracy >= 0.6:
                status = "Fair"
            else:
                status = "Poor"
            
            print(f"{size:<12} {accuracy:,.2%}    {status:<10}")
        
        # Print overall generalization performance
        avg_generalization = sum(generalization_results.values()) / len(generalization_results)
        print(f"\nOverall Generalization Performance: {avg_generalization:.2%}")
        
        # Print DFA structure
        print("\nFinal DFA Structure:")
        print(pretty_print_dfa(best_dfa))
    else:
        print("No solution found!")

def main(config: Optional[XOREvolutionConfig] = None):
    """Evolve a DFA to solve the n-input XOR problem.
    
    Args:
        config: Optional configuration, uses default if not provided
    """
    # Use default config if none provided
    if config is None:
        config = XOREvolutionConfig.create_default()
    
    # Initialize genetic algorithm
    ga = XORDFAGeneticAlgorithm(config)
    
    print(f"\nStarting evolution for {config.num_inputs}-input XOR...")
    print(f"Configuration:")
    print(f"  Population size: {config.population_size}")
    print(f"  Number of states: {config.num_states}")
    print(f"  Mutation rate: {config.mutation_rate}")
    print(f"  Crossover rate: {config.crossover_rate}")
    print(f"  Tournament size: {config.tournament_size}")
    print(f"  Elite size: {config.elite_size}")
    print(f"  Maximum generations: {config.max_generations}")
    
    # Record start time
    start_time = time.time()
    
    # Evolve DFA
    best_dfa, metrics = ga.evolve()
    
    # Update time taken
    metrics.time_taken = time.time() - start_time
    
    # Print final metrics
    print_final_metrics(
        generations=metrics.generations,
        convergence_gen=metrics.convergence_generation,
        time_taken=metrics.time_taken,
        best_fitness=metrics.best_fitness,
        avg_fitness=metrics.avg_fitness,
        worst_fitness=metrics.worst_fitness,
        success_rate=metrics.success_rate,
        diversity=metrics.population_diversity
    )
    
    if best_dfa:
        print("\nBest DFA found:")
        print(pretty_print_dfa(best_dfa))
        
        print("\nTesting all XOR cases:")
        total_tests = 0
        correct_tests = 0
        
        test_cases = get_test_cases(config.num_inputs)
        for test_input, expected in test_cases:
            print(f"\nTesting input: {test_input}")
            print(show_input_transitions(best_dfa, test_input))
            result = best_dfa.accepts(test_input)
            result_str = "ACCEPTED" if result else "REJECTED"
            expected_str = "ACCEPTED" if expected else "REJECTED"
            print(f"Result: {result_str} (Expected: {expected_str})")
            
            total_tests += 1
            if result == expected:
                correct_tests += 1
        
        print("\nTest Results:")
        print(f"  Total Test Cases: {total_tests}")
        print(f"  Correct Results: {correct_tests}")
        print(f"  Accuracy: {(correct_tests/total_tests)*100:.2f}%")
        
        # Test generalization
        test_range = range(
            max(1, config.num_inputs - 5),  # Test 5 sizes smaller
            config.num_inputs + 6,          # Test 5 sizes larger
            1
        )
        generalization_results = test_generalization(best_dfa, config.num_inputs, test_range)
        
        # Print generalization summary
        print("\nGeneralization Summary:")
        print("="*50)
        smaller_inputs = [n for n in generalization_results.keys() if n < config.num_inputs]
        larger_inputs = [n for n in generalization_results.keys() if n > config.num_inputs]
        
        if smaller_inputs:
            avg_smaller = sum(generalization_results[n] for n in smaller_inputs) / len(smaller_inputs)
            print(f"Average accuracy on smaller inputs: {avg_smaller:.2%}")
        
        if larger_inputs:
            avg_larger = sum(generalization_results[n] for n in larger_inputs) / len(larger_inputs)
            print(f"Average accuracy on larger inputs: {avg_larger:.2%}")
    else:
        print("No solution found!")

def run_random_training(min_inputs: int = 3, max_inputs: int = 15, num_trials: int = 5,
                       base_config: Optional[XOREvolutionConfig] = None) -> None:
    """Train multiple DFAs on random input sizes and analyze generalization.
    
    Args:
        min_inputs: Minimum number of inputs to test
        max_inputs: Maximum number of inputs to test
        num_trials: Number of different input sizes to try
        base_config: Base configuration to use, will be modified for each trial
    """
    print("\nRandom Input Size Training Experiment")
    print("="*50)
    print(f"Training {num_trials} DFAs with input sizes between {min_inputs} and {max_inputs}")
    print("="*50)
    
    # Generate random input sizes
    input_sizes = random.sample(range(min_inputs, max_inputs + 1), num_trials)
    input_sizes.sort()  # Sort for clearer presentation
    
    results = {}
    best_dfas = {}
    
    # Train DFA for each input size
    for trial, num_inputs in enumerate(input_sizes, 1):
        print(f"\nTrial {trial}/{num_trials} - Training {num_inputs}-input XOR")
        print("-"*50)
        
        # Create configuration for this trial
        if base_config is None:
            config = XOREvolutionConfig.create_default(num_inputs)
        else:
            config = XOREvolutionConfig(
                num_states=max(4, 2 * num_inputs),
                alphabet=base_config.alphabet,
                min_final_states=base_config.min_final_states,
                max_final_states_ratio=base_config.max_final_states_ratio,
                num_dead_states=base_config.num_dead_states,
                population_size=base_config.population_size,
                mutation_rate=base_config.mutation_rate,
                crossover_rate=base_config.crossover_rate,
                tournament_size=base_config.tournament_size,
                max_generations=base_config.max_generations,
                elite_size=base_config.elite_size,
                num_inputs=num_inputs
            )
        
        # Train DFA
        ga = XORDFAGeneticAlgorithm(config)
        best_dfa, metrics = ga.evolve()
        
        if best_dfa:
            best_dfas[num_inputs] = best_dfa
            
            # Test generalization on all input sizes
            test_range = range(min_inputs, max_inputs + 1)
            results[num_inputs] = test_generalization(best_dfa, num_inputs, test_range)
    
    # Analyze results
    print("\nCross-Generalization Analysis")
    print("="*50)
    print(f"{'Trained On':<12} {'Tested On':<12} {'Accuracy':<10} {'Status':<10}")
    print("-"*50)
    
    overall_accuracy = []
    for trained_size in input_sizes:
        if trained_size not in results:
            continue
            
        accuracies = []
        for test_size in range(min_inputs, max_inputs + 1):
            accuracy = results[trained_size].get(test_size, 0.0)
            accuracies.append(accuracy)
            
            # Determine status
            if accuracy == 1.0:
                status = "Perfect"
            elif accuracy >= 0.95:
                status = "Excellent"
            elif accuracy >= 0.8:
                status = "Good"
            elif accuracy >= 0.6:
                status = "Fair"
            else:
                status = "Poor"
                
            print(f"{trained_size:<12} {test_size:<12} {accuracy:,.2%}    {status:<10}")
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        overall_accuracy.append(avg_accuracy)
        print(f"\nDFA trained on {trained_size} inputs - Average accuracy: {avg_accuracy:.2%}")
        print("-"*50)
    
    if overall_accuracy:
        print(f"\nOverall Generalization Performance: {sum(overall_accuracy)/len(overall_accuracy):.2%}")
        
        # Find best generalizing DFA
        best_generalizing_size = input_sizes[overall_accuracy.index(max(overall_accuracy))]
        print(f"\nBest Generalizing DFA (trained on {best_generalizing_size} inputs):")
        print(pretty_print_dfa(best_dfas[best_generalizing_size]))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolve a DFA to solve the n-input XOR problem')
    parser.add_argument('--num-inputs', type=int, default=10, help='Number of inputs for XOR')
    parser.add_argument('--population-size', type=int, default=100, help='Population size')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--tournament-size', type=int, default=3, help='Tournament size')
    parser.add_argument('--max-generations', type=int, default=200, help='Maximum generations')
    parser.add_argument('--elite-size', type=int, default=2, help='Number of elite individuals')
    parser.add_argument('--min-final-states', type=int, default=5, help='Minimum number of final states')
    parser.add_argument('--max-final-ratio', type=float, default=0.5, help='Maximum ratio of final states')
    parser.add_argument('--num-dead-states', type=int, help='Number of dead states')
    parser.add_argument('--random-training', action='store_true', help='Run random input size training experiment')
    parser.add_argument('--combined-training', action='store_true', help='Run combined dataset training experiment')
    parser.add_argument('--max-train-size', type=int, default=5, help='Maximum input size for training dataset')
    parser.add_argument('--max-test-size', type=int, default=10, help='Maximum input size for testing generalization')
    parser.add_argument('--min-inputs', type=int, default=3, help='Minimum number of inputs for random training')
    parser.add_argument('--max-inputs', type=int, default=15, help='Maximum number of inputs for random training')
    parser.add_argument('--num-trials', type=int, default=5, help='Number of trials for random training')
    
    args = parser.parse_args()
    
    if args.combined_training:
        # Create base configuration for combined training
        base_config = XOREvolutionConfig(
            num_states=4,  # Will be adjusted based on max_train_size
            alphabet={'0', '1'},
            min_final_states=args.min_final_states,
            max_final_states_ratio=args.max_final_ratio,
            num_dead_states=args.num_dead_states,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            tournament_size=args.tournament_size,
            max_generations=args.max_generations,
            elite_size=args.elite_size,
            num_inputs=0  # Will be set based on max_train_size
        )
        train_on_combined_dataset(
            max_train_size=args.max_train_size,
            max_test_size=args.max_test_size,
            base_config=base_config
        )
    elif args.random_training:
        # Create base configuration for random training
        base_config = XOREvolutionConfig(
            num_states=4,  # Will be adjusted for each trial
            alphabet={'0', '1'},
            min_final_states=args.min_final_states,
            max_final_states_ratio=args.max_final_ratio,
            num_dead_states=args.num_dead_states,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            tournament_size=args.tournament_size,
            max_generations=args.max_generations,
            elite_size=args.elite_size,
            num_inputs=0  # Will be set for each trial
        )
        run_random_training(
            min_inputs=args.min_inputs,
            max_inputs=args.max_inputs,
            num_trials=args.num_trials,
            base_config=base_config
        )
    else:
        # Create configuration from command line arguments
        config = XOREvolutionConfig(
            num_states=max(4, 2 * args.num_inputs),
            alphabet={'0', '1'},
            min_final_states=args.min_final_states,
            max_final_states_ratio=args.max_final_ratio,
            num_dead_states=args.num_dead_states,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            tournament_size=args.tournament_size,
            max_generations=args.max_generations,
            elite_size=args.elite_size,
            num_inputs=args.num_inputs
        )
        main(config) 