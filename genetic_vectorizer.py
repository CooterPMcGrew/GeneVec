"""
Genetic Vectorizer - Modular Genetic Algorithm Framework

Author: Cooter McGrew

Overview:
    This module provides a framework for encoding arbitrary data into vector representations
    so that genetic algorithms can be applied universally. The goal is to ensure that the
    genetic algorithms are decoupled from specific data formats, allowing optimization to
    occur on any problem domain without requiring custom implementations for each new dataset.

Structure:
    - Vectorizer: Handles encoding and decoding of data to and from vector space.
    - Genetic Operators: Mutation, crossover, and selection functions.
    - Evolution Engine: Handles the execution and progression of genetic algorithms.
"""

import numpy as np
import random

class Vectorizer:
    """
    Handles encoding and decoding of different data modalities to and from vector representations.
    This allows genetic algorithms to work on any type of data without custom adaptation.
    """
    
    @staticmethod
    def encode(data):
        """
        Converts input data into a numerical vector representation.
        This needs to be expanded to support multiple data types.

        Args:
            data (list): List of inputs (can be strings, images, numerical data, etc.)
        
        Returns:
            np.ndarray: Vectorized representation of input data
        """
        # Placeholder: Right now, this is just a simple hash-based encoding for strings.
        # Eventually, we'll add real feature extraction for different types of inputs.
        
        if isinstance(data, list):
            return np.array([hash(item) % 1000 for item in data], dtype=np.float32)
        else:
            raise TypeError("Unsupported data type for encoding")
    
    @staticmethod
    def decode(vector):
        """
        Decodes a vector representation back into human-readable format.
        Placeholder: We can’t recover original inputs from hashes, but this is where
        more advanced feature engineering will come in.
        
        Args:
            vector (np.ndarray): Vectorized data
        
        Returns:
            list: Decoded representations (currently just returning the raw numbers)
        """
        return vector.tolist()

class EvolutionEngine:
    """
    Runs the genetic algorithm on encoded data.
    This class is the core of the framework and handles population evolution over generations.
    """
    
    def __init__(self, population_size=100, generations=50, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
    
    def initialize_population(self, vector_size):
        """
        Initializes a population of random vectors.

        Args:
            vector_size (int): Size of each individual in the population
        """
        self.population = [np.random.rand(vector_size) for _ in range(self.population_size)]
    
    def fitness(self, individual):
        """
        Placeholder fitness function. Replace this with problem-specific logic.

        Args:
            individual (np.ndarray): A single solution candidate

        Returns:
            float: Fitness score
        """
        return -np.sum(individual)  # Example: Minimize sum (arbitrary for now)
    
    def mutate(self, individual):
        """
        Applies mutation to an individual based on the mutation rate.
        """
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(individual) - 1)
            individual[idx] += np.random.normal(0, 0.1)  # Small Gaussian perturbation
        return individual
    
    def crossover(self, parent1, parent2):
        """
        Applies crossover between two parents to produce a child.
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child = np.concatenate((parent1[:point], parent2[point:]))
            return child
        return parent1
    
    def evolve(self):
        """
        Runs the genetic algorithm for the specified number of generations.
        """
        for generation in range(self.generations):
            # Evaluate fitness of each individual
            fitness_scores = [self.fitness(ind) for ind in self.population]
            
            # Select the top individuals
            sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda pair: pair[0])]
            self.population = sorted_population[:self.population_size // 2]
            
            # Reproduce new individuals
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(self.population, 2)
                child = self.mutate(self.crossover(parent1, parent2))
                new_population.append(child)
            
            self.population = new_population
            print(f"Generation {generation+1} complete. Best fitness: {min(fitness_scores)}")
        
        return self.population[0]  # Return the best individual

# Example usage
if __name__ == "__main__":
    data = ["Hello", "World", "Genetic", "Algorithms"]
    vectorized_data = Vectorizer.encode(data)
    
    engine = EvolutionEngine()
    engine.initialize_population(vector_size=len(vectorized_data))
    best_solution = engine.evolve()
    print("Best Solution Found:", best_solution)
