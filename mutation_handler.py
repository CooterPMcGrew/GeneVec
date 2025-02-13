"""
Author: Cooter McGrew
Date: 2025-02-13
Description:
    This program applies mutations to genes, handling both live mutations (within a single generation)
    and intergenerational mutations (across generations). 

    Features:
    - Accepts a structured gene dictionary (from the gene codifier).
    - Supports different mutation distributions: Normal, Uniform, Exponential.
    - Provides separate functions for live and intergenerational mutations.
    - Allows tuning of mutation rates and variance.

Usage:
    ```
    mutator = MutationHandler(mutation_rate=0.05, distribution="normal", variance=0.1)
    mutated_genes = mutator.apply_mutation(genes, mutation_type="intergenerational")
    ```
"""

import random
import numpy as np

class MutationHandler:
    """
    A class that handles mutations for genetic algorithms.
    """

    def __init__(self, mutation_rate=0.01, distribution="normal", variance=0.05):
        """
        Initialize the mutation handler.

        Parameters:
        - mutation_rate (float): Probability of a gene mutating (default: 0.01).
        - distribution (str): Type of mutation distribution. Options: 'normal', 'uniform', 'exponential'.
        - variance (float): Controls mutation strength (higher = more drastic changes).
        """
        self.mutation_rate = mutation_rate
        self.distribution = distribution.lower()
        self.variance = variance

    def mutate_value(self, value):
        """
        Applies a mutation to a numerical value based on the chosen distribution.

        Parameters:
        - value (float or int): The original value.

        Returns:
        - float: Mutated value.
        """
        if self.distribution == "normal":
            mutation = np.random.normal(loc=0, scale=self.variance)  # Gaussian mutation
        elif self.distribution == "uniform":
            mutation = np.random.uniform(-self.variance, self.variance)  # Uniform mutation
        elif self.distribution == "exponential":
            mutation = np.random.exponential(self.variance) * random.choice([-1, 1])  # Skewed mutation
        else:
            raise ValueError("Unsupported mutation distribution. Choose from: 'normal', 'uniform', 'exponential'.")
        
        return value + mutation  # Apply mutation

    def mutate_gene(self, gene):
        """
        Mutates a single gene.

        Parameters:
        - gene (dict): The gene dictionary.

        Returns:
        - dict: Mutated gene.
        """
        mutated_gene = gene.copy()
        for key, value in mutated_gene["data"].items():
            if isinstance(value, (int, float)) and random.random() < self.mutation_rate:
                mutated_gene["data"][key] = self.mutate_value(value)  # Mutate numeric values

        return mutated_gene

    def apply_mutation(self, genes, mutation_type="intergenerational"):
        """
        Applies mutations to all genes in a population.

        Parameters:
        - genes (dict): The structured gene dictionary.
        - mutation_type (str): Either "live" for in-generation mutations or "intergenerational".

        Returns:
        - dict: Mutated gene dictionary.
        """
        if mutation_type not in ["live", "intergenerational"]:
            raise ValueError("mutation_type must be 'live' or 'intergenerational'.")

        mutated_genes = {}
        for gene_id, gene in genes.items():
            if mutation_type == "intergenerational" or random.random() < 0.5:  # Live mutations happen with 50% chance
                mutated_genes[gene_id] = self.mutate_gene(gene)
            else:
                mutated_genes[gene_id] = gene  # No mutation

        return mutated_genes

# Example Usage
if __name__ == "__main__":
    # Example genes (normally loaded from the codifier)
    sample_genes = {
        "A23": {"data": {"fitness": 0.85, "size": 100, "speed": 4.5}},
        "B17": {"data": {"fitness": 0.92, "size": 95, "speed": 5.0}},
        "C09": {"data": {"fitness": 0.78, "size": 110, "speed": 4.2}},
    }

    # Initialize mutator with 5% mutation rate, normal distribution, and variance of 0.1
    mutator = MutationHandler(mutation_rate=0.05, distribution="normal", variance=0.1)

    # Apply live mutations (happens during a gene's life)
    live_mutations = mutator.apply_mutation(sample_genes, mutation_type="live")
    print("Live Mutations:", live_mutations)

    # Apply intergenerational mutations (happens when genes pass to next generation)
    intergen_mutations = mutator.apply_mutation(sample_genes, mutation_type="intergenerational")
    print("Intergenerational Mutations:", intergen_mutations)
