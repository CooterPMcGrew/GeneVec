"""
Author: Cooter McGrew
Date: 2025-02-13
Description:
    This program handles gene crossover for genetic algorithms. It provides multiple crossover types 
    (single-point, multi-point, uniform, blended, recombination) to allow controlled inheritance 
    between parent genes.

    Features:
    - Fully compatible with gene codification and mutation modules.
    - Modular: Easily integrates new crossover techniques.
    - Works on structured gene dictionaries from the codifier.
    
Usage:
    ```
    crossover = CrossoverHandler()
    offspring = crossover.perform_crossover(parent1, parent2, method="multi-point", points=2)
    ```
"""

import random
import numpy as np
from copy import deepcopy

class CrossoverHandler:
    """
    Handles crossover operations between genes for genetic algorithms.
    """

    def __init__(self):
        """
        Initializes the crossover handler.
        """
        pass

    def single_point_crossover(self, parent1, parent2):
        """
        Performs single-point crossover on two genes.

        Parameters:
        - parent1 (dict): First parent gene.
        - parent2 (dict): Second parent gene.

        Returns:
        - dict: Offspring gene.
        """
        keys = list(parent1["data"].keys())
        split_idx = random.randint(1, len(keys) - 1)

        offspring_data = {k: (parent1 if i < split_idx else parent2)["data"][k] for i, k in enumerate(keys)}

        return {"data": offspring_data}

    def multi_point_crossover(self, parent1, parent2, points=2):
        """
        Performs multi-point crossover.

        Parameters:
        - parent1 (dict): First parent gene.
        - parent2 (dict): Second parent gene.
        - points (int): Number of crossover points.

        Returns:
        - dict: Offspring gene.
        """
        keys = list(parent1["data"].keys())
        points = sorted(random.sample(range(1, len(keys)), points))
        
        offspring_data = {}
        parent_toggle = False  # Start with parent1

        prev_idx = 0
        for idx in points + [len(keys)]:
            segment = keys[prev_idx:idx]
            for k in segment:
                offspring_data[k] = (parent1 if parent_toggle else parent2)["data"][k]
            parent_toggle = not parent_toggle
            prev_idx = idx

        return {"data": offspring_data}

    def uniform_crossover(self, parent1, parent2):
        """
        Performs uniform crossover (randomly selecting attributes from either parent).

        Parameters:
        - parent1 (dict): First parent gene.
        - parent2 (dict): Second parent gene.

        Returns:
        - dict: Offspring gene.
        """
        offspring_data = {k: random.choice([parent1, parent2])["data"][k] for k in parent1["data"].keys()}

        return {"data": offspring_data}

    def blended_crossover(self, parent1, parent2, alpha=0.5):
        """
        Performs blended crossover for continuous values (weighted average).

        Parameters:
        - parent1 (dict): First parent gene.
        - parent2 (dict): Second parent gene.
        - alpha (float): Blend factor (default: 0.5 for equal blending).

        Returns:
        - dict: Offspring gene.
        """
        offspring_data = {
            k: alpha * parent1["data"][k] + (1 - alpha) * parent2["data"][k]
            if isinstance(parent1["data"][k], (int, float)) else random.choice([parent1, parent2])["data"][k]
            for k in parent1["data"].keys()
        }

        return {"data": offspring_data}

    def segment_recombination(self, parent1, parent2, segment_map):
        """
        Recombines genes based on logical grouping of attributes.

        Parameters:
        - parent1 (dict): First parent gene.
        - parent2 (dict): Second parent gene.
        - segment_map (dict): Mapping of attribute groups (e.g., {"physical": ["size", "weight"], "performance": ["speed", "efficiency"]}).

        Returns:
        - dict: Offspring gene.
        """
        offspring_data = {}
        for segment, attributes in segment_map.items():
            chosen_parent = random.choice([parent1, parent2])
            for attr in attributes:
                if attr in parent1["data"]:
                    offspring_data[attr] = chosen_parent["data"][attr]

        return {"data": offspring_data}

    def perform_crossover(self, parent1, parent2, method="single-point", **kwargs):
        """
        General crossover function that selects the appropriate crossover method.

        Parameters:
        - parent1 (dict): First parent gene.
        - parent2 (dict): Second parent gene.
        - method (str): Crossover method (options: "single-point", "multi-point", "uniform", "blended", "segment").
        - kwargs: Additional parameters for specific crossover methods.

        Returns:
        - dict: Offspring gene.
        """
        if method == "single-point":
            return self.single_point_crossover(parent1, parent2)
        elif method == "multi-point":
            points = kwargs.get("points", 2)
            return self.multi_point_crossover(parent1, parent2, points=points)
        elif method == "uniform":
            return self.uniform_crossover(parent1, parent2)
        elif method == "blended":
            alpha = kwargs.get("alpha", 0.5)
            return self.blended_crossover(parent1, parent2, alpha=alpha)
        elif method == "segment":
            segment_map = kwargs.get("segment_map", {})
            return self.segment_recombination(parent1, parent2, segment_map)
        else:
            raise ValueError("Invalid crossover method. Choose from: 'single-point', 'multi-point', 'uniform', 'blended', 'segment'.")

# Example Usage
if __name__ == "__main__":
    # Example parent genes
    parent1 = {"data": {"size": 100, "speed": 4.5, "fitness": 0.85}}
    parent2 = {"data": {"size": 95, "speed": 5.0, "fitness": 0.92}}

    crossover = CrossoverHandler()

    # Single-point crossover
    offspring_1 = crossover.perform_crossover(parent1, parent2, method="single-point")
    print("Single-Point Crossover:", offspring_1)

    # Multi-point crossover
    offspring_2 = crossover.perform_crossover(parent1, parent2, method="multi-point", points=2)
    print("Multi-Point Crossover:", offspring_2)

    # Uniform crossover
    offspring_3 = crossover.perform_crossover(parent1, parent2, method="uniform")
    print("Uniform Crossover:", offspring_3)

    # Blended crossover
    offspring_4 = crossover.perform_crossover(parent1, parent2, method="blended", alpha=0.6)
    print("Blended Crossover:", offspring_4)

    # Segment-based crossover
    segment_map = {"physical": ["size"], "performance": ["speed", "fitness"]}
    offspring_5 = crossover.perform_crossover(parent1, parent2, method="segment", segment_map=segment_map)
    print("Segment-Based Crossover:", offspring_5)
