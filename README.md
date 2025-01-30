# Genetic Vectorizer

## Overview
**Genetic Vectorizer** is a modular, input-agnostic genetic algorithm framework designed to work with any data type. The core functionality encodes different modalities (numerical data, text, images, signals, etc.) into a standardized vector format, enabling seamless application of genetic algorithms across various domains.

## Features
- **Modality-Agnostic Encoding**: Supports structured data, text, images, and signals.
- **Pluggable Genetic Algorithms**: Includes NEAT-style evolution, classical GAs, and evolutionary strategies.
- **Flexible Fitness Functions**: Users can define custom fitness criteria tailored to their problem domain.
- **Parallel & GPU Acceleration**: Optimized for fast execution using vectorized NumPy/PyTorch.
- **Modular & Extensible**: Each component can be swapped out or extended for specific use cases.

## Installation
```bash
pip install genetic-vectorizer
```

## Usage Example
```python
from genetic_vectorizer import EvolutionEngine, Vectorizer

data = ["Example input 1", "Example input 2", "Example input 3"]  # Any input type
vectorized_data = Vectorizer.encode(data)

ega = EvolutionEngine(population_size=100, generations=50)
optimal_solution = ga.evolve(vectorized_data)
print(optimal_solution)
```

## Roadmap
- Implement encoding strategies for various data types.
- Develop efficient genetic operations (mutation, crossover, selection).
- Expand support for evolutionary strategies beyond traditional genetic algorithms.
- Add visualization tools for tracking fitness improvement.

## Contributing
We welcome contributions! Please submit issues, feature requests, or pull requests to help improve this project.

## License
MIT License

