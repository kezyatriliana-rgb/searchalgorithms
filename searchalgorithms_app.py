from dataclasses import dataclass
import random
import streamlit as st

POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50
MAX_FITNESS = 80

# Fixed experiment settings (no sidebar inputs)
RANDOM_SEED = 42
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.01


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: int
    average_fitness: float
    best_ones: int


def fitness(chromosome: str) -> int:
    """Return 80 only when an 80-bit chromosome contains exactly 50 ones."""
    if len(chromosome) != CHROMOSOME_LENGTH or any(bit not in '01' for bit in chromosome):
        raise ValueError('Chromosome must contain exactly 80 binary digits.')
    return MAX_FITNESS - abs(TARGET_ONES - chromosome.count('1'))


def create_chromosome(rng: random.Random) -> str:
