"""
Fitness function for CLF-GWO: weighted combination of validation accuracy
and inference-latency penalty (see manuscript Section 4.3, alpha/gamma terms).
"""
import numpy as np


def compute_fitness(val_accuracy: float, latency_ms: float,
                     alpha: float = 0.8, gamma: float = 0.1,
                     latency_budget_ms: float = 1.0) -> float:
    """
    fitness = alpha * val_accuracy - gamma * max(0, latency_ms - latency_budget_ms)
    Higher is better.
    """
    latency_penalty = max(0.0, latency_ms - latency_budget_ms)
    return alpha * val_accuracy - gamma * latency_penalty


def evaluate_population(population, eval_fn, alpha=0.8, gamma=0.1, latency_budget_ms=1.0):
    """
    population: list of candidate solutions (feature mask + hyperparameters)
    eval_fn: callable(candidate) -> (val_accuracy, latency_ms)
    Returns list of fitness values aligned with population.
    """
    fitness_values = []
    for candidate in population:
        acc, latency = eval_fn(candidate)
        fitness_values.append(compute_fitness(acc, latency, alpha, gamma, latency_budget_ms))
    return np.array(fitness_values)