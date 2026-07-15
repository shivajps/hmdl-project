"""
Chaotic Lévy-Flight Grey Wolf Optimizer (CLF-GWO) for joint feature selection
and hyperparameter tuning (manuscript Section 4.3).

Reproducibility settings (Section 4.3.4):
  population size N_pop = 30
  max iterations T_max = 100
  early stopping: no improvement > 1e-4 over 15 consecutive iterations
  seeds: {7, 42, 123, 2024, 31415}
"""
import argparse
import numpy as np
import yaml

from fitness_function import evaluate_population

SINGER_MAP_MU = 1.07  # chaotic map parameter — verify against your actual run config


def singer_map_init(n_pop: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n_pop, dim))
    for _ in range(50):  # burn-in iterations
        x = SINGER_MAP_MU * (7.86 * x - 23.31 * x**2 + 28.75 * x**3 - 13.302875 * x**4)
        x = np.clip(x, 0, 1)
    return x


def levy_flight(dim: int, seed: int, beta: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = (
        (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2))
        / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1 / beta))


def clf_gwo(dim: int, eval_fn, n_pop=30, t_max=100, seed=42,
            early_stop_delta=1e-4, early_stop_patience=15):
    positions = singer_map_init(n_pop, dim, seed)
    fitness = evaluate_population(positions, eval_fn)

    best_idx = np.argsort(-fitness)[:3]
    alpha, beta, delta = positions[best_idx[0]], positions[best_idx[1]], positions[best_idx[2]]
    best_fitness_history = [fitness[best_idx[0]]]
    stale_iters = 0

    for t in range(t_max):
        a = 2 - 2 * t / t_max
        for i in range(n_pop):
            for lead in (alpha, beta, delta):
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * lead - positions[i])
                positions[i] = lead - A * D
            if np.random.rand() < 0.1:  # Lévy-flight perturbation probability
                positions[i] += levy_flight(dim, seed + t + i)
            positions[i] = np.clip(positions[i], 0, 1)

        fitness = evaluate_population(positions, eval_fn)
        best_idx = np.argsort(-fitness)[:3]
        alpha, beta, delta = positions[best_idx[0]], positions[best_idx[1]], positions[best_idx[2]]

        current_best = fitness[best_idx[0]]
        if current_best - best_fitness_history[-1] <= early_stop_delta:
            stale_iters += 1
        else:
            stale_iters = 0
        best_fitness_history.append(current_best)

        if stale_iters >= early_stop_patience:
            print(f"Early stopping at iteration {t+1} (no improvement > {early_stop_delta} for {early_stop_patience} iters)")
            break

    return alpha, best_fitness_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raise NotImplementedError(
        "Wire eval_fn to your actual train/validate loop (training/train.py) "
        "before running — this file provides the optimizer logic only."
    )