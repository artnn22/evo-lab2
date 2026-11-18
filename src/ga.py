"""Генетический алгоритм для дискретного вектора назначений."""
import time

import numpy as np

from .matching import random_feasible, repair
from .problem import evaluate, slot_groups


def evaluation_budget(config):
    return config["population_size"] + config["generations"] * (
        config["population_size"] - config["elite_size"])


def crossover(first, second, rng):
    return [first[i] if rng.random() < 0.5 else second[i] for i in range(len(first))]


def mutate(point, config, data, rng):
    child = point.copy()
    if rng.random() < config["mutation_prob"]:
        if config["mutation"] == "reassign":
            event = int(rng.integers(len(child)))
            other_rooms = [r for r in range(len(data["rooms"])) if r != child[event]]
            if other_rooms:
                child[event] = int(rng.choice(other_rooms))
        elif config["mutation"] == "swap":
            groups = [g for g in slot_groups(data) if len(g) >= 2]
            if groups:
                group = groups[int(rng.integers(len(groups)))]
                a, b = rng.choice(group, size=2, replace=False)
                child[a], child[b] = child[b], child[a]
        else:
            raise ValueError("Неизвестный оператор мутации")
    return child


def tournament(population, scores, config, rng):
    participants = rng.integers(len(population), size=config["tournament_size"])
    winner = min(participants, key=lambda i: scores[i])
    return population[winner].copy()


def validate_config(config):
    if not 0 < config["elite_size"] < config["population_size"]:
        raise ValueError("Элита должна быть положительной и меньше популяции")
    if config["generations"] < 0 or config["tournament_size"] < 1:
        raise ValueError("Некорректное число поколений или размер турнира")
    if config["constraints"] not in ("penalty", "repair"):
        raise ValueError("Неизвестный способ обработки ограничений")
    if config["mutation"] not in ("reassign", "swap"):
        raise ValueError("Неизвестный оператор мутации")
    for key in ["mutation_prob", "crossover_prob"]:
        if not 0 <= config[key] <= 1:
            raise ValueError("Вероятность должна быть от 0 до 1")


def run_ga(config, data, seed):
    validate_config(config)
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    best_x = None
    best_f = float("inf")
    n_evals = 0
    n_feasible = 0
    history = []
    feasible_history = []

    def score(point):
        nonlocal best_x, best_f, n_evals, n_feasible
        assessment = evaluate(point, data)
        n_evals += 1
        if assessment["feasible"]:
            n_feasible += 1
            if assessment["empty_seats"] < best_f:
                best_x = point.copy()
                best_f = assessment["empty_seats"]
        return assessment["fitness"]

    population = []
    for i in range(config["population_size"]):
        if i == 0:
            point = random_feasible(data, rng)
        else:
            point = [int(rng.integers(len(data["rooms"]))) for _ in data["events"]]
            if config["constraints"] == "repair":
                point = repair(point, data)
        population.append(point)
    scores = [score(point) for point in population]
    history.append(best_f)
    feasible_history.append(n_feasible / n_evals)

    for generation in range(config["generations"]):
        ranking = sorted(range(len(population)), key=lambda i: scores[i])
        elite = ranking[:config["elite_size"]]
        new_population = [population[i].copy() for i in elite]
        new_scores = [scores[i] for i in elite]
        while len(new_population) < config["population_size"]:
            child = tournament(population, scores, config, rng)
            if rng.random() < config["crossover_prob"]:
                other = tournament(population, scores, config, rng)
                child = crossover(child, other, rng)
            child = mutate(child, config, data, rng)
            if config["constraints"] == "repair":
                child = repair(child, data)
            new_population.append(child)
            new_scores.append(score(child))
        population, scores = new_population, new_scores
        history.append(best_f)
        feasible_history.append(n_feasible / n_evals)

    return {"best_x": best_x, "best_f": best_f, "history": history,
            "feasible_history": feasible_history, "n_evals": n_evals,
            "n_feasible": n_feasible, "seconds": time.perf_counter() - started}
