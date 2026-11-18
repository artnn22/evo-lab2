import time

import numpy as np

from .ga import evaluation_budget
from .matching import random_feasible
from .problem import evaluate


def run_random_search(config, data, seed):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    best_x = None
    best_f = float("inf")
    history = []
    n_evals = 0
    for generation in range(config["generations"] + 1):
        count = config["population_size"] if generation == 0 else (
            config["population_size"] - config["elite_size"])
        for _ in range(count):
            point = random_feasible(data, rng)
            assessment = evaluate(point, data)
            n_evals += 1
            if assessment["empty_seats"] < best_f:
                best_x, best_f = point.copy(), assessment["empty_seats"]
        history.append(best_f)
    assert n_evals == evaluation_budget(config)
    return {"best_x": best_x, "best_f": best_f, "history": history,
            "feasible_history": [1.0] * len(history), "n_evals": n_evals,
            "n_feasible": n_evals, "seconds": time.perf_counter() - started}
