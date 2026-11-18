"""Проверки предметной модели и поиска без дополнительного тестового фреймворка."""
import itertools

import numpy as np

from src.experiment import load_config
from src.ga import crossover, evaluation_budget, mutate, run_ga
from src.matching import assign_rooms, random_feasible, repair
from src.problem import evaluate, generate_data, load_data, penalty_coefficient
from src.random_search import run_random_search


def tiny(capacities, students, slots=None):
    return {"rooms": [{"id": i + 1, "capacity": c} for i, c in enumerate(capacities)],
            "events": [{"id": i + 1, "students": s, "slot": slots[i] if slots else 0}
                       for i, s in enumerate(students)]}


def test_model():
    data = tiny([25, 40], [20, 30])
    assert evaluate([0, 1], data) == {"empty_seats": 15, "capacity_errors": 0,
                                       "conflicts": 0, "feasible": True, "fitness": 15}
    bad = evaluate([0, 0], data)
    assert bad["capacity_errors"] == 1 and bad["conflicts"] == 1
    assert bad["fitness"] == 5 + 2 * penalty_coefficient(data)
    assert evaluate([0, 0], tiny([25], [20, 20], [0, 1]))["feasible"]
    assert evaluate([0, 0, 0], tiny([25], [20, 20, 20]))["conflicts"] == 3
    for point in [[0], [0, 2], [-1, 1]]:
        try:
            evaluate(point, data)
        except ValueError:
            pass
        else:
            raise AssertionError("Некорректное назначение принято")
    feasible_scores, invalid_scores = [], []
    for point in itertools.product(range(2), repeat=2):
        info = evaluate(point, data)
        (feasible_scores if info["feasible"] else invalid_scores).append(info["fitness"])
    assert max(feasible_scores) < min(invalid_scores)


def test_matching():
    # Полный перебор подтверждает существование решения независимо от ремонта.
    for sizes in itertools.product([15, 25, 35], repeat=3):
        data = tiny([20, 30, 40], sizes)
        exists = any(evaluate(list(p), data)["feasible"]
                     for p in itertools.product(range(3), repeat=3))
        try:
            result = repair([0, 0, 0], data)
        except ValueError:
            assert not exists
        else:
            assert exists and evaluate(result, data)["feasible"]
    # Требуется вытеснить первое занятие из предпочитаемой большой комнаты.
    data = tiny([20, 40], [15, 35])
    assert repair([1, 1], data) == [0, 1]
    for data in [tiny([20], [25]), tiny([30], [15, 15])]:
        try:
            assign_rooms(data)
        except ValueError:
            pass
        else:
            raise AssertionError("Отсутствие паросочетания не обнаружено")


def test_data_and_operators():
    data = load_data()
    assert data == generate_data(900)
    assert len(data["events"]) == 30 and len(data["rooms"]) == 6
    rng = np.random.default_rng(1)
    config = load_config("repair_reassign")
    first = random_feasible(data, rng)
    second = random_feasible(data, rng)
    saved_first, saved_second = first.copy(), second.copy()
    for method in ["reassign", "swap"]:
        config["mutation"] = method
        config["mutation_prob"] = 1
        for _ in range(40):
            child = mutate(crossover(first, second, rng), config, data, rng)
            old_child = child.copy()
            fixed = repair(child, data)
            assert child == old_child and evaluate(fixed, data)["feasible"]
            assert all(0 <= r < len(data["rooms"]) for r in child)
    assert first == saved_first and second == saved_second
    assert evaluate(first, data)["feasible"] and evaluate(second, data)["feasible"]


def deterministic_part(result):
    return {k: v for k, v in result.items() if k != "seconds"}


def test_search():
    data = load_data()
    for name in ["penalty_reassign", "repair_reassign", "repair_swap", "random"]:
        config = load_config("repair_reassign" if name == "random" else name)
        assert evaluation_budget(config) == 11660
        config["generations"] = 5
        method = run_random_search if name == "random" else run_ga
        result = method(config, data, 1000)
        repeat = method(config, data, 1000)
        assert deterministic_part(result) == deterministic_part(repeat)
        assert result["n_evals"] == evaluation_budget(config)
        assert len(result["history"]) == 6
        info = evaluate(result["best_x"], data)
        assert info["feasible"] and info["empty_seats"] == result["best_f"]
        assert all(b <= a for a, b in zip(result["history"], result["history"][1:]))
        if name != "penalty_reassign":
            assert result["n_feasible"] == result["n_evals"]
        assert all(0 <= v <= 1 for v in result["feasible_history"])


if __name__ == "__main__":
    test_model()
    test_matching()
    test_data_and_operators()
    test_search()
    print("Все проверки пройдены.")
