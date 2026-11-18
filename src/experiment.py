"""Одна команда повторяет все 80 запусков и сохраняет результаты."""
import csv
import json
import os
import statistics
from pathlib import Path

LAB_FOLDER = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(LAB_FOLDER / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ga import run_ga
from .problem import evaluate, load_data
from .random_search import run_random_search

RUNS = 20
RESULTS = LAB_FOLDER / "results"
NAMES = ["penalty_reassign", "repair_reassign", "repair_swap", "random"]
LABELS = ["Штраф + переназначение", "Ремонт + переназначение", "Ремонт + обмен", "Случайный поиск"]


def load_config(name):
    with (LAB_FOLDER / "configs" / (name + ".json")).open(encoding="utf-8") as file:
        return json.load(file)


def write_csv(name, header, rows):
    with (RESULTS / name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def schedule_rows(point, data):
    rows = []
    for i, event in enumerate(data["events"]):
        room = data["rooms"][point[i]]
        rows.append([event["id"], data["times"][event["slot"]], event["students"],
                     room["id"], room["capacity"], room["capacity"] - event["students"]])
    return rows


def plot_results(series):
    colors = ["#bf5b39", "#276b9c", "#388453", "#8264a2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, results in zip(LABELS, colors, series):
        histories = [r["history"] for r in results]
        low = [min(v) for v in zip(*histories)]
        mean = [statistics.mean(v) for v in zip(*histories)]
        high = [max(v) for v in zip(*histories)]
        ax.plot(mean, label=label, color=color)
        ax.fill_between(range(len(mean)), low, high, color=color, alpha=0.10)
    ax.set(xlabel="Поколение / равный блок оценок", ylabel="Лучшее допустимое число пустых мест",
           title="Сходимость: среднее и диапазон минимум–максимум, 20 запусков")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([[r["best_f"] for r in results] for results in series])
    ax.set_xticks(range(1, 5), ["Штраф +\nпереназначение", "Ремонт +\nпереназначение",
                                "Ремонт +\nобмен", "Случайный\nпоиск"])
    ax.set(ylabel="Пустые места", title="Итоговое качество при 11 660 оценках на запуск")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_box.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, results in zip(LABELS, colors, series):
        mean = [statistics.mean(v) for v in zip(*(r["feasible_history"] for r in results))]
        ax.plot(mean, label=label, color=color)
    ax.set(xlabel="Поколение / равный блок оценок", ylabel="Накопленная доля допустимых оценок",
           ylim=(-0.02, 1.06), title="Допустимость после обработки ограничений")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "feasibility.png", dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    data = load_data()
    series = []
    summary = []
    overall = None
    for name in NAMES:
        config = load_config("repair_reassign" if name == "random" else name)
        results = []
        rows = []
        trajectories = []
        for seed in range(config["seed_start"], config["seed_start"] + RUNS):
            method = run_random_search if name == "random" else run_ga
            result = method(config, data, seed)
            assert evaluate(result["best_x"], data)["feasible"]
            results.append(result)
            ratio = result["n_feasible"] / result["n_evals"]
            rows.append([name, seed, result["best_f"], json.dumps(result["best_x"]),
                         result["n_evals"], result["n_feasible"], ratio, result["seconds"]])
            for generation, (value, feasible) in enumerate(zip(result["history"], result["feasible_history"])):
                evaluations = config["population_size"] + generation * (
                    config["population_size"] - config["elite_size"])
                trajectories.append([name, seed, generation, evaluations, value, feasible])
            if overall is None or result["best_f"] < overall["best_f"]:
                overall = {"method": name, "seed": seed, "best_f": result["best_f"],
                           "best_x": result["best_x"]}
            print(f"{name}, seed {seed}: {result['best_f']}, допустимость {ratio:.3f}", flush=True)
        write_csv("runs_" + name + ".csv",
                  ["config", "seed", "best_f", "best_x", "n_evals", "n_feasible", "feasible_ratio", "seconds"], rows)
        write_csv("traj_" + name + ".csv",
                  ["config", "seed", "generation", "n_evals", "best_f", "feasible_ratio"], trajectories)
        values = [r["best_f"] for r in results]
        summary.append([name, RUNS, min(values), statistics.mean(values), statistics.median(values),
                        statistics.stdev(values), max(values),
                        statistics.mean(r["n_feasible"] / r["n_evals"] for r in results),
                        results[0]["n_evals"], statistics.mean(r["seconds"] for r in results)])
        best = min(results, key=lambda r: r["best_f"])
        write_csv("schedule_" + name + ".csv",
                  ["event", "time", "students", "room", "capacity", "empty_seats"],
                  schedule_rows(best["best_x"], data))
        series.append(results)
    write_csv("summary.csv", ["config", "n_runs", "best", "mean", "median", "std", "worst",
                              "feasible_ratio", "n_evals", "mean_seconds"], summary)
    (RESULTS / "best.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv("best_schedule.csv", ["event", "time", "students", "room", "capacity", "empty_seats"],
              schedule_rows(overall["best_x"], data))
    plot_results(series)
    print("Все таблицы, расписания и графики сохранены в results.")


if __name__ == "__main__":
    main()
