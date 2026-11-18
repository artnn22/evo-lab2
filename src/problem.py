"""Входные данные, целевая функция и ограничения варианта 9."""
import json
from pathlib import Path

import numpy as np

LAB_FOLDER = Path(__file__).resolve().parent.parent
DATA_PATH = LAB_FOLDER / "data" / "instance.json"


def generate_data(seed=900):
    rng = np.random.default_rng(seed)
    rooms = [{"id": i + 1, "capacity": c}
             for i, c in enumerate([25, 35, 45, 60, 80, 100])]
    times = ["08:30-10:00", "10:15-11:45", "12:00-13:30",
             "13:45-15:15", "15:30-17:00", "17:15-18:45"]
    events = []
    for slot in range(6):
        # Пять разных комнат образуют свидетельство существования решения.
        selected = rng.choice(len(rooms), size=5, replace=False)
        for room in selected:
            events.append({"id": len(events) + 1, "slot": slot,
                           "students": int(rng.integers(15, rooms[room]["capacity"] + 1))})
    return {"variant": 9, "seed": seed, "times": times,
            "rooms": rooms, "events": events}


def load_data():
    with DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def slot_groups(data):
    groups = {}
    for i, event in enumerate(data["events"]):
        groups.setdefault(event["slot"], []).append(i)
    return list(groups.values())


def penalty_coefficient(data):
    return len(data["events"]) * max(r["capacity"] for r in data["rooms"]) + 1


def evaluate(point, data):
    """Номера комнат в генотипе — индексы 0..R-1, в таблицах — id 1..R."""
    if len(point) != len(data["events"]):
        raise ValueError("Число назначений не совпадает с числом занятий")
    empty = 0
    capacity_errors = 0
    conflicts = 0
    occupied = {}
    for i, room in enumerate(point):
        if not isinstance(room, (int, np.integer)) or not 0 <= room < len(data["rooms"]):
            raise ValueError("Некорректный номер аудитории")
        event = data["events"][i]
        difference = data["rooms"][room]["capacity"] - event["students"]
        empty += max(0, difference)
        capacity_errors += int(difference < 0)
        key = (event["slot"], room)
        # Каждая новая запись конфликтует со всеми предыдущими в этой комнате.
        conflicts += occupied.get(key, 0)
        occupied[key] = occupied.get(key, 0) + 1
    violations = capacity_errors + conflicts
    return {"empty_seats": empty, "capacity_errors": capacity_errors,
            "conflicts": conflicts, "feasible": violations == 0,
            "fitness": empty + penalty_coefficient(data) * violations}


def main():
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_text(json.dumps(generate_data(), ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print("Входные данные сохранены:", DATA_PATH)


if __name__ == "__main__":
    main()
