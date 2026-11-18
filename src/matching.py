"""Собственный поиск паросочетания увеличивающими путями."""
from .problem import slot_groups


def assign_rooms(data, preferred=None, rng=None):
    result = [-1] * len(data["events"])
    for group in slot_groups(data):
        choices = {}
        for event in group:
            size = data["events"][event]["students"]
            available = [r for r, room in enumerate(data["rooms"]) if room["capacity"] >= size]
            if rng is not None:
                rng.shuffle(available)
            else:
                available.sort(key=lambda r: (data["rooms"][r]["capacity"] - size, r))
                if preferred is not None and preferred[event] in available:
                    available.remove(preferred[event])
                    available.insert(0, preferred[event])
            choices[event] = available

        owners = {}

        def augment(event, visited):
            for room in choices[event]:
                if room in visited:
                    continue
                visited.add(room)
                if room not in owners or augment(owners[room], visited):
                    owners[room] = event
                    return True
            return False

        order = group.copy()
        if rng is not None:
            rng.shuffle(order)
        for event in order:
            if not augment(event, set()):
                raise ValueError("Не существует допустимого назначения аудиторий")
        for room, event in owners.items():
            result[event] = room
    return result


def repair(point, data):
    return assign_rooms(data, preferred=point)


def random_feasible(data, rng):
    # Это случайная конструктивная выборка, не равномерная по всем решениям.
    return assign_rooms(data, rng=rng)
