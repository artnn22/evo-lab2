"""Сборка обоих форматов отчёта из фактических CSV; оптимизация здесь не запускается."""
import csv
import html
import json
import os
import re
import statistics
from pathlib import Path

from .problem import LAB_FOLDER, evaluate, load_data, slot_groups

os.environ.setdefault("MPLCONFIGDIR", str(LAB_FOLDER / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS = LAB_FOLDER / "results"
LABELS = {"penalty_reassign": "Штраф + переназначение",
          "repair_reassign": "Ремонт + переназначение",
          "repair_swap": "Ремонт + обмен", "random": "Случайный поиск"}


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(value, digits=2):
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(header, rows):
    lines = ["| " + " | ".join(map(str, header)) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def draw_scheme():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set(xlim=(0, 10), ylim=(-0.2, 10))
    ax.axis("off")
    boxes = [(5, 9, "Создать и оценить 60 особей; сохранить допустимый архив"),
             (5, 7.3, "200 поколений: сохранить 2 лучшие особи"),
             (5, 5.6, "Турнир из 3 → равномерный кроссовер (0,9)"),
             (5, 3.9, "Мутация (0,2): переназначение или обмен"),
             (5, 2.2, "Ремонт, если включён → оценить 58 потомков"),
             (5, 0.5, "Обновить допустимый архив и историю; вернуть лучшее")]
    for i, (x, y, text) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((0.2, y - 0.48), 9.6, 0.96,
                                   boxstyle="round,pad=0.06", facecolor="#f2f2f2", edgecolor="#595959"))
        ax.text(x, y, text, ha="center", va="center", fontsize=10)
        if i:
            ax.annotate("", xy=(5, y + 0.55), xytext=(5, boxes[i-1][1] - 0.55),
                        arrowprops={"arrowstyle": "->", "color": "#595959"})
    fig.tight_layout(pad=0.3)
    fig.savefig(RESULTS / "algorithm.png", dpi=180)
    plt.close(fig)


def build_markdown():
    data = load_data()
    summary = read_csv("summary.csv")
    best = json.loads((RESULTS / "best.json").read_text(encoding="utf-8"))
    point = best["best_x"]
    assessment = evaluate(point, data)
    assert assessment["feasible"] and assessment["empty_seats"] == best["best_f"]
    stats_rows = [[LABELS[r["config"]], r["best"], number(r["mean"]), number(r["median"]),
                   number(r["std"]), r["worst"]] for r in summary]
    stats = markdown_table(["Метод", "Лучшее", "Среднее", "Медиана", "Ст. откл.", "Худшее"], stats_rows)
    operation_rows = []
    first_hit_rows = []
    for row in summary:
        name = row["config"]
        trajectories = read_csv("traj_" + name + ".csv")
        hits = []
        for seed in range(1000, 1020):
            generations = [int(r["generation"]) for r in trajectories
                           if int(r["seed"]) == seed and float(r["best_f"]) == best["best_f"]]
            if generations:
                hits.append(min(generations))
        first_hit_rows.append([LABELS[name], str(len(hits)) + "/20",
                               number(statistics.mean(hits)) if hits else "не достигнуто",
                               f"{min(hits)}–{max(hits)}" if hits else "—"])
        operation_rows.append([LABELS[name], number(100 * float(row["feasible_ratio"])) + "%",
                               row["n_evals"], number(row["mean_seconds"], 3)])
    operations = markdown_table(["Метод", "Допустимые оценки", "Оценок / запуск", "Время, с"], operation_rows)
    hits_table = markdown_table(["Метод", "Достигли 535", "Среднее поколение", "Диапазон"], first_hit_rows)

    other = point.copy()
    other[0], other[2] = other[2], other[0]
    too_small = point.copy()
    too_small[4] = 4
    conflict = point.copy()
    conflict[0] = conflict[1]
    example_rows = []
    for label, assignment in [("A: лучшее расписание", point), ("B: обмен комнат занятий 1 и 3", other),
                               ("C: занятие 5 → аудитория 5", too_small),
                               ("D: занятие 1 → аудитория занятия 2", conflict)]:
        value = evaluate(assignment, data)
        example_rows.append([label, value["capacity_errors"], value["conflicts"],
                             "да" if value["feasible"] else "нет"])
    examples = markdown_table(["Пример", "Вместимость", "Конфликты", "Допустим"], example_rows)
    assert evaluate(other, data)["feasible"]
    assert not evaluate(too_small, data)["feasible"] and not evaluate(conflict, data)["feasible"]
    (RESULTS / "examples.json").write_text(json.dumps(
        {"A": point, "B": other, "C": too_small, "D": conflict}, indent=2) + "\n", encoding="utf-8")

    schedule = read_csv("best_schedule.csv")
    schedule_table = markdown_table(["Занятие", "Время", "Студентов", "Аудитория", "Мест", "Пустых"],
                                     [[r[k] for k in ["event", "time", "students", "room", "capacity", "empty_seats"]]
                                      for r in schedule])
    # Нижняя граница без оптимизатора: пять различных комнат, с обязательной
    # комнатой 100 мест в каждом интервале, где есть группа больше 80 человек.
    capacities = sorted(r["capacity"] for r in data["rooms"])
    bound_capacity = 0
    for group in slot_groups(data):
        base = sum(capacities[:len(group)])
        largest_group = max(data["events"][i]["students"] for i in group)
        if largest_group > capacities[len(group) - 1]:
            required = min(c for c in capacities if c >= largest_group)
            base += required - capacities[len(group) - 1]
        bound_capacity += base
    students = sum(e["students"] for e in data["events"])
    assert bound_capacity == 1510 and students == 975 and best["best_f"] == bound_capacity - students
    return f"""# Лабораторная работа № 2

## Генетический алгоритм для назначения аудиторий

Дисциплина «Эволюционное программирование и генетические алгоритмы». Вариант 9.
Программа написана на Python 3. Генетический алгоритм, операторы и поиск
паросочетания реализованы самостоятельно. NumPy используется для случайных
чисел, Matplotlib — для графиков, ReportLab — для PDF.

## 1. Задача

Нужно назначить аудитории 30 занятиям. В каждом из шести непересекающихся
временных интервалов проходят пять занятий. Доступны шесть аудиторий на
25, 35, 45, 60, 80 и 100 человек. Время фиксировано; переносить занятия нельзя.

Аудитория должна вмещать всю группу. В один интервал аудиторию можно назначить
только одному занятию. В разных интервалах её можно использовать повторно.
Каждое занятие относится к отдельной группе; преподаватели, оснащение,
переходы между корпусами и предпочтения по времени в модель не входят.

Пространство решений — все списки из 30 целых чисел от 0 до 5; xᵢ — индекс
аудитории занятия i. В таблицах аудитории имеют номера от 1 до 6. Обозначим число студентов
через sᵢ, вместимость комнаты через cᵣ, интервал занятия через tᵢ.

Для допустимого расписания минимизируем сумму пустых мест F(x) = Σᵢ (c[xᵢ] − sᵢ).

Для каждого занятия требуется c[xᵢ] ≥ sᵢ, а для каждой пары одновременных
занятий (tᵢ = tⱼ) требуется xᵢ ≠ xⱼ. В каждом интервале 5 × 4 / 2 = 10 таких пар,
поэтому всего 60 попарных ограничений конфликтности.
Пустые места суммируются по занятиям, поэтому комната учитывается заново
в каждом интервале. Минимизация пустых мест уточняет постановку варианта.

Входные данные синтетические, seed 900. Для каждого интервала сначала выбираются
пять различных аудиторий, затем размер каждой группы равномерно выбирается
из целых чисел от 15 до вместимости выбранной аудитории включительно.
Это гарантирует существование допустимого решения. Исходное назначение
при генерации не сохраняется и не передаётся алгоритмам. Все данные приложены
в data/instance.json, процедура генерации — в src/problem.py.

## 2. Представление решения и алгоритм

Одна особь — список из 30 номеров аудиторий. Генотип прямо задаёт назначение;
фенотип — расписание с временем, числом студентов и вместимостью. Декодирование
заменяет индексы комнат их номерами и добавляет сведения из входных данных.
Дискретный ген не изменяет время занятия и не требует округления.

![Схема генетического алгоритма](results/algorithm.png)

Общие параметры трёх конфигураций:

{markdown_table(["Параметр", "Значение"], [["Размер популяции", 60], ["Число поколений", 200], ["Элита", 2], ["Размер турнира", 3], ["Вероятность кроссовера", "0,9"], ["Вероятность мутации потомка", "0,2"], ["Запуски и seed", "20; 1000–1019"]])}

Турнир выбирает три индекса с возвращением; побеждает меньший фитнес.
Равномерный кроссовер независимо выбирает каждое назначение от одного из
родителей с вероятностью 1/2. С вероятностью 0,9 создаётся такой потомок,
иначе берётся копия первого родителя. Затем возможна мутация.

Две лучшие особи переходят без повторной оценки. Остальные 58 мест занимают
потомки. Число оценок на запуск: 60 + 200 × 58 = 11 660.
Остановка — строго после 200 поколений, даже если улучшений больше нет.
Лучшее допустимое назначение хранится отдельно от текущей популяции.

## 3. Ограничения, операторы и примеры

В штрафном варианте H(x) — число занятий с недостаточной вместимостью плюс
число конфликтующих пар. Три занятия в одной комнате в одно время дают
три конфликтующие пары. Фитнес:

Q(x) = Σᵢ max(0, c[xᵢ] − sᵢ) + 3001 × H(x).

Для допустимого решения сумма не превышает 30 × 100 = 3000. Одно нарушение
даёт не менее 3001. Поэтому любое допустимое решение лучше любого
недопустимого. История качества использует F только допустимых решений,
а не штрафной фитнес. Начальная популяция включает одну случайную допустимую
особь и 59 случайных векторов назначений.

Ремонт строит полное паросочетание отдельно для каждого интервала. Ребро
связывает занятие с аудиторией достаточной вместимости. Сначала пробуется
назначение потомка, затем комнаты по возрастанию избытка вместимости.
Занятия идут в порядке данных; занятая комната освобождается поиском
увеличивающего пути. Если полного назначения нет, программа выдаёт ValueError.
У ГА с ремонтом начальная особь случайная допустимая, остальные 59 векторов
ремонтируются до оценки. Все потомки также ремонтируются перед оценкой.

Переназначение меняет комнату одного случайного занятия и позволяет
задействовать свободную аудиторию. Обмен меняет комнаты двух занятий
одного интервала и сохраняет число назначений в каждую комнату. Он может
нарушить вместимость, если одна из групп крупнее новой аудитории.
При кроссовере двух допустимых родителей вместимость сохраняется для
каждого занятия, но аудитория может достаться двум занятиям одновременно.
Ремонт восстанавливает оба ограничения и может менять исходные назначения.

Ниже четыре полных решения заданы через расписание из раздела 6 и одно
изменение. Их векторы также сохранены в results/examples.json.

{examples}

В C группа из 84 человек попадает в комнату на 80 мест без временного конфликта.
В D аудиторию используют два занятия одновременно; вместимости хватает.
В A и B нет нарушений, целевая функция равна 535.

## 4. Проведённые эксперименты

Проведены четыре серии по 20 запусков. Конфигурации ГА: штрафы и переназначение;
ремонт и переназначение; ремонт и обмен. Первая пара изолирует способ обработки
ограничений, вторая — оператор мутации. Остальные параметры совпадают.

Базовый случайный поиск независимо строит допустимые назначения через
паросочетание со случайным порядком занятий и рёбер. Распределение не является
равномерным по всем допустимым назначениям. Бюджет также равен 11 660 оценкам.
Для записи истории первые 60 оценок образуют поколение 0, затем идут 200
блоков по 58. Seed каждого метода — 1000–1019. Одинаковый seed не означает
одинаковые случайные решения у разных алгоритмов.

В таблице показано число пустых мест лучшего допустимого решения каждого
запуска. Стандартное отклонение — выборочное, с делителем 19.

{stats}

Все три ГА достигли значения 535 в каждом запуске. Случайный поиск получил
минимум 570, среднее 600 и максимум 625. В этой серии ГА лучше выбранного
случайного поиска по итоговому качеству при равном числе полных оценок.

{operations}

Доля допустимости считается по всем новым оценённым кандидатам, включая
начальную популяцию. Копии элиты повторно не учитываются. У ремонта это
допустимость после восстановления; доля допустимых сырых потомков не измерялась.
Время включает создание решений, ремонт и оценку, но не запись таблиц и графиков.
В таблице приведено среднее время сохранённой серии запусков.

{hits_table}

В каждой конфигурации с ремонтом оптимум найден в начальной популяции
в 7 из 20 запусков. Обе мутации дали одинаковое итоговое качество.
Штрафный ГА достигал оптимума после большего числа оценок, а весь запуск
из 200 поколений в сохранённой серии занимал меньше времени.

## 5. Графики

Линии показывают среднее лучшего допустимого значения по 20 запускам;
полупрозрачные области — диапазон от минимума до максимума. Начальная точка
соответствует 60 оценкам. Каждый последующий шаг — ещё 58 оценок.

![Сходимость четырёх методов](results/convergence.png)

![Сравнение итогового качества](results/comparison_box.png)

На диаграмме размаха сравниваются финальные значения. У трёх ГА все результаты
совпали, поэтому прямоугольники сжались в линии на уровне 535.

![Доля допустимых оценок](results/feasibility.png)

Три линии на уровне 1 совпадают: это два ГА с ремонтом и случайный допустимый
поиск. Штрафный ГА сначала оценивает преимущественно недопустимые назначения,
затем накапливает допустимые. В среднем итоговая доля допустимых оценок
составила 79,08%. Это не доля допустимых особей последнего поколения.

Равный бюджет учитывает каждую полную оценку после обработки ограничений,
включая дубликаты. Проверки вместимости и поиск паросочетания внутри ремонта
не считаются дополнительными полными оценками. Они требуют времени, поэтому
равенство бюджетов не обеспечивает равенства вычислительных затрат.

Штрафы позволяют оценивать недопустимые назначения. При ремонте сначала
пробуется исходная аудитория, затем подходящие аудитории от меньшей к большей.
Этот порядок может способствовать раннему выходу на хорошие решения.
Сравнение включает и обработку начальной популяции, и обработку потомков.

<!-- pagebreak -->

## 6. Лучшее найденное расписание

Первый из лучших результатов: {LABELS[best['method']]}, seed {best['seed']}.
Суммарно 535 пустых мест, нарушений вместимости и конфликтов нет.
Каждый номер занятия обозначает отдельную группу. Расписание сохранено
в results/best_schedule.csv, генотип и seed — в results/best.json.

{schedule_table}

Вместимость суммарно по всем занятиям — 1510 мест, численность групп — 975.
Пустые места: 1510 − 975 = 535. Эти суммы относятся ко всем интервалам;
975 — сумма численностей отдельных групп, не одновременная загрузка.

## 7. Оценка качества и вывод

Для данного экземпляра можно доказать оптимальность 535. В каждом интервале
заняты пять различных аудиторий. Без дополнительных условий их суммарная
вместимость не меньше 25 + 35 + 45 + 60 + 80 = 245.

В первом и третьем интервалах есть группа из 84 человек. Для неё обязательна
комната на 100 мест. Поэтому в этих двух интервалах вместимость не меньше
25 + 35 + 45 + 60 + 100 = 265. Получаем нижнюю границу суммарной вместимости:
4 × 245 + 2 × 265 = 1510. Размеры групп суммарно равны 975,
следовательно, любое допустимое решение имеет F(x) ≥ 1510 − 975 = 535.
Предъявленное расписание достигает границы. Значит, для этих данных оно оптимально.

Три конфигурации ГА получили этот оптимум во всех 20 запусках. Ремонт обеспечил
допустимость каждой оценки и быстрое достижение результата; штрафный подход
также нашёл оптимум, несмотря на недопустимые промежуточные решения.
Обмен и переназначение не различились по конечному качеству. Случайная
конструктивная выборка при том же числе оценок оказалась хуже.

Задача с фиксированным временем распадается на независимые назначения по
интервалам и может решаться точными методами минимальной стоимости.
Генетический алгоритм здесь нужен для изучения дискретных операторов и
ограничений. Эксперимент не доказывает его преимущество над точным решением
или эффективность на более крупных и связанных расписаниях. У этой целевой
функции многие перестановки одних и тех же пяти комнат имеют одинаковую цену;
поэтому значение 535 не означает единственность расписания.

## 8. Повторный запуск

Команды для запуска из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m tests.test_problem
python3 -m src.experiment
python3 -m src.report
```

Данные уже приложены. Команда python3 -m src.problem повторно создаёт их
с seed 900. Настройки находятся в configs/, число запусков — RUNS в начале
src/experiment.py. Для повторения конкретного запуска можно вызвать
run_ga(config, load_data(), seed), загрузив конфигурацию через load_config().

Проверки охватывают вместимость, пары конфликтов, повторное использование
комнаты в другое время, ремонт и отсутствие решения, сравнение с полным
перебором на малых задачах, сохранение родителей, воспроизводимость по seed,
монотонность архива и равенство бюджетов. Измеренное время не является
воспроизводимым полем. После изменения входных данных или параметров
нужно заново проверить выводы отчёта.
"""


def find_fonts():
    candidates = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf")]
    for normal, bold in candidates:
        if Path(normal).exists() and Path(bold).exists():
            return normal, bold
    raise RuntimeError("Для PDF установите Arial или DejaVu Sans с кириллицей")


def build_pdf(markdown):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                   SimpleDocTemplate, Spacer, Table, TableStyle)
    normal, bold = find_fonts()
    pdfmetrics.registerFont(TTFont("Body", normal))
    pdfmetrics.registerFont(TTFont("BodyBold", bold))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    body = ParagraphStyle("body", fontName="Body", fontSize=9.4, leading=13.6, spaceAfter=7,
                          textColor=colors.black)
    heading = ParagraphStyle("heading", parent=body, fontName="BodyBold", fontSize=15,
                             leading=18, spaceBefore=10, spaceAfter=8, keepWithNext=True)
    title = ParagraphStyle("title", parent=heading, fontSize=22, leading=26, spaceBefore=0)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.1, leading=10.5, spaceAfter=0)
    header_cell = ParagraphStyle("header_cell", parent=cell, fontName="BodyBold")
    code = ParagraphStyle("code", parent=body, fontSize=8, leading=12, leftIndent=10,
                          backColor=colors.HexColor("#f2f2f2"))
    margin = 56
    width = A4[0] - 2 * margin - 12  # Внутренние отступы Frame по 6 pt.
    story = []

    def para(text, style=body):
        value = html.escape(text)
        for symbol, index in (("ᵢ", "i"), ("ⱼ", "j"), ("ᵣ", "r")):
            value = value.replace(symbol, f"<sub>{index}</sub>")
        return Paragraph(value, style)

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(para(line[2:], title))
        elif line.startswith("## "):
            story.append(para(line[3:], heading))
        elif line.startswith("!["):
            path = re.search(r"\]\((.*?)\)", line).group(1)
            image = Image(str(LAB_FOLDER / path))
            target_width = min(width, 420)
            target_height = image.imageHeight / image.imageWidth * target_width
            image.drawWidth, image.drawHeight = target_width, target_height
            story.append(image)
            story.append(Spacer(1, 8))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                values = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", v) for v in values):
                    style = cell if rows else header_cell
                    rows.append([para(v, style) for v in values])
                i += 1
            if len(rows[0]) == 6 and len(rows) > 25:
                col_widths = [48, 111, 78, 78, 78, width - 393]
            elif len(rows[0]) == 6:
                col_widths = [160] + [(width - 160) / 5] * 5
            elif len(rows[0]) == 4:
                col_widths = [205] + [(width - 205) / 3] * 3
            else:
                col_widths = [width / len(rows[0])] * len(rows[0])
            table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#777777")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d9d9d9")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([KeepTogether([table]) if len(rows) <= 10 else table, Spacer(1, 9)])
            continue
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(para(lines[i], code))
                i += 1
            story.append(KeepTogether(block))
            story.append(Spacer(1, 7))
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                paragraph.append(lines[i].strip())
                i += 1
            story.append(para(" ".join(paragraph)))
            continue
        i += 1

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d9d9d9"))
        canvas.line(margin, 42, A4[0] - margin, 42)
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.HexColor("#595959"))
        canvas.drawString(margin, 29, "Лабораторная работа № 2 · Генетический алгоритм")
        canvas.drawRightString(A4[0] - margin, 29, str(document.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(LAB_FOLDER / "report.pdf"), pagesize=A4,
                            leftMargin=margin, rightMargin=margin, topMargin=48, bottomMargin=56,
                            title="Лабораторная работа № 2 — Назначение аудиторий",
                            author="Лабораторная работа")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    draw_scheme()
    markdown = build_markdown()
    (LAB_FOLDER / "report.md").write_text(markdown, encoding="utf-8")
    build_pdf(markdown)
    print("Отчёт обновлён: report.md и report.pdf")


if __name__ == "__main__":
    main()
