import csv
from pathlib import Path

# Определяем путь к основной папке проекта
ROOT = Path(__file__).resolve().parent.parent

# Пути к входным данным и результатам
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "paper-replay"

# Параметры обучения
GAMMA = 1.0
ALPHA = 0.1

# Функция для чтения данных из CSV-файла
def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


# Загружаем состояния и истории игр
states = load_csv(DATA_DIR / "states.csv")
games = load_csv(DATA_DIR / "paper-games.csv")

# Проверяем загруженные состояния
print("States:")
for state in states:
    print(state)

# Проверяем загруженные переходы
print("\nGame histories:")
for game in games:
    print(game)

print("\nNumber of states:", len(states))
print("Number of transitions:", len(games))


# MODEL-BASED LEARNING

# Словари для количества действий, переходов и наград
action_counts = {}
transition_counts = {}
reward_sums = {}

# Обрабатываем все переходы из четырех игр
for game in games:
    s = game["state"]
    a = int(game["action"])
    next_s = game["next_state"]
    r = float(game["reward"])

    pair = (s, a)
    triple = (s, a, next_s)

    # Считаем количество выполнений действия
    action_counts[pair] = action_counts.get(pair, 0) + 1

    # Считаем количество переходов
    transition_counts[triple] = transition_counts.get(triple, 0) + 1

    # Суммируем награды
    reward_sums[triple] = reward_sums.get(triple, 0) + r

# Словари для вероятностей и средних наград
P = {}
R = {}

print("\nMODEL-BASED LEARNING")

for triple in transition_counts:
    s, a, next_s = triple

    # Вычисляем вероятность перехода
    P[triple] = transition_counts[triple] / action_counts[(s, a)]

    # Вычисляем среднюю награду
    R[triple] = reward_sums[triple] / transition_counts[triple]

    action_name = "hit" if a == 1 else "stick"

    print(
        f"{s}, {action_name}, {next_s}: "
        f"N(s,a)={action_counts[(s, a)]}, "
        f"N(s,a,s')={transition_counts[triple]}, "
        f"P={P[triple]:.6f}, R={R[triple]:.6f}"
    )

# Показываем действия, которые не выполнялись
print("\nUnobserved state-action pairs:")

for state in states:
    s = state["label"]

    for a in [0, 1]:
        if (s, a) not in action_counts:
            action_name = "hit" if a == 1 else "stick"
            print(s, action_name, "- unobserved")


#Рассчитываем V_1 и V_2 по формуле Беллмана

# Создаем фиксированную стратегию
policy = {}

for state in states:
    s = state["label"]
    player_sum = int(state["player_sum"])

    policy[s] = 1 if player_sum < 20 else 0

# Начальные значения всех состояний равны нулю
V = {state["label"]: 0.0 for state in states}
V["END"] = 0.0

print("\nInitial values V0:")
print(V)

# Сохраняем результаты каждого прохода
sweep_results = [V.copy()]

# Выполняем два синхронных прохода
for sweep in range(1, 3):

    # Создаем отдельный словарь для новых значений
    new_V = {"END": 0.0}

    print(f"\nSweep {sweep}:")

    for state in states:
        s = state["label"]
        a = policy[s]
        value = 0.0

        for triple in P:
            current_s, action, next_s = triple

            if current_s == s and action == a:

                # Формула Беллмана
                term = P[triple] * (
                    R[triple] + GAMMA * V[next_s]
                )

                value += term

                print(
                    f"{s} -> {next_s}: "
                    f"{P[triple]:.6f} * "
                    f"({R[triple]} + {GAMMA} * "
                    f"{V[next_s]:.6f}) = {term:.6f}"
                )

        new_V[s] = value

    # Обновляем значения только после полного прохода
    V = new_V
    sweep_results.append(V.copy())

# Выводим итоговую таблицу
print("\nModel-based final table:")
print("State\tV0\t\tV1\t\tV2")

for state in states:
    s = state["label"]

    print(
        f"{s}\t"
        f"{sweep_results[0][s]:.6f}\t"
        f"{sweep_results[1][s]:.6f}\t"
        f"{sweep_results[2][s]:.6f}"
    )


# DIRECT EVALUATION (MONTE CARLO)

# Создаем словари для наград и количества посещений
returns = {state["label"]: [] for state in states}
visit_counts = {state["label"]: 0 for state in states}

# Разделяем переходы по номерам игр
episodes = {}

for game in games:
    game_id = int(game["game"])

    if game_id not in episodes:
        episodes[game_id] = []

    episodes[game_id].append(game)

print("\nDIRECT EVALUATION (MONTE CARLO)")

# Обрабатываем каждую игру один раз
for game_id, episode in episodes.items():

    print(f"\nGame {game_id}:")

    for i in range(len(episode)):
        s = episode[i]["state"]
        G = 0.0

        # Считаем суммарную награду до конца игры
        for j in range(i, len(episode)):
            r = float(episode[j]["reward"])
            G += GAMMA ** (j - i) * r

        # Сохраняем награду и увеличиваем число посещений
        returns[s].append(G)
        visit_counts[s] += 1

        print(f"State {s}: Return = {G}")

# Вычисляем среднее значение для каждого состояния
V_mc = {}

print("\nMonte Carlo final table:")
print("State\tReturns\t\tCount\tValue")

for state in states:
    s = state["label"]

    V_mc[s] = sum(returns[s]) / visit_counts[s]

    print(
        f"{s}\t{returns[s]}\t"
        f"{visit_counts[s]}\t{V_mc[s]:.6f}"
    )


# TD(0) LEARNING

# Создаем начальные значения состояний
V_td = {state["label"]: 0.0 for state in states}
V_td["END"] = 0.0

# Сохраняем результаты каждого прохода
td_results = [V_td.copy()]

# Сохраняем промежуточные обновления
td_updates = []

print("\nTD(0) LEARNING")

# Выполняем два прохода по четырем играм
for pass_num in range(1, 3):

    print(f"\nPass {pass_num}:")

    for game in games:
        s = game["state"]
        next_s = game["next_state"]
        r = float(game["reward"])

        # Получаем старое значение состояния
        old_value = V_td[s]

        # Получаем значение следующего состояния
        next_value = V_td[next_s]

        # Вычисляем целевое значение
        target = r + GAMMA * next_value

        # Обновляем значение по формуле TD(0)
        new_value = old_value + ALPHA * (target - old_value)

        V_td[s] = new_value

        # Сохраняем данные каждого обновления
        td_updates.append({
            "pass": pass_num,
            "game": int(game["game"]),
            "step": int(game["step"]),
            "state": s,
            "old_value": old_value,
            "reward": r,
            "next_value": next_value,
            "target": target,
            "new_value": new_value
        })

        # Выводим промежуточные расчеты
        print(
            f"Game {game['game']}, Step {game['step']}, "
            f"State {s}: "
            f"Old={old_value:.6f}, "
            f"Reward={r}, "
            f"Next={next_value:.6f}, "
            f"Target={target:.6f}, "
            f"New={new_value:.6f}"
        )

    # Сохраняем значения после каждого прохода
    td_results.append(V_td.copy())

# Выводим итоговую таблицу
print("\nTD(0) final table:")
print("State\tInitial\t\tPass 1\t\tPass 2")

for state in states:
    s = state["label"]

    print(
        f"{s}\t"
        f"{td_results[0][s]:.6f}\t"
        f"{td_results[1][s]:.6f}\t"
        f"{td_results[2][s]:.6f}"
    )


# Q-LEARNING

# Создаем Q-таблицу для обоих действий
Q = {}

for state in states:
    s = state["label"]

    for a in [0, 1]:
        Q[(s, a)] = 0.0

# Сохраняем результаты каждого прохода
q_results = [Q.copy()]

# Сохраняем промежуточные обновления
q_updates = []

print("\nQ-LEARNING")

# Выполняем два прохода по четырем играм
for pass_num in range(1, 3):

    print(f"\nPass {pass_num}:")

    for game in games:
        s = game["state"]
        a = int(game["action"])
        next_s = game["next_state"]
        r = float(game["reward"])

        # Получаем старое значение Q
        old_q = Q[(s, a)]

        # Для терминального состояния продолжение равно нулю
        if next_s == "END":
            max_next_q = 0.0
        else:
            # Выбираем максимум среди hit и stick
            max_next_q = max(
                Q[(next_s, 0)],
                Q[(next_s, 1)]
            )

        # Вычисляем целевое значение
        target = r + GAMMA * max_next_q

        # Обновляем Q по формуле Q-learning
        new_q = old_q + ALPHA * (target - old_q)

        Q[(s, a)] = new_q

        action_name = "hit" if a == 1 else "stick"

        # Сохраняем каждое обновление
        q_updates.append({
            "pass": pass_num,
            "game": int(game["game"]),
            "step": int(game["step"]),
            "state": s,
            "action": action_name,
            "old_q": old_q,
            "reward": r,
            "max_next_q": max_next_q,
            "target": target,
            "new_q": new_q
        })

        # Выводим промежуточные расчеты
        print(
            f"Game {game['game']}, Step {game['step']}, "
            f"State {s}, Action {action_name}: "
            f"Old={old_q:.6f}, "
            f"Reward={r}, "
            f"Max next Q={max_next_q:.6f}, "
            f"Target={target:.6f}, "
            f"New={new_q:.6f}"
        )

    # Сохраняем Q-таблицу после прохода
    q_results.append(Q.copy())

# Выводим итоговую Q-таблицу
print("\nQ-learning final table:")
print("State\tHit\t\tStick\t\tGreedy action")

for state in states:
    s = state["label"]

    q_hit = Q[(s, 1)]
    q_stick = Q[(s, 0)]

    # Определяем действие с максимальным Q
    if q_hit > q_stick:
        greedy = "hit"
    elif q_stick > q_hit:
        greedy = "stick"
    else:
        greedy = "tie"

    print(
        f"{s}\t{q_hit:.6f}\t"
        f"{q_stick:.6f}\t{greedy}"
    )


# PAPER VS PROGRAM COMPARISON

# Результаты ручных расчетов
paper_model = {
    "A": -1/3, "B": 0, "C": 0, "D": 0, "E": 0
}

paper_mc = {
    "A": -1/3, "B": 0, "C": 0, "D": 0, "E": 0
}

paper_td = {
    "A": -0.14003, "B": -0.0181,
    "C": 0, "D": 0, "E": 0
}

paper_q = {
    ("A", 1): -0.13922,
    ("A", 0): 0,
    ("B", 1): 0,
    ("B", 0): -0.0181,
    ("C", 1): 0,
    ("C", 0): 0,
    ("D", 1): 0,
    ("D", 0): 0,
    ("E", 1): 0,
    ("E", 0): 0
}

# Список для хранения результатов сравнения
comparison = []

# Функция для сравнения ручного и программного результата
def compare(method, state, stage, paper, program):
    difference = program - paper

    comparison.append({
        "method": method,
        "state_or_action": state,
        "stage": stage,
        "paper_result": paper,
        "program_result": program,
        "difference": difference
    })

# Сравниваем первые три метода
for state in states:
    s = state["label"]

    compare("Model-based", s, "Sweep 2",
            paper_model[s], sweep_results[2][s])

    compare("Monte Carlo", s, "Pass 1",
            paper_mc[s], V_mc[s])

    compare("TD(0)", s, "Pass 2",
            paper_td[s], V_td[s])

# Сравниваем Q-learning
for (s, a), paper_value in paper_q.items():
    action_name = "hit" if a == 1 else "stick"

    compare("Q-learning", f"{s}, {action_name}",
            "Pass 2", paper_value, Q[(s, a)])

# Выводим результаты сравнения
print("\nPAPER VS PROGRAM COMPARISON")

for row in comparison:
    print(
        f"{row['method']} | "
        f"{row['state_or_action']} | "
        f"{row['stage']} | "
        f"Paper={row['paper_result']:.6f} | "
        f"Program={row['program_result']:.6f} | "
        f"Difference={row['difference']:.10f}"
    )


# COMPARISON OF INTERMEDIATE RESULTS

# Ручные результаты первого прохода Model-based
for s in paper_model:
    compare("Model-based", s, "Sweep 1",
            paper_model[s], sweep_results[1][s])

# Ручные результаты всех 16 обновлений TD(0)
paper_td_updates = [
    0, 0.1, -0.1, 0, 0, 0, -0.08, -0.01,
    -0.073, 0.091, -0.1657, 0, 0, 0,
    -0.14003, -0.0181
]

# Сравниваем каждое обновление TD(0)
for i, update in enumerate(td_updates):
    stage = (
        f"Pass {update['pass']}, "
        f"Game {update['game']}, "
        f"Step {update['step']}"
    )

    compare(
        "TD(0)", update["state"], stage,
        paper_td_updates[i], update["new_value"]
    )

# Ручные результаты всех 16 обновлений Q-learning
paper_q_updates = [
    0, 0.1, -0.1, 0, 0, 0, -0.08, -0.01,
    -0.072, 0.091, -0.1648, 0, 0, 0,
    -0.13922, -0.0181
]

# Сравниваем каждое обновление Q-learning
for i, update in enumerate(q_updates):
    stage = (
        f"Pass {update['pass']}, "
        f"Game {update['game']}, "
        f"Step {update['step']}"
    )

    compare(
        "Q-learning",
        f"{update['state']}, {update['action']}",
        stage,
        paper_q_updates[i],
        update["new_q"]
    )

# Проверяем все результаты с учетом погрешности
mismatches = [
    row for row in comparison
    if abs(row["difference"]) > 1e-9
]

print("\nCOMPARISON SUMMARY")
print("Total comparisons:", len(comparison))
print("Matching results:", len(comparison) - len(mismatches))
print("Mismatches:", len(mismatches))


# SAVE RESULTS

# Создаем папку для результатов
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Функция для сохранения таблиц в CSV
def save_csv(filename, rows):
    path = OUTPUT_DIR / filename

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {filename}")


# Сохраняем сравнение ручных и программных расчетов
save_csv("paper_vs_program.csv", comparison)

# Сохраняем все обновления TD и Q-learning
save_csv("td_updates.csv", td_updates)
save_csv("q_updates.csv", q_updates)

# Сохраняем итоговые значения всех четырех методов
final_results = []

for state in states:
    s = state["label"]

    final_results.append({
        "state": s,
        "model_based": sweep_results[2][s],
        "monte_carlo": V_mc[s],
        "td": V_td[s],
        "q_hit": Q[(s, 1)],
        "q_stick": Q[(s, 0)]
    })

save_csv("final_results.csv", final_results)

print("\nPaper replay completed.")