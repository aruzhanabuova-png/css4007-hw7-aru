import gymnasium as gym
import csv
import random
from pathlib import Path
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt

# Определяем пути к папкам проекта
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "simulated-games"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Задаем параметры обучения и количество игр
GAMMA = 1.0
ALPHA = 0.1
EPSILON = 0.2

SEEDS = [11, 22, 33]
FIXED_GAMES = 10000
TRAIN_GAMES = 50000
EVAL_GAMES = 10000
MAX_ACTIONS = 100

# Создаем среду Blackjack с правилами из задания
env = gym.make(
    "Blackjack-v1",
    natural=False,
    sab=False
)

# Функция выбирает действие по фиксированной стратегии
def fixed_policy(state):
    player_sum, dealer_showing, usable_ace = state

    # Берем карту при сумме меньше 20, иначе останавливаемся
    if player_sum < 20:
        return 1
    else:
        return 0

# Функция генерирует и сохраняет игры с фиксированной стратегией
def collect_fixed_games(seed, num_games):
    histories = []
    cutoffs = 0

    # Устанавливаем seed только для первой игры
    state, info = env.reset(seed=seed)

    for game in range(1, num_games + 1):

        # Начинаем следующую игру без повторной установки seed
        if game > 1:
            state, info = env.reset()

        for step in range(1, MAX_ACTIONS + 1):

            # Выбираем и выполняем действие
            action = fixed_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Проверяем, была ли игра принудительно прервана
            cutoff = truncated or (
                step == MAX_ACTIONS and not terminated
            )

            # Сохраняем информацию о каждом переходе
            histories.append({
                "seed": seed,
                "game": game,
                "step": step,
                "state": state,
                "action": action,
                "reward": reward,
                "next_observation": next_state,
                "next_state": "END" if terminated else next_state,
                "terminated": terminated,
                "cutoff": cutoff
            })

            # Завершаем цикл, если игра закончилась
            if terminated or cutoff:
                if cutoff:
                    cutoffs += 1
                break

            state = next_state

    # Сохраняем историю игр в CSV-файл
    path = OUTPUT_DIR / f"fixed_games_seed_{seed}.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(histories[0].keys())
        )
        writer.writeheader()
        writer.writerows(histories)

    # Выводим результаты генерации
    print(f"\nSeed: {seed}")
    print("Games:", num_games)
    print("Transitions:", len(histories))
    print("Cut-offs:", cutoffs)
    print("Saved:", path.name)

    return histories


# Генерируем 10000 игр для каждого seed
fixed_histories = {}

for seed in SEEDS:
    fixed_histories[seed] = collect_fixed_games(seed, FIXED_GAMES)

env.close()

# Функция загружает сохраненные истории игр из CSV
def load_fixed_games(seed):
    path = OUTPUT_DIR / f"fixed_games_seed_{seed}.csv"
    histories = []

    with open(path, encoding="utf-8") as file:
        for row in csv.DictReader(file):

            # Преобразуем данные CSV обратно в нужные типы
            row["game"] = int(row["game"])
            row["step"] = int(row["step"])
            row["state"] = literal_eval(row["state"])
            row["action"] = int(row["action"])
            row["reward"] = float(row["reward"])

            if row["next_state"] != "END":
                row["next_state"] = literal_eval(row["next_state"])

            row["terminated"] = row["terminated"] == "True"
            row["cutoff"] = row["cutoff"] == "True"

            histories.append(row)

    return histories


# Загружаем одинаковые истории для всех трех методов
for seed in SEEDS:
    fixed_histories[seed] = load_fixed_games(seed)

# MODEL-BASED LEARNING

TOLERANCE = 0.000001
MAX_SWEEPS = 10000
CHECKPOINTS = [100, 1000, 10000]


# Функция строит модель переходов и оценивает состояния
def model_based_learning(histories):
    action_counts = {}
    transition_counts = {}
    reward_sums = {}

    # Подсчитываем переходы и полученные награды
    for row in histories:
        s = row["state"]
        a = row["action"]
        next_s = row["next_state"]
        r = row["reward"]

        pair = (s, a)
        triple = (s, a, next_s)

        action_counts[pair] = action_counts.get(pair, 0) + 1
        transition_counts[triple] = transition_counts.get(triple, 0) + 1
        reward_sums[triple] = reward_sums.get(triple, 0) + r

    model = {}

    # Вычисляем вероятности переходов и средние награды
    for triple, count in transition_counts.items():
        s, a, next_s = triple

        probability = count / action_counts[(s, a)]
        mean_reward = reward_sums[triple] / count

        if (s, a) not in model:
            model[(s, a)] = []

        model[(s, a)].append(
            (next_s, probability, mean_reward)
        )

    # Создаем начальные значения наблюдаемых состояний
    V = {s: 0.0 for s, a in action_counts}
    V["END"] = 0.0

    # Находим состояния с неизвестными переходами
    affected = set()

    for (s, a), transitions in model.items():
        for next_s, probability, reward in transitions:
            if next_s not in V:
                affected.add(s)

    # Отмечаем состояния, которые зависят от неизвестных значений
    while True:
        previous = affected.copy()

        for (s, a), transitions in model.items():
            if any(next_s in affected
                   for next_s, p, r in transitions):
                affected.add(s)

        if affected == previous:
            break

    converged = False

    # Выполняем синхронные проходы по формуле Беллмана
    for sweep in range(1, MAX_SWEEPS + 1):
        new_V = {"END": 0.0}
        delta = 0.0

        for s in V:
            if s == "END":
                continue

            a = fixed_policy(s)
            value = 0.0

            for next_s, probability, reward in model[(s, a)]:
                value += probability * (
                    reward + GAMMA * V.get(next_s, 0.0)
                )

            new_V[s] = value

            delta = max(delta, abs(value - V[s]))

        V = new_V

        # Останавливаемся, когда значения почти не меняются
        if delta < TOLERANCE:
            converged = True
            break

    # Не выдаем неизвестные значения за достоверные оценки
    for s in affected:
        V[s] = None

    return V, action_counts, sweep, converged, affected

# Проверяем Model-based Learning на каждом seed и checkpoint
print("\nMODEL-BASED LEARNING")

for seed in SEEDS:
    for checkpoint in CHECKPOINTS:

        # Берем только первые 100, 1000 или 10000 игр
        histories = [
            row for row in fixed_histories[seed]
            if row["game"] <= checkpoint
        ]

        V_model, counts, sweeps, converged, affected = (
            model_based_learning(histories)
        )

        print(f"\nSeed: {seed}")
        print("Games:", checkpoint)
        print("Observed states:", len(V_model) - 1)
        print("Sweeps:", sweeps)
        print("Converged:", converged)
        print("Affected states:", len(affected))

# DIRECT EVALUATION (MONTE CARLO)

# Функция вычисляет среднюю награду для каждого состояния
def monte_carlo_learning(histories):
    returns_sum = {}
    visit_counts = {}
    episodes = {}

    # Группируем переходы по отдельным играм
    for row in histories:
        game = row["game"]

        if game not in episodes:
            episodes[game] = []

        episodes[game].append(row)

    # Обрабатываем каждую завершенную игру
    for episode in episodes.values():

        # Исключаем принудительно прерванные игры
        if episode[-1]["cutoff"] or not episode[-1]["terminated"]:
            continue

        G = 0.0

        # Вычисляем накопленную награду от конца игры к началу
        for row in reversed(episode):
            s = row["state"]
            r = row["reward"]

            G = r + GAMMA * G

            # Сохраняем награду и количество посещений состояния
            returns_sum[s] = returns_sum.get(s, 0.0) + G
            visit_counts[s] = visit_counts.get(s, 0) + 1

    # Вычисляем среднее значение для каждого состояния
    V_mc = {}

    for s in returns_sum:
        V_mc[s] = returns_sum[s] / visit_counts[s]

    return V_mc, visit_counts


# Проверяем Monte Carlo для каждого seed и checkpoint
print("\nMONTE CARLO LEARNING")

for seed in SEEDS:
    for checkpoint in CHECKPOINTS:

        # Используем те же игры, что и в Model-based
        histories = [
            row for row in fixed_histories[seed]
            if row["game"] <= checkpoint
        ]

        V_mc, mc_counts = monte_carlo_learning(histories)

        print(f"\nSeed: {seed}")
        print("Games:", checkpoint)
        print("Observed states:", len(V_mc))
        print("Total visits:", sum(mc_counts.values()))
        print("Minimum value:", min(V_mc.values()))
        print("Maximum value:", max(V_mc.values()))

# TD(0) LEARNING

# Функция обновляет значение состояния после каждого перехода
def td_learning(histories):
    V_td = {}
    visit_counts = {}

    for row in histories:
        s = row["state"]
        next_s = row["next_state"]
        r = row["reward"]

        # Получаем текущее значение состояния
        old_value = V_td.get(s, 0.0)

        # Для завершенной игры следующее значение равно нулю
        if next_s == "END":
            next_value = 0.0
        else:
            next_value = V_td.get(next_s, 0.0)

        # Обновляем значение по формуле TD(0)
        target = r + GAMMA * next_value

        V_td[s] = old_value + ALPHA * (
            target - old_value
        )

        visit_counts[s] = visit_counts.get(s, 0) + 1

    return V_td, visit_counts


# Проверяем TD(0) для каждого seed и checkpoint
print("\nTD(0) LEARNING")

for seed in SEEDS:
    for checkpoint in CHECKPOINTS:

        # Используем те же сохраненные истории игр
        histories = [
            row for row in fixed_histories[seed]
            if row["game"] <= checkpoint
        ]

        V_td, td_counts = td_learning(histories)

        print(f"\nSeed: {seed}")
        print("Games:", checkpoint)
        print("Observed states:", len(V_td))
        print("Total visits:", sum(td_counts.values()))
        print("Minimum value:", min(V_td.values()))
        print("Maximum value:", max(V_td.values()))

# SAVE FIXED-POLICY RESULTS

VALUE_DIR = OUTPUT_DIR / "value-tables"
VALUE_DIR.mkdir(parents=True, exist_ok=True)

# Сохраняем результаты трех методов для каждого checkpoint
for seed in SEEDS:
    for checkpoint in CHECKPOINTS:

        histories = [
            row for row in fixed_histories[seed]
            if row["game"] <= checkpoint
        ]

        V_model, counts, sweeps, converged, affected = (
            model_based_learning(histories)
        )

        V_mc, mc_counts = monte_carlo_learning(histories)
        V_td, td_counts = td_learning(histories)

        rows = []

        # Формируем таблицу возможных состояний Blackjack
        for ace in [0, 1]:
            for player in range(4, 22):
                for dealer in range(1, 11):

                    s = (player, dealer, ace)
                    action = fixed_policy(s)

                    rows.append({
                        "player_sum": player,
                        "dealer_showing": dealer,
                        "usable_ace": ace,
                        "policy_action": "hit" if action == 1 else "stick",
                        "visits": counts.get((s, action), 0),
                        "hit_visits": counts.get((s, 1), 0),
                        "stick_visits": counts.get((s, 0), 0),
                        "model_based": V_model.get(s),
                        "monte_carlo": V_mc.get(s),
                        "td": V_td.get(s),
                        "missing_successor": s in affected,
                        "model_converged": converged,
                        "model_sweeps": sweeps
                    })

        df = pd.DataFrame(rows)

        # Сохраняем общую таблицу сравнения методов
        filename = f"values_seed_{seed}_games_{checkpoint}"

        df.to_csv(
            VALUE_DIR / f"{filename}.csv",
            index=False,
            na_rep="NA"
        )

        # Разделяем результаты по наличию usable ace
        for ace in [0, 1]:

            ace_name = "usable_ace" if ace == 1 else "no_usable_ace"
            subset = df[df["usable_ace"] == ace]

            for method in ["model_based", "monte_carlo", "td", "visits"]:

                table = subset.pivot(
                    index="player_sum",
                    columns="dealer_showing",
                    values=method
                )

                table.to_csv(
                    VALUE_DIR / f"{filename}_{ace_name}_{method}.csv",
                    na_rep="NA"
                )

        print(f"Saved value tables: Seed {seed}, Games {checkpoint}")

# Q-LEARNING

# Функция выбирает лучшее действие со случайным выбором при равенстве
def greedy_action(Q, state, rng):
    q_stick = Q.get((state, 0), 0.0)
    q_hit = Q.get((state, 1), 0.0)

    if q_stick > q_hit:
        return 0
    elif q_hit > q_stick:
        return 1
    else:
        return rng.choice([0, 1])


# Функция выбирает действие по epsilon-greedy стратегии
def epsilon_greedy(Q, state, epsilon, rng):

    # Исследуем случайное действие с вероятностью epsilon
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice([0, 1])

    return greedy_action(Q, state, rng)

# Функция обучает Q-learning на новых играх Blackjack
def train_q_learning(seed, epsilon):
    Q = {}
    visit_counts = {}
    training_curve = []
    cutoffs = 0
    block_rewards = []

    # Создаем новую среду и генератор случайных действий
    env = gym.make(
        "Blackjack-v1",
        natural=False,
        sab=False
    )

    rng = random.Random(seed)

    for game in range(1, TRAIN_GAMES + 1):

        # Устанавливаем seed только в начале обучения
        if game == 1:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        total_reward = 0.0

        for step in range(1, MAX_ACTIONS + 1):

            # Выбираем действие и получаем результат
            action = epsilon_greedy(Q, state, epsilon, rng)

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            pair = (state, action)
            old_q = Q.get(pair, 0.0)

            # Находим максимальное значение следующего состояния
            if terminated:
                max_next_q = 0.0
            else:
                max_next_q = max(
                    Q.get((next_state, 0), 0.0),
                    Q.get((next_state, 1), 0.0)
                )

            # Обновляем Q-value по формуле Q-learning
            target = reward + GAMMA * max_next_q

            Q[pair] = old_q + ALPHA * (target - old_q)

            visit_counts[pair] = visit_counts.get(pair, 0) + 1

            # Проверяем завершение или прерывание игры
            cutoff = truncated or (
                step == MAX_ACTIONS and not terminated
            )

            if cutoff:
                cutoffs += 1
                break

            if terminated:
                block_rewards.append(total_reward)
                break

            state = next_state

        # Вычисляем среднюю награду каждые 1000 игр
        if game % 1000 == 0:

            mean_reward = (
                sum(block_rewards) / len(block_rewards)
                if block_rewards else None
            )

            training_curve.append({
                "game": game,
                "mean_reward": mean_reward,
                "completed_games": len(block_rewards),
                "cutoffs": 1000 - len(block_rewards)
            })

            block_rewards = []

    env.close()

    return Q, visit_counts, training_curve, cutoffs

# Обучаем отдельные Q-таблицы для каждого seed и epsilon
print("\nQ-LEARNING TRAINING")

q_results = {}

for epsilon in [0.2, 0.0]:
    for seed in SEEDS:

        Q, q_counts, curve, cutoffs = train_q_learning(
            seed, epsilon
        )

        q_results[(seed, epsilon)] = {
            "Q": Q,
            "counts": q_counts,
            "curve": curve,
            "cutoffs": cutoffs
        }

        print(f"\nSeed: {seed}")
        print("Epsilon:", epsilon)
        print("Training games:", TRAIN_GAMES)
        print("Observed state-action pairs:", len(Q))
        print("Total visits:", sum(q_counts.values()))
        print("Cut-offs:", cutoffs)
        print("Last block mean reward:", curve[-1]["mean_reward"])

# FROZEN-POLICY EVALUATION

# Функция проверяет стратегию на новых играх без обучения
def evaluate_policy(eval_seed, Q=None):
    wins = 0
    losses = 0
    draws = 0
    cutoffs = 0
    rewards = []

    # Создаем новую среду для независимой проверки
    env = gym.make(
        "Blackjack-v1",
        natural=False,
        sab=False
    )

    rng = random.Random(eval_seed)

    for game in range(1, EVAL_GAMES + 1):

        if game == 1:
            state, info = env.reset(seed=eval_seed)
        else:
            state, info = env.reset()

        total_reward = 0.0

        for step in range(1, MAX_ACTIONS + 1):

            # Выбираем фиксированную или обученную стратегию
            if Q is None:
                action = fixed_policy(state)
            else:
                action = greedy_action(Q, state, rng)

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            # Проверяем ограничение количества действий
            cutoff = truncated or (
                step == MAX_ACTIONS and not terminated
            )

            if cutoff:
                cutoffs += 1
                break

            # Подсчитываем результаты завершенных игр
            if terminated:
                rewards.append(total_reward)

                if total_reward > 0:
                    wins += 1
                elif total_reward < 0:
                    losses += 1
                else:
                    draws += 1

                break

            state = next_state

    env.close()

    mean_reward = (
        sum(rewards) / len(rewards)
        if rewards else None
    )

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "cutoffs": cutoffs,
        "mean_reward": mean_reward
    }

# Проверяем все стратегии на одинаковых evaluation seeds
print("\nFROZEN-POLICY EVALUATION")

evaluation_results = []

for seed in SEEDS:

    eval_seed = 1000 + seed

    policies = [
        ("Fixed policy", None),
        ("Q-learning epsilon 0.2", q_results[(seed, 0.2)]["Q"]),
        ("Q-learning epsilon 0.0", q_results[(seed, 0.0)]["Q"])
    ]

    for policy_name, Q in policies:

        result = evaluate_policy(eval_seed, Q)

        evaluation_results.append({
            "training_seed": seed,
            "policy": policy_name,
            "evaluation_seed": eval_seed,
            **result
        })

        print(f"\nTraining seed: {seed}")
        print("Policy:", policy_name)
        print("Evaluation seed:", eval_seed)
        print("Wins:", result["wins"])
        print("Losses:", result["losses"])
        print("Draws:", result["draws"])
        print("Cut-offs:", result["cutoffs"])
        print("Mean reward:", result["mean_reward"])


# Сохраняем результаты независимой оценки
evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    OUTPUT_DIR / "evaluation_results.csv",
    index=False
)

# SAVE Q-LEARNING RESULTS

Q_DIR = OUTPUT_DIR / "q-tables"
Q_DIR.mkdir(parents=True, exist_ok=True)

CURVE_DIR = OUTPUT_DIR / "training-curves"
CURVE_DIR.mkdir(parents=True, exist_ok=True)

# Сохраняем результаты каждого эксперимента Q-learning
for (seed, epsilon), result in q_results.items():

    Q = result["Q"]
    counts = result["counts"]
    curve = result["curve"]

    rows = []

    # Формируем таблицу значений действий для каждого состояния
    for ace in [0, 1]:
        for player in range(4, 22):
            for dealer in range(1, 11):

                s = (player, dealer, ace)

                q_hit = Q.get((s, 1), 0.0)
                q_stick = Q.get((s, 0), 0.0)

                hit_count = counts.get((s, 1), 0)
                stick_count = counts.get((s, 0), 0)

                # Определяем лучшее действие по Q-values
                if hit_count + stick_count == 0:
                    policy = "unobserved"
                elif q_hit > q_stick:
                    policy = "hit"
                elif q_stick > q_hit:
                    policy = "stick"
                else:
                    policy = "tie"

                rows.append({
                    "player_sum": player,
                    "dealer_showing": dealer,
                    "usable_ace": ace,
                    "q_hit": q_hit,
                    "q_stick": q_stick,
                    "hit_visits": hit_count,
                    "stick_visits": stick_count,
                    "hit_observed": hit_count > 0,
                    "stick_observed": stick_count > 0,
                    "greedy_policy": policy
                })

    df = pd.DataFrame(rows)

    filename = f"q_seed_{seed}_epsilon_{epsilon}"

    # Сохраняем общую Q-таблицу
    df.to_csv(
        Q_DIR / f"{filename}.csv",
        index=False
    )

    # Разделяем таблицы по наличию usable ace
    for ace in [0, 1]:

        ace_name = "usable_ace" if ace == 1 else "no_usable_ace"

        subset = df[df["usable_ace"] == ace]

        for column in [
            "q_hit",
            "q_stick",
            "hit_visits",
            "stick_visits",
            "greedy_policy"
        ]:

            table = subset.pivot(
                index="player_sum",
                columns="dealer_showing",
                values=column
            )

            table.to_csv(
                Q_DIR / f"{filename}_{ace_name}_{column}.csv"
            )

    # Сохраняем средние награды по блокам из 1000 игр
    pd.DataFrame(curve).to_csv(
        CURVE_DIR / f"training_seed_{seed}_epsilon_{epsilon}.csv",
        index=False
    )

    print(f"Saved Q-tables: Seed {seed}, Epsilon {epsilon}")

# TRAINING AND EVALUATION PLOTS

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Строим графики обучения отдельно для каждого seed
for seed in SEEDS:

    plt.figure(figsize=(10, 5))

    for epsilon in [0.2, 0.0]:

        curve = pd.DataFrame(
            q_results[(seed, epsilon)]["curve"]
        )

        plt.plot(
            curve["game"],
            curve["mean_reward"],
            label=f"Epsilon = {epsilon}"
        )

    plt.title(f"Q-learning Training — Seed {seed}")
    plt.xlabel("Training games")
    plt.ylabel("Mean reward per 1000 games")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"training_seed_{seed}.png",
        dpi=300
    )

    plt.close()

    print(f"Saved training plot: Seed {seed}")


# Сравниваем стратегии после завершения обучения
comparison = evaluation_df.pivot(
    index="training_seed",
    columns="policy",
    values="mean_reward"
)

comparison.plot(
    kind="bar",
    figsize=(10, 5),
    rot=0
)

plt.title("Frozen-policy Evaluation")
plt.xlabel("Training seed")
plt.ylabel("Mean reward")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "evaluation_comparison.png",
    dpi=300
)

plt.close()

print("Saved evaluation comparison plot")

# FREQUENT AND RARE STATES COMPARISON

comparison_rows = []

state_columns = [
    "player_sum",
    "dealer_showing",
    "usable_ace"
]

for seed in SEEDS:

    # Загружаем результаты после 10000 игр
    final_df = pd.read_csv(
        VALUE_DIR / f"values_seed_{seed}_games_10000.csv"
    )

    observed = final_df[final_df["visits"] > 0]

    # Выбираем три частых и три редких состояния
    frequent = observed.nlargest(3, "visits")
    rare = observed.nsmallest(3, "visits")

    selected = [
        ("Frequent", frequent),
        ("Rare", rare)
    ]

    for category, states in selected:

        for checkpoint in CHECKPOINTS:

            # Загружаем оценки после каждого количества игр
            df = pd.read_csv(
                VALUE_DIR / f"values_seed_{seed}_games_{checkpoint}.csv"
            )

            matched = states[state_columns].merge(
                df,
                on=state_columns
            )

            for _, row in matched.iterrows():

                comparison_rows.append({
                    "seed": seed,
                    "category": category,
                    "games": checkpoint,
                    "player_sum": row["player_sum"],
                    "dealer_showing": row["dealer_showing"],
                    "usable_ace": row["usable_ace"],
                    "visits": row["visits"],
                    "model_based": row["model_based"],
                    "monte_carlo": row["monte_carlo"],
                    "td": row["td"]
                })


# Сохраняем таблицу сравнения
states_comparison = pd.DataFrame(comparison_rows)

states_comparison.to_csv(
    OUTPUT_DIR / "frequent_rare_comparison.csv",
    index=False,
    na_rep="NA"
)

print("\nFREQUENT AND RARE STATES COMPARISON")

print(
    states_comparison[
        states_comparison["seed"] == 11
    ].round(4).to_string(index=False)
)

print("\nSaved frequent_rare_comparison.csv")