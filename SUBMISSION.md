# HW7 submission — Blackjack

## AI-use declaration

I used ChatGPT to understand the learning algorithms, check my paper calculations, write and explain Python code, and check the program results. I also used VS Code to run and test my code.

## Setup and reproduction

I used Python, VS Code, Gymnasium, NumPy, Pandas and Matplotlib.

Python version: 3.13.15.

Gymnasium version: 1.4.0.

All dependency versions are saved in [requirements.txt](requirements.txt).

### Environment setup

Create a virtual environment:

`python -m venv .venv`

Activate the environment:

`.\.venv\Scripts\Activate.ps1`

Install dependencies:

`python -m pip install -r requirements.txt`

### Part C — Paper replay

Run the following command to reproduce the paper calculations:

`python .\src\part_c_replay.py | Tee-Object -FilePath .\outputs\paper-replay\replay_log.txt`

### Part D — Simulated games

Run the following command:

`python .\src\part_d_learning.py`

This command generates fixed-policy games, runs all four learning methods, trains Q-learning, evaluates frozen policies and saves tables and plots.

All results are saved in the `outputs/` folder.

No GPU or API key is required.

## Part A — Count states and actions

I calculated the number of states and actions for Blackjack, Durak, tic-tac-toe and Nim. I also explained legal actions, hidden information and the differences between these games.

[Paper calculations and comparison table](paper/24-11_05-39.pdf)

## Part B — Paper calculations

I calculated all four learning methods by hand:

- Model-based learning
- Direct evaluation (Monte Carlo)
- TD(0)
- Q-learning

I included formulas, intermediate calculations and final tables.

[Paper calculations](paper/24-11_05-39.pdf)

## Part C — Reproduce the calculations

All four learning methods were implemented using the original game histories from `data/paper-games.csv`. I used the same parameters, initial values and game order as in Part B.

### Output files

- [Full replay log](outputs/paper-replay/replay_log.txt)
- [Paper vs Program comparison](outputs/paper-replay/paper_vs_program.csv)
- [TD(0) updates](outputs/paper-replay/td_updates.csv)
- [Q-learning updates](outputs/paper-replay/q_updates.csv)
- [Final results](outputs/paper-replay/final_results.csv)

### Comparison

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | --- | --- | --- | --- |
| Model-based | A | Sweep 1 | -0.333333 | -0.333333 | 0 | Match |
| Model-based | A | Sweep 2 | -0.333333 | -0.333333 | 0 | Match |
| Monte Carlo | A | Pass 1 | -0.333333 | -0.333333 | 0 | Match |
| TD(0) | A | Pass 1 | -0.080000 | -0.080000 | 0 | Match |
| TD(0) | A | Pass 2 | -0.140030 | -0.140030 | 0 | Match |
| TD(0) | B | Pass 2 | -0.018100 | -0.018100 | 0 | Match |
| Q-learning | A, hit | Pass 1 | -0.080000 | -0.080000 | 0 | Match |
| Q-learning | A, hit | Pass 2 | -0.139220 | -0.139220 | 0 | Match |
| Q-learning | B, stick | Pass 2 | -0.018100 | -0.018100 | 0 | Match |

The complete comparison table contains 62 comparisons.

- Total comparisons: 62
- Matching results: 62
- Mismatches: 0
- Numerical tolerance: 0.000000001

All my paper and program results match. There are only very small differences because of how Python works with decimal numbers.

The four methods work in different ways. Model-based learning and Monte Carlo give the same results in this example. TD(0) updates the value after each step, while Q-learning learns the value of each action.

In this part, I checked if the algorithms calculate the values correctly. I did not test exploration or action selection.

## Part D — Simulated games

### Fixed-policy value estimates

I generated 10,000 games for each seed: 11, 22 and 33. I used the same saved games for Model-based Learning, Monte Carlo and TD(0).

The fixed policy takes a card when the player sum is below 20 and stops otherwise.

Game logs:

- [Seed 11](outputs/simulated-games/fixed_games_seed_11.csv)
- [Seed 22](outputs/simulated-games/fixed_games_seed_22.csv)
- [Seed 33](outputs/simulated-games/fixed_games_seed_33.csv)

[Value tables and visit counts](outputs/simulated-games/value-tables/)

| Seed | States after 100 games | States after 1,000 games | States after 10,000 games |
| --- | --- | --- | --- |
| 11 | 96 | 243 | 280 |
| 22 | 116 | 245 | 280 |
| 33 | 118 | 247 | 280 |

With more games, the methods observed more states. Model-based Learning converged in all experiments. There were no states affected by missing successor information.

Unobserved states have missing values (NA). A visit count of zero means that the state or action was not observed.

### Frequent and rare states

[Full comparison table](outputs/simulated-games/frequent_rare_comparison.csv)

Frequent state: (20, 10, 0), seed 11.

| Games | Visits | Model-based | Monte Carlo | TD(0) |
| --- | --- | --- | --- | --- |
| 100 | 6 | 0.3333 | 0.3333 | 0.1729 |
| 1000 | 60 | 0.4667 | 0.4667 | 0.5111 |
| 10000 | 606 | 0.4158 | 0.4158 | 0.4393 |

After 10,000 games, the three values are close because this state has many visits.

Rare state: (4, 9, 0), seed 11.

| Games | Visits | Model-based | Monte Carlo | TD(0) |
| --- | --- | --- | --- | --- |
| 100 | 0 | NA | NA | NA |
| 1000 | 1 | -0.6530 | 1.0000 | 0.0000 |
| 10000 | 1 | -0.4883 | 1.0000 | 0.0000 |

This state has only one visit. Monte Carlo uses the result of that game. Model-based Learning uses learned transitions and other state values. TD(0) updates its value step by step. The estimates are different because there is not enough data.

### Q-learning and baseline evaluation

I trained Q-learning for 50,000 games for each seed and each epsilon value.

Parameters:

- Learning rate: 0.1
- Discount factor: 1.0
- Epsilon: 0.2 and 0
- Training seeds: 11, 22 and 33
- Evaluation games: 10,000 per policy
- Maximum actions: 100 per game

I used new games for evaluation. The Q-tables were frozen during evaluation. There was no exploration branch and no learning during evaluation.

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Fixed policy | 1011 | 2962 | 6462 | 576 | 0 | -0.3500 |
| 11 | Q-learning 0.2 | 1011 | 4228 | 4882 | 890 | 0 | -0.0654 |
| 11 | Q-learning 0 | 1011 | 4100 | 4929 | 971 | 0 | -0.0829 |
| 22 | Fixed policy | 1022 | 2935 | 6481 | 584 | 0 | -0.3546 |
| 22 | Q-learning 0.2 | 1022 | 4220 | 4919 | 861 | 0 | -0.0699 |
| 22 | Q-learning 0 | 1022 | 4100 | 4975 | 925 | 0 | -0.0875 |
| 33 | Fixed policy | 1033 | 2950 | 6496 | 554 | 0 | -0.3546 |
| 33 | Q-learning 0.2 | 1033 | 4124 | 5025 | 851 | 0 | -0.0901 |
| 33 | Q-learning 0 | 1033 | 4098 | 4943 | 959 | 0 | -0.0845 |

[Evaluation results](outputs/simulated-games/evaluation_results.csv)

Both Q-learning policies performed better than the fixed policy for all three seeds.

The average rewards across the three seeds were:

| Policy | Average reward |
| --- | --- |
| Fixed policy | -0.3531 |
| Q-learning epsilon 0.2 | -0.0751 |
| Q-learning epsilon 0 | -0.0850 |

Epsilon 0.2 gave the best average result. However, epsilon 0 performed better for seed 33. All average rewards were still negative.

There were no cut-offs in collection, training or evaluation. No games were excluded.

### Plots and learned policies

The training plots show the mean reward for each block of 1,000 games.

- [Training plot — Seed 11](outputs/simulated-games/plots/training_seed_11.png)
- [Training plot — Seed 22](outputs/simulated-games/plots/training_seed_22.png)
- [Training plot — Seed 33](outputs/simulated-games/plots/training_seed_33.png)
- [Evaluation comparison](outputs/simulated-games/plots/evaluation_comparison.png)
- [Training curve data](outputs/simulated-games/training-curves/)
- [Q-values, visit counts and greedy policies](outputs/simulated-games/q-tables/)

The tables are separated into usable-ace and no-usable-ace states.

### Explain your results

**Model-based Learning:** This method learns transition probabilities and rewards. It uses the learned model to calculate state values.

**Monte Carlo:** This method learns state values from completed games. It uses the average return from all visits.

**TD(0):** This method learns state values after each transition. It does not need to wait until the game ends.

**Q-learning:** This method learns action values. It uses Q-values to choose between hit and stick.

**Prediction and control:** Model-based evaluation, Monte Carlo and TD(0) evaluate a fixed policy in this experiment. Q-learning learns a policy and chooses actions.

**Terminal transitions:** When a game ends, the next state has zero continuation value. The update uses the final reward without a future reward.

**Usable ace:** An ace can count as 11 or 1. A usable ace can reduce the risk of going over 21. This is why it is important for choosing an action.

**Exploration:** With epsilon 0.2, the agent sometimes chooses a random action. It observed all 560 reachable state-action pairs for each seed. With epsilon 0, it observed 538, 537 and 545 pairs. Random tie-breaking was still used.

**Limited visits:** Rare states and actions have less information. An initial Q-value of zero does not mean that an unobserved action is good.

**Final conclusion:** Q-learning improved the mean reward compared with the fixed policy for all three seeds. Exploration gave a better average result, but it was not better for every seed. More training does not guarantee an optimal policy.