# HW7 submission — Blackjack

## AI-use declaration

I used ChatGPT to understand the learning algorithms, check my paper calculations, write and explain Python code, and check the program results. I also used VS Code to run and test my code.

## Setup and reproduction

### Part C — Paper replay

To reproduce my paper calculations, run this command in PowerShell:

`python .\src\part_c_replay.py | Tee-Object -FilePath .\outputs\paper-replay\replay_log.txt`

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

Link the shared logs, checkpoint value tables and visit counts. Explain
missing states and differences among model-based, Monte Carlo and TD estimates.

### Q-learning and baseline evaluation

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Plots and learned policies

Link training curves, Q-tables, hit/stick policy tables and state-action counts.
Separate usable-ace and no-usable-ace results.

### Explain your results

Address prediction versus control, terminal updates, usable aces, exploration,
coverage, variation across seeds and whether the learned policy improved.
