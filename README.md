# Argus

A vision-based reinforcement learning agent that learns to play Atari games
directly from raw pixels. Every component - preprocessing, the CNN, the
Double DQN algorithm, and the training loop - is implemented from scratch
in PyTorch. No pretrained weights, no external AI APIs.

## What it does

The agent receives only raw screen pixels (no game state, no ball/paddle
coordinates) and learns through trial and error to play well, using the
same core algorithm from the original 2015 DeepMind DQN paper, plus the
Double DQN fix for its known Q-value overestimation problem.

Tested on four games:

| Game | Episodes | Mean reward (n=50) | Training time |
|---|---|---|---|
| Breakout | 2,000  | 10.78   | ~8.4 hours |
| Pong     | 2,000  | 14.14   | ~66.2 hours |
| Seaquest | 2,000  | 1129.60 | ~29.8 hours |
| MsPacman | 20,000 | see report - peaked mid-training, then declined | ~day+ |

Notably, MsPacman was trained for 20,000 episodes (10x the others) to test
whether performance keeps improving indefinitely. It did not: reward rose
to a peak around episode 9,000, then gradually declined over the remaining
10,000+ episodes. See ARGUS_REPORT.md Section 11 for full analysis.

## Project structure

argus/
  notebooks/
    01_explore_env.ipynb        - step-by-step pipeline verification
  src/
    preprocess.py               - grayscale + frame-stacking (the CV layer)
    model.py                    - CNN Q-network architecture
    agent.py                    - replay buffer + Double DQN / vanilla DQN
    train.py                    - training loop (--game, --mode, --episodes, --resume)
    evaluate.py                 - statistical evaluation over N episodes
    record_gameplay.py          - records .mp4 footage from a checkpoint
    plot_results.py             - plots reward/epsilon curves
    plot_comparison.py          - vanilla vs Double DQN comparison plot
  checkpoints/
    breakout/double/            - Breakout, Double DQN
    pong/double/                - Pong, Double DQN
    seaquest/double/            - Seaquest, Double DQN
    mspacman/double/            - MsPacman, Double DQN (20,000 episodes)
    vanilla/                    - Breakout, vanilla DQN comparison run
  videos/
    breakout/, pong/, seaquest/ - early_ep100.mp4, trained_ep2000.mp4
    mspacman/                   - early_ep100.mp4, peak_ep9700.mp4, final_ep20000.mp4
  ARGUS_REPORT.md               - full written report

## Setup

python -m venv venv
venv\Scripts\Activate.ps1
pip install torch torchvision gymnasium[atari] opencv-python matplotlib numpy jupyter moviepy

## Usage

Train an agent:
python src/train.py --game Breakout --mode double --episodes 2000

For long runs, resume after any interruption:
python src/train.py --game MsPacman --mode double --episodes 20000 --resume

Evaluate a trained checkpoint:
python src/evaluate.py --checkpoint checkpoints/mspacman/double/dqn_ep9700.pt --game MsPacman --episodes 50

Record gameplay video:
python src/record_gameplay.py --checkpoint checkpoints/mspacman/double/dqn_ep9700.pt --game MsPacman --episodes 1

Plot training progress:
python src/plot_results.py
python src/plot_comparison.py

## Method summary

- Preprocessing: raw RGB frames -> grayscale -> resized to 84x84 -> 4
  consecutive frames stacked for motion information
- Network: 3 convolutional layers (32/64/64 channels) + 2 fully connected
  layers, mapping (4, 84, 84) input to one Q-value per action
- Training: experience replay (100k capacity), a periodically-synced
  target network, Double DQN's decoupled action selection/evaluation,
  epsilon-greedy exploration decaying from 1.0 to 0.05 over 200k steps

Full methodology, results, and discussion are in ARGUS_REPORT.md, including
a vanilla-vs-Double-DQN comparison, a second/third-game generalization
test, and a 20,000-episode long-run analysis on MsPacman.

## Known limitations

- Trained on CPU only; final performance is capped well below published
  benchmark scores due to limited training budgets for most games
- The vanilla-vs-Double-DQN Q-value comparison is not fully matched (see
  report Section 5.3 for details)
- The MsPacman training log contains ~529 duplicated episode-number rows
  around episode 3701, caused by the resume mechanism re-logging a small
  range of already-completed episodes after a restart. This does not
  affect the trained model weights, only some redundant log rows.

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement
  learning. Nature, 518(7540), 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement
  learning with double Q-learning. AAAI, 30(1).
