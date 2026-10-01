# Argus: A Vision-Based Reinforcement Learning Agent for Atari Breakout

## Appendix

### A.1 Hyperparameters

| Parameter | Value |
|---|---|
| Environment | ALE/Breakout-v5 |
| Episodes | 2000 |
| Batch size | 32 |
| Discount factor (gamma) | 0.99 |
| Learning rate | 1e-4 |
| Replay buffer capacity | 100,000 |
| Min buffer size before training | 5,000 |
| Target network update frequency | every 1,000 steps |
| Epsilon start / end | 1.0 / 0.05 |
| Epsilon decay steps | 200,000 |
| Optimizer | Adam |
| Loss function | Smooth L1 (Huber) |
| Gradient clip norm | 10.0 |

### A.2 Repository Structure

argus/
+-- notebooks/
| +-- 01_explore_env.ipynb - environment/pipeline exploration and verification
+-- src/
| +-- preprocess.py - grayscale + frame-stacking preprocessing
| +-- model.py - CNN Q-network architecture
| +-- agent.py - replay buffer, Double DQN / vanilla DQN agent
| +-- train.py - training loop (--mode double|vanilla)
| +-- evaluate.py - statistical evaluation over N episodes
| +-- record_gameplay.py - records .mp4 gameplay footage from a checkpoint
| +-- plot_results.py - plots reward/epsilon curves from training logs
+-- checkpoints/ - saved model weights and training logs
+-- videos/ - recorded gameplay footage

### A.3 Referenced Media

- checkpoints/reward_curve.png - full training reward and epsilon curve
- videos/early_ep100_episode-{0,1,2}.mp4 - gameplay at episode 100
- videos/trained_ep2000_episode-{0,1,2}.mp4 - gameplay at episode 2000

### A.4 References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement
  learning. Nature, 518(7540), 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement
  learning with double Q-learning. Proceedings of the AAAI Conference on
  Artificial Intelligence, 30(1).
  
## 1. Introduction

Reinforcement learning agents that operate directly on raw sensory input,
rather than on hand-engineered game state, must solve two problems at
once: perceiving the environment and deciding how to act within it. This
project, Argus, addresses both problems for the classic Atari game
Breakout. The agent receives only raw pixel frames from the game screen -
no access to ball position, paddle position, or brick layout as explicit
variables - and must learn, purely from trial and error, both how to
interpret those pixels and how to act on them to maximize score.

The motivation for this project was to implement a complete vision-based
reinforcement learning pipeline entirely from first principles: the
convolutional feature extractor, the Q-learning algorithm, the experience
replay mechanism, and the training loop were all implemented from scratch
in PyTorch, with no pretrained weights and no reliance on external AI
services. Breakout was chosen as the target environment because it is a
well-studied benchmark with a clear, well-documented body of prior work to
validate results against, while still presenting a genuinely nontrivial
learning problem: the agent must learn to track a moving ball, position a
paddle accurately, and associate a sparse, delayed reward signal (points
scored on brick contact) with the sequence of actions that led to it.

## 2. Background and Related Work

The algorithmic foundation of this project is the Deep Q-Network (DQN)
introduced by Mnih et al. (2015), which first demonstrated that a single
neural network could learn to play a range of Atari games directly from
pixels, at or above human level, using a combination of convolutional
feature extraction and Q-learning. The original DQN architecture
introduced two key stabilization techniques that this project also
implements: experience replay, which breaks the temporal correlation
between consecutive training samples by sampling randomly from a buffer
of past transitions, and a separate target network, which is updated only
periodically to prevent the training target from shifting on every
gradient step.

A known limitation of vanilla DQN is its tendency to overestimate
action-value (Q) estimates, because the same network is used both to
select the best next action and to evaluate that action's value - any
noise in the network's estimates is systematically biased toward
overestimation, since the max operator preferentially selects
overestimated values. Van Hasselt et al. (2016) proposed Double DQN as a
fix: the online network selects the best next action, but the (separate)
target network evaluates that action's value. This project implements
both variants and empirically compares them (see Section 5.3).

## 3. Method and Architecture

### 3.1 Environment

The environment used is `ALE/Breakout-v5`, accessed through the
Gymnasium interface. Each raw observation is a 210x160 RGB image
representing the current game screen. The agent's available actions are
the four native Breakout controls (no-op, fire, move right, move left).

### 3.2 Preprocessing

Raw frames are converted to grayscale and resized to 84x84 pixels,
following the standard preprocessing pipeline established in the original
DQN paper. Color information is discarded as it carries no information
relevant to gameplay, and downsizing substantially reduces the
computational cost of the convolutional network without a meaningful loss
of the visual detail needed to track the ball and paddle.

A single static frame does not contain enough information to infer
motion - from one frame alone, the direction the ball is travelling is
ambiguous. To address this, four consecutive preprocessed frames are
stacked together into a single (4, 84, 84) input tensor, giving the
network implicit access to short-term motion information without
requiring an explicit optical-flow computation.

### 3.3 Network Architecture

The core network is a convolutional Q-network with the following
structure:

| Layer | Type | Output shape |
|---|---|---|
| Input | - | (4, 84, 84) |
| Conv1 | 32 filters, 8x8, stride 4 | (32, 20, 20) |
| Conv2 | 64 filters, 4x4, stride 2 | (64, 9, 9) |
| Conv3 | 64 filters, 3x3, stride 1 | (64, 7, 7) |
| Flatten | - | 3136 |
| FC1 | Linear + ReLU | 512 |
| FC2 (output) | Linear | n_actions (4) |

Each convolutional layer is followed by a ReLU nonlinearity. The final
output layer produces one scalar Q-value per possible action, representing
the network's estimate of expected future reward if that action is taken
from the current state.

### 3.4 Learning Algorithm

The agent is trained using Q-learning with the following components:

- **Experience replay**: a buffer (capacity 100,000 transitions) stores
  past (state, action, reward, next_state, done) tuples. Training batches
  of 32 transitions are sampled uniformly at random from this buffer,
  which breaks the strong temporal correlation present in sequential
  gameplay frames and substantially stabilizes training.
- **Target network**: a second copy of the network, used only to compute
  the bootstrap target for each training update, synchronized with the
  online network's weights every 1,000 environment steps rather than every
  step.
- **Double DQN** (in the primary configuration): the online network
  selects the action it currently believes is best for the next state, but
  the target network is used to evaluate that action's Q-value, decoupling
  selection from evaluation.
- **Epsilon-greedy exploration**: the agent selects a random action with
  probability epsilon, and its currently-preferred action otherwise.
  Epsilon decays linearly from 1.0 to 0.05 over the first 200,000
  environment steps, so the agent explores heavily early in training and
  increasingly exploits its learned policy as training progresses.
- **Optimization**: the network is trained using the Adam optimizer
  (learning rate 1e-4) to minimize the smooth L1 (Huber) loss between
  predicted and target Q-values, with gradient norm clipping (max norm
  10.0) applied to prevent occasional large, destabilizing updates.

## 4. Experimental Setup

Training was conducted entirely on CPU (no GPU acceleration), over 2000
episodes. Training only begins once the replay buffer contains at least
5,000 transitions, to avoid training on an unrepresentative, near-empty
buffer early on. A full hyperparameter listing is provided in the
Appendix.
## 5. Results

### 5.1 Learning curve

Over the course of 2000 training episodes, the agent's average episode
reward rose from near zero during the initial, mostly-random exploration
phase to a stable plateau of approximately 10-13 points as epsilon decayed
from 1.0 to its floor of 0.05 (occurring over episodes 0-650). Reward
remained stable in this range for the remaining ~1350 episodes, indicating
the agent converged to a consistent policy rather than continuing to
improve within the available training budget (see reward_curve.png).

### 5.2 Quantitative before/after comparison

To confirm the learning curve reflected genuine, reproducible improvement
rather than noise, the agent was evaluated over 50 independent episodes
(fully greedy, epsilon=0) at two checkpoints:

| Checkpoint | Episodes trained | Mean reward (n=50) | Std dev | Min / Max |
|---|---|---|---|---|
| dqn_ep100.pt  | 100  | 0.00  | 0.00 | 0.0 / 0.0 |
| dqn_ep2000.pt | 2000 | 10.78 | 4.61 | 3.0 / 24.0 |

At episode 100, the agent scored zero in every one of 50 evaluation
episodes, indicating it had not yet learned to reliably return the ball.
By episode 2000, the same agent scored a mean of 10.78 (std 4.61) across
50 episodes, with a minimum of 3 and a maximum of 24 - a substantial and
statistically consistent improvement over a purely random baseline.

### 5.3 Vanilla DQN vs. Double DQN comparison

A second training run was conducted using vanilla DQN (target network
both selects and evaluates the next action's value, rather than
decoupling the two roles as in Double DQN), holding all other
hyperparameters, network architecture, and random game identical.

| Run | Episodes | Final mean reward (last 100 ep.) | Final mean Q-value (last 100 ep.) | Wall-clock time |
|---|---|---|---|---|
| Vanilla DQN | 2000 | 12.67 | 3.988 | 8.44 hours |
| Double DQN  | 2000 | ~12-13 (see Section 5.1) | not logged (see limitation below) | ~8.2 hours |

At this training scale, final reward was comparable between the two
variants, both plateauing in a similar range. The vanilla DQN run's mean
Q-value estimate rose steadily from 0.15 at the start of training to
approximately 4.0 by the end, and had not yet clearly plateaued at
episode 2000 - consistent with the expected behavior of unconstrained
value estimates continuing to drift upward absent the corrective effect
of decoupled action selection and evaluation.

**Limitation**: Q-value logging (avg_q) was added to the training
script after the original Double DQN run had already completed, so a
direct, matched Q-value comparison between the two variants is not
available at the time of writing. The vanilla DQN run's own Q-value
trajectory is reported above as a standalone observation consistent with
the overestimation behavior described in the literature (van Hasselt et
al., 2016), but the stronger claim - that Double DQN's Q-values are
measurably lower under identical conditions - was not directly measured
in this project and is noted here as an area for future verification
rather than asserted without evidence.

### 5.4 Qualitative results

Recorded gameplay footage at episode 100 shows the agent's paddle
remaining largely stationary or moving without clear correlation to the
ball's position, consistent with its near-random policy (epsilon = 0.91)
and zero-reward evaluation results at this stage. Footage at episode 2000
shows the paddle consistently tracking and intercepting the ball across
multiple rallies, visually corroborating the quantitative improvement
reported in Section 5.2.

## 6. Discussion

The results demonstrate that the implemented pipeline - custom
preprocessing, a convolutional Q-network, experience replay, a target
network, and Double DQN - successfully learns a real, reproducible
Breakout-playing policy from raw pixels alone, improving from a
zero-reward random baseline to a consistent mean reward of approximately
10-13 points. This progression matches the expected qualitative shape of
DQN learning curves reported in the original literature: reward tracks
inversely with the epsilon exploration schedule, then stabilizes once
epsilon reaches its floor.

The final performance plateau (~10-13 points) is well below scores
reported for fully-trained DQN agents in the original benchmark
literature (100+ points), which is attributable to the substantially
smaller training budget used here (2000 episodes, on the order of
900,000 environment steps) compared to the tens of millions of frames
used in published results, combined with CPU-only compute limiting the
practical size of that budget. This is interpreted as a scope limitation
of the available compute, not a defect in the implementation, and is
consistent with the plateau occurring almost immediately after epsilon
reached its floor - suggesting the network had converged to the best
policy it could extract from its current architecture and training
budget, rather than being cut off mid-improvement.

## 7. Limitations and Future Work

- **Training budget**: all training was conducted on CPU, capping the
  practical number of episodes at 2000. Access to GPU acceleration would
  likely allow substantially longer training and a higher final reward
  plateau.
- **Incomplete Double DQN vs. vanilla Q-value comparison**: as noted in
  Section 5.3, matched Q-value logs are not available for the original
  Double DQN run. Re-running Double DQN training with the current logging
  code would allow a fully matched comparison.
- **Single environment tested**: only Breakout was evaluated. Testing the
  same pipeline on a second Atari environment (e.g., Pong) would help
  establish that the implementation generalizes rather than being
  incidentally tuned to Breakout's specific dynamics.
- **Possible extensions**: Prioritized Experience Replay (sampling replay
  transitions by TD-error rather than uniformly), a Dueling DQN
  architecture (separating state-value and action-advantage estimation),
  and frame-skip/hyperparameter tuning are all natural next steps that
  could push performance past the observed plateau.

## 8. Conclusion

This project implemented a complete vision-based reinforcement learning
pipeline from scratch - raw-pixel preprocessing, a convolutional Q-network,
experience replay, a target network, and Double DQN - and trained it to
play Atari Breakout without any pretrained models or external AI
services. The trained agent showed a clear, statistically verified
improvement over a random baseline (0.00 to 10.78 mean reward across 50
evaluation episodes), corroborated by both a full training reward curve
and recorded gameplay footage. A supplementary experiment training a
vanilla (non-Double) DQN variant under identical conditions showed
comparable final reward but a Q-value trajectory still rising at the end
of training, consistent with, though not conclusively demonstrating, the
overestimation behavior Double DQN is designed to address - a comparison
this report identifies as incomplete and flags as a direction for
follow-up work.

## 9. Generalization to a Second Game

To test whether the implemented pipeline generalizes beyond Breakout, or
was implicitly tuned to its specific dynamics, the identical codebase
(preprocessing, network architecture, Double DQN agent, and training
loop) was trained on a second Atari environment, Pong, with no
game-specific code changes beyond the environment name and the network's
action-count parameter (Pong has 6 available actions versus Breakout's
4).

### 9.1 Training results

Pong was trained for 2000 episodes, the same budget used for Breakout,
though wall-clock time differed substantially: Pong training took
approximately 66.2 hours, compared to Breakout's approximately 8.4 hours.
This difference is attributable to Pong's longer average episode length
under a trained policy (extended rallies), not any implementation
difference between the two runs.

| Episode range | Mean reward | Mean Q-value |
|---|---|---|
| 1-100    | -20.37 | -0.681 |
| 501-600  | -14.55 |  0.109 |
| 1001-1100| -4.01  |  1.218 |
| 1201-1300|  5.03  |  1.234 |
| 1601-1700| 10.87  |  1.223 |
| 1901-2000| 10.22  |  1.188 |

Reward climbed almost monotonically from -20.37 (near-total loss,
consistent with a random policy against Pong's built-in opponent) to a
stable plateau around +9 to +11 by the final several hundred episodes,
crossing from net losses to net wins around episode 1100. This is a
cleaner, more monotonic learning curve than the one observed for
Breakout, though direct comparison should be treated cautiously given the
two games' different reward structures and dynamics.

### 9.2 Quantitative evaluation

The trained Pong checkpoint (episode 2000) was evaluated identically to
the Breakout agent: 50 independent episodes, fully greedy (epsilon=0).

| Game | Mean reward (n=50) | Std dev | Min / Max | Median |
|---|---|---|---|---|
| Breakout | 10.78 | 4.61 | 3.0 / 24.0  | 10.0 |
| Pong     | 14.14 | 3.32 | 4.0 / 20.0  | 14.5 |

Both games show the same qualitative outcome: the agent moved from an
effectively non-functional or losing random policy to a consistent,
statistically stable positive-reward policy, using identical underlying
code.

### 9.3 Interpretation

This result supports the claim that the pipeline's learning mechanism -
not any Breakout-specific tuning - is responsible for the observed
performance improvements. The same convolutional feature extractor,
replay buffer, target network, and Double DQN update rule successfully
learned two visually and mechanically distinct games without
modification, other than the trivial adjustment to the number of output
actions. This is taken as reasonable evidence of generalization within
the class of paddle/ball Atari games, though further testing across a
wider and more diverse set of environments (e.g., navigation-based games,
games with sparser rewards) would be needed to support a broader
generalization claim.

## 10. A Third, Harder Game: Seaquest

To further test generalization and probe the limits of the 2000-episode
training budget, the same pipeline was trained on Seaquest, a
substantially more complex Atari environment involving submarine
navigation, oxygen management, and multiple enemy types, using identical
code to the Breakout and Pong runs (Sections 5 and 9).

### 10.1 Training results

| Episode range | Mean reward | Mean Q-value |
|---|---|---|
| 1-150     | 118.13  | 101.842 |
| 601-750   | 204.93  | 90.085  |
| 1051-1200 | 226.80  | 24.151  |
| 1351-1500 | 461.20  | 44.305  |
| 1651-1800 | 804.00  | 63.696  |
| 1951-2000 | 1120.40 | 81.738  |

Unlike Breakout and Pong, which plateaued shortly after epsilon reached
its floor (around episode 600-1100), Seaquest reward continued climbing
substantially throughout the full 2000-episode run, with no clear
plateau reached by the end of training. Total training time was
approximately 29.8 hours.

### 10.2 Quantitative evaluation

| Game | Mean reward (n=50) | Std dev | Min / Max |
|---|---|---|---|
| Breakout | 10.78   | 4.61   | 3.0 / 24.0   |
| Pong     | 14.14   | 3.32   | 4.0 / 20.0   |
| Seaquest | 1129.60 | 313.57 | 440.0 / 1800.0 |

Note that Seaquest much larger reward magnitude reflects its scoring
scale (larger per-event point values), not a categorically stronger
result than Breakout or Pong; the comparable finding across all three
games is the qualitative shape of learning (substantial, sustained
improvement from an early baseline).

### 10.3 Interpretation

Seaquest continued improvement through the full training budget, in
contrast to the earlier plateaus observed on Breakout and Pong, suggests
that training budget - rather than architecture capacity - was the
binding constraint for this environment specifically. This motivates
extending training well beyond 2000 episodes for harder environments, as
a direction for further work.

## 11. A Long-Run Case Study: MsPacman (20,000 Episodes)

To test whether performance continues to improve indefinitely with more
training, or whether it eventually plateaus or degrades, the pipeline was
trained on MsPacman for 20,000 episodes - ten times the budget used for
Breakout, Pong, and Seaquest - using the resume-capable version of the
training script to survive interruptions across the multi-day run.

### 11.1 Training results

| Episode range | Mean reward | Mean Q-value |
|---|---|---|
| 1-1000       | 759.05  | 35.831  |
| 4953-5952    | 2069.76 | 146.970 |
| 8720-9719    | 2280.84 | 155.682 |
| 11666-12615  | 1963.44 | 167.221 |
| 15616-16615  | 1987.23 | 147.734 |
| 19250-20000  | 1726.07 | 139.393 |

Reward rose steadily and substantially for the first ~9,000 episodes,
reaching a peak average of approximately 2280 around episodes 8720-9719.
Beyond this point, reward gradually declined over the remaining ~10,000
episodes, settling to an average of approximately 1726 by the final 750
episodes - a real, sustained decline rather than short-term noise, as it
persists consistently across many successive 1000-episode windows. Mean
Q-value follows a similar rise-then-decline pattern, peaking around
episode 11,000-12,000 before also declining.

### 11.2 Quantitative evaluation: peak vs. final checkpoint

To confirm this was a genuine regression and not a training-log artifact,
both the peak-region checkpoint (episode 9700) and the final checkpoint
(episode 20000) were evaluated over 50 independent episodes (epsilon=0):

| Checkpoint | Mean reward (n=50) | Std dev | Min / Max |
|---|---|---|---|
| dqn_ep9700.pt (peak)   | 2860.00 | 876.90 | 740.0 / 4830.0  |
| dqn_ep20000.pt (final) | 1691.80 | 494.73 | 680.0 / 3290.0  |

This confirms a substantial, statistically clear decline of approximately
41% in mean reward between the peak-region checkpoint and the final
checkpoint, evaluated identically. This is not attributable to
evaluation noise, given the consistency of the decline across the
training log and its confirmation via independent evaluation.

### 11.3 Interpretation

This result is a genuine and informative negative finding: extending
training well beyond the point of peak performance did not continue to
improve the agent, and in this case measurably harmed it. Plausible
contributing factors, none of which were directly isolated in this
project, include: continued policy drift after the value function has
converged near its practical ceiling, accumulated instability in Q-value
estimation over an unusually long training horizon, and the fixed
learning rate (1e-4) continuing to apply substantial updates long after
the point where a lower rate or a learning-rate schedule might have been
more appropriate. This finding is consistent with a broader known
phenomenon in deep reinforcement learning where training stability
degrades over very long horizons without additional stabilization
techniques.

Practical implication: for a fixed compute budget, checkpoint selection
based on periodic evaluation (rather than training to a fixed,
predetermined episode count and using the final checkpoint by default) is
important - the best-performing model in this run was not the final one.

### 11.4 Data quality note

The training log for this run contains 529 duplicated episode-number
rows, concentrated around episode 3701, caused by the resume mechanism
re-logging a small range of already-completed episodes following a
mid-training restart. This affects only the log file row count (20,751
rows for 20,000 unique episodes) and does not affect the trained model
checkpoints themselves; the chunked analysis above is based on the full
row set and may be marginally skewed in the affected range, but the
overall peak-then-decline pattern is well outside the range this minor
duplication could plausibly explain.
