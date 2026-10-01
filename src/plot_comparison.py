import csv
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOUBLE_CSV = os.path.join(SCRIPT_DIR, "..", "checkpoints", "breakout", "double", "training_log.csv")
VANILLA_CSV = os.path.join(SCRIPT_DIR, "..", "checkpoints", "vanilla", "training_log.csv")
PLOT_PATH = os.path.join(SCRIPT_DIR, "..", "checkpoints", "vanilla_vs_double_comparison.png")


def moving_average(values, window=50):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def load_log(path):
    episodes, rewards, qs = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            qs.append(float(row.get("avg_q", 0.0)))
    return episodes, rewards, qs


def main():
    d_ep, d_reward, d_q = load_log(DOUBLE_CSV)
    v_ep, v_reward, v_q = load_log(VANILLA_CSV)

    d_reward_smooth = moving_average(d_reward, window=50)
    v_reward_smooth = moving_average(v_reward, window=50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(d_ep, d_reward_smooth, label="Double DQN", linewidth=2)
    ax1.plot(v_ep, v_reward_smooth, label="Vanilla DQN", linewidth=2, alpha=0.8)
    ax1.set_ylabel("Episode reward (50-ep moving avg)")
    ax1.set_title("Vanilla DQN vs Double DQN: Breakout training comparison")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if any(q != 0.0 for q in d_q):
        ax2.plot(d_ep, d_q, label="Double DQN", linewidth=1.5)
    ax2.plot(v_ep, v_q, label="Vanilla DQN", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Q-value")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
