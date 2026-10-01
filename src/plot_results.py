import csv
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "checkpoints", "training_log.csv")
PLOT_PATH = os.path.join(SCRIPT_DIR, "..", "checkpoints", "reward_curve.png")


def moving_average(values, window=50):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def main():
    episodes, rewards, epsilons = [], [], []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            epsilons.append(float(row["epsilon"]))

    smoothed = moving_average(rewards, window=50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(episodes, rewards, alpha=0.3, label="raw reward")
    ax1.plot(episodes, smoothed, linewidth=2, label="50-episode moving average")
    ax1.set_ylabel("Episode reward")
    ax1.set_title("Argus training progress: Breakout reward over time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(episodes, epsilons, color="orange")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon (exploration rate)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
