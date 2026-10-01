import argparse
import glob
import os
import re
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent
from preprocess import FrameStacker

gym.register_envs(ale_py)

BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
BUFFER_CAPACITY = 100_000
MIN_BUFFER_BEFORE_TRAINING = 5_000
TARGET_UPDATE_EVERY = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 200_000
CHECKPOINT_EVERY = 100


def get_epsilon(step):
    fraction = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "dqn_ep*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None, 0

    def extract_ep(path):
        match = re.search(r"dqn_ep(\d+)\.pt", path)
        return int(match.group(1)) if match else -1

    latest = max(candidates, key=extract_ep)
    return latest, extract_ep(latest)


def read_global_step_at_episode(log_file, target_episode):
    if not os.path.exists(log_file):
        return 0
    with open(log_file) as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",")
        if int(parts[0]) == target_episode:
            return int(parts[2])
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["double", "vanilla"], default="double")
    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    env_name = f"ALE/{args.game}-v5"
    use_double_dqn = (args.mode == "double")

    checkpoint_dir = os.path.join("checkpoints", args.game.lower(), args.mode)
    log_file = os.path.join(checkpoint_dir, "training_log.csv")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_name, render_mode="rgb_array")
    n_actions = env.action_space.n
    stacker = FrameStacker(k=4, size=84)

    agent = DQNAgent(
        n_actions=n_actions,
        device=device,
        lr=LR,
        gamma=GAMMA,
        buffer_capacity=BUFFER_CAPACITY,
        use_double_dqn=use_double_dqn,
    )

    start_episode = 1
    global_step = 0

    if args.resume:
        ckpt_path, latest_ep = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            agent.online_net.load_state_dict(torch.load(ckpt_path, map_location=device))
            agent.target_net.load_state_dict(agent.online_net.state_dict())
            global_step = read_global_step_at_episode(log_file, latest_ep)
            start_episode = latest_ep + 1
            print(f"Resuming from {ckpt_path} (episode {latest_ep}, global_step {global_step})")
        else:
            print("No existing checkpoint found - starting fresh despite --resume.")

    if start_episode == 1:
        with open(log_file, "w") as f:
            f.write("episode,reward,steps,epsilon,avg_loss,avg_q,elapsed_sec\n")

    print(f"Training on device: {device} | game: {args.game} | mode: {args.mode} "
          f"(use_double_dqn={use_double_dqn}) | episodes {start_episode}->{args.episodes}")

    start_time = time.time()

    for episode in range(start_episode, args.episodes + 1):
        obs, info = env.reset()
        state = stacker.reset(obs)
        episode_reward = 0.0
        episode_losses = []
        episode_qs = []
        done = False

        while not done:
            epsilon = get_epsilon(global_step)
            action = agent.select_action(state, epsilon)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = stacker.step(obs)

            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            global_step += 1

            if len(agent.buffer) >= MIN_BUFFER_BEFORE_TRAINING:
                loss, avg_q = agent.train_step(BATCH_SIZE)
                episode_losses.append(loss)
                episode_qs.append(avg_q)

            if global_step % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        avg_q = float(np.mean(episode_qs)) if episode_qs else 0.0
        elapsed = time.time() - start_time

        print(
            f"[{args.mode}] Episode {episode:5d}/{args.episodes} | reward {episode_reward:6.1f} | "
            f"steps {global_step:9d} | epsilon {epsilon:.3f} | "
            f"avg_loss {avg_loss:.5f} | avg_q {avg_q:8.3f} | elapsed {elapsed/60:.1f} min"
        )

        with open(log_file, "a") as f:
            f.write(f"{episode},{episode_reward},{global_step},{epsilon},{avg_loss},{avg_q},{elapsed:.1f}\n")

        if episode % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dqn_ep{episode}.pt")
            torch.save(agent.online_net.state_dict(), ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
