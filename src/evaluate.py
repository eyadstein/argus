import argparse

import ale_py
import gymnasium as gym
import numpy as np
import torch

from model import DQN
from preprocess import FrameStacker

gym.register_envs(ale_py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--game", type=str, default="Breakout")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(f"ALE/{args.game}-v5", render_mode="rgb_array")
    n_actions = env.action_space.n

    net = DQN(n_actions).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    stacker = FrameStacker(k=4, size=84)
    rewards = []

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        state = stacker.reset(obs)
        total_reward = 0.0
        done = False

        while not done:
            if np.random.rand() < args.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
                    action = int(net(state_t).argmax(dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = stacker.step(obs)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode:3d}/{args.episodes}: reward = {total_reward}")

    env.close()

    rewards = np.array(rewards)
    print("\n---- Evaluation summary ----")
    print(f"Game:       {args.game}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {args.episodes}")
    print(f"Mean:       {rewards.mean():.2f}")
    print(f"Std dev:    {rewards.std():.2f}")
    print(f"Min / Max:  {rewards.min():.1f} / {rewards.max():.1f}")
    print(f"Median:     {np.median(rewards):.1f}")


if __name__ == "__main__":
    main()
