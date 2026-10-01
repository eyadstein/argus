import argparse
import os

import ale_py
import gymnasium as gym
import torch

from gymnasium.wrappers import RecordVideo

from model import DQN
from preprocess import FrameStacker

gym.register_envs(ale_py)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "..", "videos")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--game", type=str, default="Breakout")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = gym.make(f"ALE/{args.game}-v5", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        name_prefix=f"argus_{args.game.lower()}_gameplay",
        episode_trigger=lambda ep: True,
    )

    n_actions = env.action_space.n
    net = DQN(n_actions).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    stacker = FrameStacker(k=4, size=84)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        state = stacker.reset(obs)
        total_reward = 0.0
        done = False

        while not done:
            if torch.rand(1).item() < args.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
                    q_values = net(state_t)
                    action = int(q_values.argmax(dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = stacker.step(obs)
            total_reward += reward

        print(f"Recorded episode {episode}: reward = {total_reward}")

    env.close()
    print(f"Videos saved to: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
