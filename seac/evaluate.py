# from sheets_api.sheets.sheets import update_values
import json
import os
import sys

import gym
import numpy as np
import rware
import torch
from a2c import A2C
from scipy import stats as sts
from wrappers import RecordEpisodeStatistics, TimeLimit

sys.path.append("C:/Users/smithr38/Code/School/Thesis-Project/")
from custom_utils.animations import save_animation

model_name = "u2002500"
path = "C:/Users/smithr38/Code/School/Thesis-Project/results/trained_models/1/" + model_name
env_name = "rware-tiny-2ag-v1"
time_limit = 500  # 25 for LBF

GSHEET_ROW = None
EPISODES = 1000
RECORD = False
RECORD_OUTLIERS = False
SAVE_STATS = False

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

all_rewards = []
all_collisions = []
all_deliveries = []
outliers = {}
stats = {
    "metadata": {
        "model": model_name,
        "env": env_name,
        "time_limit": time_limit,
        "num_episodes": EPISODES,
        "num_people": env.n_people,
    },
    "averages": {},
    "episodes": [],
}
for ep in range(EPISODES):
    print(f"Episode: {ep + 1} / {EPISODES}", end="\r")

    env = gym.make(env_name)
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)
    seed = env.seed()[0]
    obs = env.reset()
    done = [False] * len(agents)

    action_list = []
    frames = []
    while not all(done):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        if RECORD_OUTLIERS:
            action_list.append(actions.copy())
        if RECORD:
            frames.append(env.render(mode="rgb_array"))
        obs, _, done, info = env.step(actions)
    obs = env.reset()
    env.close()

    if RECORD:
        # Create the directory if it doesn't exist
        output_dir = f"C:/Users/smithr38/Code/School/Thesis-Project/videos/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        save_animation(frames, os.path.join(output_dir, f"ep{ep+1}.mp4"))

    total_reward = sum(info["episode_reward"])
    total_collisions = sum(info["collisions"])
    total_deliveries = sum(info["deliveries"])
    all_rewards.append(total_reward)
    all_collisions.append(total_collisions)
    all_deliveries.append(total_deliveries)

    save_outlier = RECORD_OUTLIERS and (total_reward < 0 or total_collisions > 10 or total_deliveries < 1)

    if save_outlier:
        outliers[ep] = {"actions": action_list, "seed": seed}

    if SAVE_STATS:
        episode_stats = {
            "episode": ep + 1,
            "seed": seed,
            "rewards": {
                "individual": list(info["episode_reward"]),
                "total": total_reward,
            },
            "collisions": {
                "individual": list(info["collisions"]),
                "total": total_collisions,
            },
            "deliveries": {
                "individual": list(info["deliveries"]),
                "total": total_deliveries,
            },
        }
        if save_outlier:
            episode_stats["actions"] = action_list
        stats["episodes"].append(episode_stats)


# Calculate 95% confidence intervals
def calculate_confidence_interval(data):
    mean = np.mean(data)
    sem = sts.sem(data)
    confidence_interval = sts.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
    return (confidence_interval[1] - confidence_interval[0]) / 2


"""
if GSHEET_ROW:
    # Write results to google sheets if a row num is provided
    update_values(
        [
            [
                min(all_rewards),
                max(all_rewards),
                sum(all_rewards) / EPISODES,
                min(all_collisions),
                max(all_collisions),
                sum(all_collisions) / EPISODES,
                min(all_deliveries),
                max(all_deliveries),
                sum(all_deliveries) / EPISODES,
            ]
        ],
        GSHEET_ROW,
    )
"""

if SAVE_STATS:
    stats["averages"] = {
        "rewards": {
            "min": min(all_rewards),
            "max": max(all_rewards),
            "average": sum(all_rewards) / EPISODES,
            "std": np.std(all_rewards),
            "confidence_interval": calculate_confidence_interval(all_rewards),
        },
        "collisions": {
            "min": min(all_collisions),
            "max": max(all_collisions),
            "average": sum(all_collisions) / EPISODES,
            "std": np.std(all_collisions),
            "confidence_interval": calculate_confidence_interval(all_collisions) if np.mean(all_collisions) > 0 else 0,
        },
        "deliveries": {
            "min": min(all_deliveries),
            "max": max(all_deliveries),
            "average": sum(all_deliveries) / EPISODES,
            "std": np.std(all_deliveries),
            "confidence_interval": calculate_confidence_interval(all_deliveries),
        },
    }

    # Log stats to file with incrementing suffix
    base_filename = f"C:/Users/smithr38/Code/School/Thesis-Project/stats/{model_name}.json"
    suffix = 0
    while os.path.exists(base_filename):
        suffix += 1
        base_filename = f"C:/Users/smithr38/Code/School/Thesis-Project/stats/{model_name}_{suffix}.json"

    with open(base_filename, "w") as f:
        json.dump(stats, f, indent=4)

# Print stats
print(f"--- Averages Over {EPISODES} Episodes ---")
print(
    f"Average rewards: {sum(all_rewards) / EPISODES} | Min: {min(all_rewards)} | Max: {max(all_rewards)} | STD: {np.std(all_rewards)} | 95% CI: {calculate_confidence_interval(all_rewards)}"
)
print(
    f"Average collisions: {sum(all_collisions) / EPISODES} | Min: {min(all_collisions)} | Max: {max(all_collisions)} | STD: {np.std(all_collisions)} | 95% CI: {calculate_confidence_interval(all_collisions) if np.mean(all_collisions) > 0 else 0}"
)
print(
    f"Average deliveries: {sum(all_deliveries) / EPISODES} | Min: {min(all_deliveries)} | Max: {max(all_deliveries)} | STD: {np.std(all_deliveries)} | 95% CI: {calculate_confidence_interval(all_deliveries)}"
)
print("---")

# Save outlier episodes
if len(outliers) > 0:
    # Create the directory
    output_dir = f"C:/Users/smithr38/Code/School/Thesis-Project/videos/{model_name}_outliers"
    suffix = 0
    while os.path.exists(output_dir):
        suffix += 1
        output_dir = f"C:/Users/smithr38/Code/School/Thesis-Project/videos/{model_name}_outliers_{suffix}"
    os.makedirs(output_dir)

    for i, (ep, outlier) in enumerate(outliers.items()):
        print(f"Rendering outlier: {i + 1} / {len(outliers)}", end="\r")

        # Initialize the environment
        o_frames = []
        o_env = gym.make(env_name)
        o_env = TimeLimit(o_env, time_limit)
        o_env = RecordEpisodeStatistics(o_env)
        o_env.seed(outlier["seed"])
        o_env.reset()

        # Render the environment
        o_frames.append(o_env.render(mode="rgb_array"))
        for action in outlier["actions"]:
            o_env.step(action)
            o_frames.append(o_env.render(mode="rgb_array"))
        o_env.close()
        save_animation(o_frames, os.path.join(output_dir, f"ep_{ep+1}.mp4"))
