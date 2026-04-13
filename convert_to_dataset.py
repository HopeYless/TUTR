"""
Convert robot pose and obstacle tracking JSON files to TUTR pkl dataset format.

Reads:
  raw_data/Apr. 12 2026 twoPeople/bed_pose_history.json
  raw_data/Apr. 12 2026 twoPeople/obstacles_history_edited.json

Writes (by default):
  dataset/twoPeople_train.pkl
  dataset/twoPeople_test.pkl

Usage:
  python convert_to_dataset.py
  python convert_to_dataset.py --stride 4 --obs_len 8 --pred_len 12
  python convert_to_dataset.py --train_ratio 0 --dataset_name twoPeople_test_only
  python convert_to_dataset.py --no_robot   # exclude the robot; predict only people

Data format produced (matches dataset.py TrajectoryDataset):
  Each scenario is a 3-tuple (hist, future, neighbor):
    hist     np.float32  [obs_len, 2]            observed x,y of the focal agent
    future   np.float32  [pred_len, 2]            ground-truth future x,y
    neighbor np.float32  [obs_len+pred_len, N, 2] all other agents, time-first

The raw data is collected at ~10 Hz.  The default stride of 4 down-samples to
~2.5 Hz, giving observation windows of ~3.2 s and prediction horizons of ~4.8 s
– comparable to the ETH/UCY benchmark conventions used by TUTR.
"""

import json
import pickle
import numpy as np
import os
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_files(robot_path: Path, obstacles_path: Path):
    with open(robot_path) as f:
        robot_data = json.load(f)
    with open(obstacles_path) as f:
        obstacles_data = json.load(f)
    return robot_data, obstacles_data


def build_agent_arrays(robot_data, obstacles_data, include_robot: bool = True):
    """
    Align all agents on the shared timestamp grid and return x,y arrays.

    All three sources (robot, track 496, track 433) were recorded at the same
    timestamps, so no interpolation is needed.

    Returns
    -------
    times  : np.ndarray  shape [T]       Unix timestamps
    agents : dict        agent_id -> np.ndarray shape [T, 2]  (x, y)
    """
    times = np.array([e["time"] for e in robot_data], dtype=np.float64)
    agents = {}

    if include_robot:
        agents["robot"] = np.array(
            [[e["position"]["x"], e["position"]["y"]] for e in robot_data],
            dtype=np.float32,
        )

    for track_id, track_entries in obstacles_data.items():
        agents[track_id] = np.array(
            [[e["position"]["x"], e["position"]["y"]] for e in track_entries],
            dtype=np.float32,
        )

    return times, agents


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def create_scenarios(agents, obs_len: int, pred_len: int, stride: int):
    """
    Build sliding-window (hist, future, neighbor) tuples from aligned arrays.

    Parameters
    ----------
    agents  : dict  agent_id -> np.ndarray [T, 2]
    obs_len : int   number of observation frames (after subsampling)
    pred_len: int   number of prediction frames (after subsampling)
    stride  : int   subsampling step over the raw 10-Hz frames

    Returns
    -------
    list of (hist, future, neighbor) triples
    """
    total_len = obs_len + pred_len
    agent_ids = list(agents.keys())
    T_raw = next(iter(agents.values())).shape[0]

    # Subsampled frame indices into the raw arrays
    sub_indices = list(range(0, T_raw, stride))

    if len(sub_indices) < total_len:
        raise ValueError(
            f"Not enough subsampled frames ({len(sub_indices)}) for a single "
            f"window of length {total_len}.  Reduce --stride or --obs_len/--pred_len."
        )

    scenarios = []
    n_windows = len(sub_indices) - total_len + 1

    for w in range(n_windows):
        raw_idx = sub_indices[w : w + total_len]  # length total_len

        for ped_id in agent_ids:
            ped_xy = agents[ped_id][raw_idx]          # [total_len, 2]
            hist   = ped_xy[:obs_len]                 # [obs_len,   2]
            future = ped_xy[obs_len:]                 # [pred_len,  2]

            other_ids = [aid for aid in agent_ids if aid != ped_id]
            if other_ids:
                # Stack along axis-1 to produce [total_len, N, 2]  (time-first)
                neighbor = np.stack(
                    [agents[nid][raw_idx] for nid in other_ids], axis=1
                )
            else:
                # No real neighbours; fill with sentinel so distance filter drops it
                neighbor = np.full((total_len, 1, 2), 1e9, dtype=np.float32)

            scenarios.append(
                (
                    hist.astype(np.float32),
                    future.astype(np.float32),
                    neighbor.astype(np.float32),
                )
            )

    return scenarios, n_windows


# ---------------------------------------------------------------------------
# Train / test split  (temporal – no window leakage between splits)
# ---------------------------------------------------------------------------

def temporal_split(scenarios, n_windows: int, n_agents: int, train_ratio: float):
    """
    Split scenarios into train and test preserving temporal order.

    Scenarios are laid out as:
        window_0/agent_0, window_0/agent_1, ...,
        window_1/agent_0, ...
    The first `train_ratio` fraction of windows go to train, the rest to test.
    """
    train_cut = int(n_windows * train_ratio)
    train, test = [], []
    for i, s in enumerate(scenarios):
        window_idx = i // n_agents
        if window_idx < train_cut:
            train.append(s)
        else:
            test.append(s)
    return train, test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Convert robot/obstacle JSON data to TUTR pkl dataset format."
    )
    parser.add_argument(
        "--robot_json",
        default="raw_data/Apr. 12 2026 twoPeople/bed_pose_history.json",
        help="Path to robot pose JSON (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--obstacles_json",
        default="raw_data/Apr. 12 2026 twoPeople/obstacles_history_edited.json",
        help="Path to obstacles JSON (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--obs_len", type=int, default=8,
        help="Observation length in subsampled frames (default: 8)",
    )
    parser.add_argument(
        "--pred_len", type=int, default=12,
        help="Prediction length in subsampled frames (default: 12)",
    )
    parser.add_argument(
        "--stride", type=int, default=4,
        help=(
            "Subsampling stride over raw 10-Hz frames. "
            "stride=4 → ~2.5 Hz, matching ETH/UCY conventions (default: 4)"
        ),
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help=(
            "Fraction of windows used for the train split (0–1). "
            "Use 0 to write only a test file (default: 0.7)"
        ),
    )
    parser.add_argument(
        "--output_dir", default="./dataset",
        help="Directory to write pkl files (default: ./dataset)",
    )
    parser.add_argument(
        "--dataset_name", default="twoPeople",
        help="Base name for output files: <name>_train.pkl / <name>_test.pkl",
    )
    parser.add_argument(
        "--no_robot", action="store_true",
        help="Exclude the robot from the agent set; predict only the tracked people",
    )
    args = parser.parse_args()

    # Resolve input paths
    robot_path     = Path(args.robot_json)
    obstacles_path = Path(args.obstacles_json)
    if not robot_path.is_absolute():
        robot_path = script_dir / robot_path
    if not obstacles_path.is_absolute():
        obstacles_path = script_dir / obstacles_path

    print(f"Loading  robot      : {robot_path}")
    print(f"Loading  obstacles  : {obstacles_path}")

    robot_data, obstacles_data = load_json_files(robot_path, obstacles_path)
    times, agents = build_agent_arrays(
        robot_data, obstacles_data, include_robot=not args.no_robot
    )

    T_raw = next(iter(agents.values())).shape[0]
    n_sub = len(range(0, T_raw, args.stride))
    hz    = 1.0 / (0.1 * args.stride)

    print(f"\nAgents            : {list(agents.keys())}")
    print(f"Raw frames        : {T_raw}  (~10 Hz,  {times[-1]-times[0]:.1f} s)")
    print(f"Subsampled frames : {n_sub}  (~{hz:.1f} Hz,  stride={args.stride})")
    print(f"Window length     : {args.obs_len + args.pred_len} frames  "
          f"(obs={args.obs_len}, pred={args.pred_len})")

    scenarios, n_windows = create_scenarios(
        agents, args.obs_len, args.pred_len, args.stride
    )

    n_agents = len(agents)
    print(f"Windows           : {n_windows}")
    print(f"Total scenarios   : {len(scenarios)}  ({n_windows} windows × {n_agents} agents)")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_ratio > 0:
        train_scenarios, test_scenarios = temporal_split(
            scenarios, n_windows, n_agents, args.train_ratio
        )

        train_path = os.path.join(args.output_dir, f"{args.dataset_name}_train.pkl")
        test_path  = os.path.join(args.output_dir, f"{args.dataset_name}_test.pkl")

        with open(train_path, "wb") as f:
            pickle.dump(train_scenarios, f)
        print(f"\nSaved {len(train_scenarios):4d} train scenarios -> {train_path}")

        with open(test_path, "wb") as f:
            pickle.dump(test_scenarios, f)
        print(f"Saved {len(test_scenarios):4d} test  scenarios -> {test_path}")
    else:
        test_path = os.path.join(args.output_dir, f"{args.dataset_name}_test.pkl")
        with open(test_path, "wb") as f:
            pickle.dump(scenarios, f)
        print(f"\nSaved {len(scenarios)} test scenarios -> {test_path}")

    # Quick sanity check on the first scenario
    s0 = scenarios[0]
    print(
        f"\nSanity check (first scenario):\n"
        f"  hist     shape: {s0[0].shape}   dtype: {s0[0].dtype}\n"
        f"  future   shape: {s0[1].shape}   dtype: {s0[1].dtype}\n"
        f"  neighbor shape: {s0[2].shape}   dtype: {s0[2].dtype}"
    )


if __name__ == "__main__":
    main()
