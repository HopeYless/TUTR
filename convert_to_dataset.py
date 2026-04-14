"""
Convert raw JSON trajectory data to .pkl files for TUTR training and testing.

Input:
  raw_data/Apr. 12 2026 twoPeople/bed_pose_history.json      -> Robot (index 0)
  raw_data/Apr. 12 2026 twoPeople/obstacles_history_edited.json -> Obstacles (indices 1, 2)

Output:
  dataset/twoPeople_train.pkl
  dataset/twoPeople_test.pkl

Each pkl is a list of (hist, future, neighbor) tuples:
  hist     : np.float32 (OBS_LEN, 6)            [x, y, vx, vy, ax, ay]
  future   : np.float32 (PRED_LEN, 2)           [x, y]
  neighbor : np.float32 (OBS_LEN+PRED_LEN, N-1, 6)
"""

import json
import numpy as np
import pickle
import os

# ── Parameters ────────────────────────────────────────────────────────────────
OBS_LEN    = 8
PRED_LEN   = 12
HORIZON    = OBS_LEN + PRED_LEN          # 20 frames per window
DATA_DIM   = 6                            # x, y, vx, vy, ax, ay

RAW_DIR   = "raw_data/Apr. 12 2026 twoPeople"
ROBOT_FILE = os.path.join(RAW_DIR, "bed_pose_history.json")
OBS_FILE   = os.path.join(RAW_DIR, "obstacles_history_edited.json")
OUT_DIR    = "dataset"

# ── Load JSON ─────────────────────────────────────────────────────────────────
with open(ROBOT_FILE) as f:
    robot_data = json.load(f)

with open(OBS_FILE) as f:
    obs_data = json.load(f)

# Two obstacle track IDs (sorted for reproducibility)
obs_ids = sorted(obs_data.keys())          # ['433', '496']
assert len(obs_ids) == 2, f"Expected 2 obstacles, got {len(obs_ids)}: {obs_ids}"

n_frames = len(robot_data)
for oid in obs_ids:
    assert len(obs_data[oid]) == n_frames, \
        f"Obstacle {oid} has {len(obs_data[oid])} frames, expected {n_frames}"

print(f"Frames       : {n_frames}")
print(f"Agents       : robot (0), obstacle {obs_ids[0]} (1), obstacle {obs_ids[1]} (2)")
print(f"Horizon      : obs={OBS_LEN}  pred={PRED_LEN}  total={HORIZON}")

# ── Extract positions (n_frames, 3, 2) ────────────────────────────────────────
#   Agent 0 = robot
#   Agent 1 = obstacle obs_ids[0]  (first by sorted track id)
#   Agent 2 = obstacle obs_ids[1]
positions = np.zeros((n_frames, 3, 2), dtype=np.float64)

for i, entry in enumerate(robot_data):
    positions[i, 0, 0] = entry["position"]["x"]
    positions[i, 0, 1] = entry["position"]["y"]

for agent_col, oid in enumerate(obs_ids, start=1):
    for i, entry in enumerate(obs_data[oid]):
        positions[i, agent_col, 0] = entry["position"]["x"]
        positions[i, agent_col, 1] = entry["position"]["y"]

# ── Build state matrix (n_frames, 3, 6): [x, y, vx, vy, ax, ay] ──────────────
# Velocity  : backward difference  v[t] = pos[t] - pos[t-1]
#             first frame copies from frame 1 (matching dataloader.py behaviour)
# Acceleration: backward difference  a[t] = v[t] - v[t-1]
#             first frame copies from frame 1

states = np.zeros((n_frames, 3, DATA_DIM), dtype=np.float32)
states[:, :, :2] = positions.astype(np.float32)

vel = np.zeros((n_frames, 3, 2), dtype=np.float32)
vel[1:] = positions[1:] - positions[:-1]   # backward diff
vel[0]  = vel[1]                            # first frame inherits from frame 1
states[:, :, 2:4] = vel

acc = np.zeros((n_frames, 3, 2), dtype=np.float32)
acc[1:] = vel[1:] - vel[:-1]
acc[0]  = acc[1]
states[:, :, 4:6] = acc

# ── Create sliding-window trajectory items ────────────────────────────────────
def make_items(frame_start: int, frame_end: int) -> list:
    """Return list of (hist, future, neighbor) for frames [frame_start, frame_end)."""
    items = []
    n_agents = states.shape[1]
    for t in range(frame_start, frame_end - HORIZON + 1):
        window = states[t : t + HORIZON]           # (HORIZON, 3, 6)
        for agent_i in range(n_agents):
            hist   = window[:OBS_LEN,  agent_i].copy()          # (8, 6)
            future = window[OBS_LEN:,  agent_i, :2].copy()      # (12, 2)
            nbr_idx = [j for j in range(n_agents) if j != agent_i]
            neighbor = window[:, nbr_idx].copy()                 # (20, 2, 6)
            items.append((
                hist.astype(np.float32),
                future.astype(np.float32),
                neighbor.astype(np.float32),
            ))
    return items

all_items = make_items(0, n_frames)

print(f"Frames : 0-{n_frames-1}  ->  {len(all_items)} trajectory items")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

test_path = os.path.join(OUT_DIR, "twoPeople_test.pkl")

with open(test_path, "wb") as f:
    pickle.dump(all_items, f)

print(f"\nSaved: {test_path}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\n-- Sanity check (first item) --")
h, fut, nbr = all_items[0]
print(f"  hist     shape : {h.shape}   dtype: {h.dtype}")
print(f"  future   shape : {fut.shape}  dtype: {fut.dtype}")
print(f"  neighbor shape : {nbr.shape}  dtype: {nbr.dtype}")
print(f"  hist  x,y  (first frame) : {h[0, :2]}")
print(f"  future x,y (first step)  : {fut[0]}")
print(f"  neighbor 0 x,y (t=0)     : {nbr[0, 0, :2]}")
