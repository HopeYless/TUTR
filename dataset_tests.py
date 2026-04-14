import pickle
import numpy as np

with open("dataset/twoPeople_train.pkl", "rb") as f:
    scenarios = pickle.load(f)

print(f"Total scenarios: {len(scenarios)}")

# Inspect the first scenario
hist, future, neighbor = scenarios[0]
print(f"hist     shape: {hist.shape}")      # [obs_len, 2]
print(f"future   shape: {future.shape}")    # [pred_len, 2]
print(f"neighbor shape: {neighbor.shape}")  # [obs_len+pred_len, N, 2]

# Loop over a few scenarios
for i, (hist, future, neighbor) in enumerate(scenarios[:5]):
    print(f"\n--- Scenario {i} ---")
    print(f"  hist start:    {hist[0]}")       # first observed position
    print(f"  hist end:      {hist[-1]}")       # last observed position
    print(f"  future target: {future[-1]}")     # final predicted position
    print(f"  n_neighbors:   {neighbor.shape[1]}")
    # Robot is always neighbor index 0 (when not the focal agent)
    print(f"  robot neighbor pos (t=0): {neighbor[0, 0]}")
