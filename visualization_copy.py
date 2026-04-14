"""
visualization.py
────────────────
Runs the TUTR "all" model on dataset_own/twoPeople_test.pkl and plots
observed trajectories, ground-truth futures, and predicted trajectories.

Usage (from repo root, GPU available):
    python visualization.py

Usage (CPU-only):
    python visualization.py --gpu -1

Outputs are saved to ./visualization_output/
"""

import argparse
import importlib
import math
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")          # headless – works without a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from model import TrajectoryModel

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_own_path", type=str, default="./dataset_own/",
                    help="Folder containing twoPeople_test.pkl")
parser.add_argument("--dataset_name",     type=str, default="twoPeople")
parser.add_argument("--dataset_path",     type=str, default="./dataset/",
                    help="Folder containing motion-modes pkl")
parser.add_argument("--motion_modes_name", type=str,
                    default="eth_hotel_univ_zara01_zara02_sdd",
                    help="Base name of the motion-modes file (without _motion_modes.pkl)")
parser.add_argument("--checkpoint",       type=str, default="./checkpoint/all/best.pth")
parser.add_argument("--hp_config",        type=str, default="config/all.py")
parser.add_argument("--obs_len",          type=int, default=8)
parser.add_argument("--pred_len",         type=int, default=12)
parser.add_argument("--num_k",            type=int, default=20,
                    help="Number of predicted trajectories shown per agent")
parser.add_argument("--max_samples",      type=int, default=None,
                    help="Cap the number of scenarios visualised (None = all)")
parser.add_argument("--cols",             type=int, default=4,
                    help="Subplot columns per figure page")
parser.add_argument("--output_dir",       type=str, default="./visualization_output/")
parser.add_argument("--gpu",              type=str, default="0",
                    help="CUDA device index, or -1 for CPU")
parser.add_argument("--seed",             type=int, default=1)
parser.add_argument("--n_agents",         type=int, default=3,
                    help="Number of agents per window in the dataset. "
                         "Scenarios are laid out as window_0/agent_0, window_0/agent_1, ... "
                         "Agent index 0 = Robot, index 1 = Obstacle 1, index 2 = Obstacle 2. (default: 3)")
args = parser.parse_args()

# Agent-type label derived from scenario index within each window.
# convert_to_dataset.py: index 0 = Robot, index 1 = Obstacle 1 (track 433),
#                        index 2 = Obstacle 2 (track 496).
_AGENT_LABELS = ["Robot", "Obstacle 1", "Obstacle 2"]

def agent_label(scenario_idx: int) -> str:
    pos = scenario_idx % args.n_agents
    if pos < len(_AGENT_LABELS):
        return _AGENT_LABELS[pos]
    return f"Obstacle {pos}"

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Device ────────────────────────────────────────────────────────────────────
if args.gpu == "-1" or not torch.cuda.is_available():
    device = torch.device("cpu")
    print("Running on CPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")
    torch.cuda.manual_seed(args.seed)
    print(f"Running on GPU {args.gpu}")

# ── Hyperparameter config ─────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

# ── Dataset ───────────────────────────────────────────────────────────────────
test_dataset = TrajectoryDataset(
    dataset_path=args.dataset_own_path,
    dataset_name=args.dataset_name,
    dataset_type="test",
    translation=True, rotation=True, scaling=False,
    obs_len=args.obs_len,
)
test_loader = DataLoader(
    test_dataset,
    collate_fn=test_dataset.coll_fn,
    batch_size=1,          # one scenario at a time for per-plot clarity
    shuffle=False,
    num_workers=0,
)
print(f"Loaded {len(test_dataset)} test scenarios from "
      f"{args.dataset_own_path}{args.dataset_name}_test.pkl")

# ── Motion modes ──────────────────────────────────────────────────────────────
motion_modes_file = os.path.join(
    args.dataset_path, args.motion_modes_name + "_motion_modes.pkl"
)
if not os.path.exists(motion_modes_file):
    raise FileNotFoundError(
        f"Motion-modes file not found: {motion_modes_file}\n"
        "Run train_all.py first to generate it."
    )
with open(motion_modes_file, "rb") as fh:
    motion_modes = pickle.load(fh)
motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to(device)
print(f"Motion modes shape: {motion_modes.shape}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = TrajectoryModel(
    in_size=2,
    obs_len=args.obs_len,
    pred_len=args.pred_len,
    embed_size=hp_config.model_hidden_dim,
    enc_num_layers=2,
    int_num_layers_list=[1, 1],
    heads=4,
    forward_expansion=2,
)
model = model.to(device)

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()
print(f"Loaded checkpoint: {args.checkpoint}")

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

# ── Inference & collection ────────────────────────────────────────────────────
results = []   # list of dicts, one per trajectory

with torch.no_grad():
    for batch_idx, (ped, neis, mask) in enumerate(test_loader):
        if args.max_samples is not None and batch_idx >= args.max_samples:
            break

        ped  = ped.to(device)    # [1, obs+pred, 2]
        neis = neis.to(device)   # [1, N, obs+pred, 2]
        mask = mask.to(device)   # [1, N, N]

        ped_obs  = ped[:, :args.obs_len]           # [1, obs, 2]
        gt       = ped[:, args.obs_len:]           # [1, pred, 2]
        neis_obs = neis[:, :, :args.obs_len]       # [1, N, obs, 2]

        pred_trajs, scores = model(
            ped_obs, neis_obs, motion_modes, mask, None,
            test=True, num_k=args.num_k,
        )
        # pred_trajs: [1, num_k, pred*2]
        pred_trajs = pred_trajs.reshape(
            pred_trajs.shape[0], pred_trajs.shape[1], args.pred_len, 2
        )  # [1, num_k, pred, 2]

        # pick best-of-K by ADE
        gt_rep  = gt.unsqueeze(1)                           # [1, 1, pred, 2]
        norms   = torch.norm(pred_trajs - gt_rep, p=2, dim=-1)   # [1, num_k, pred]
        ade_k   = norms.mean(dim=-1)                        # [1, num_k]
        best_k  = ade_k.argmin(dim=-1).item()               # scalar

        results.append({
            "obs":        ped_obs[0].cpu().numpy(),           # [obs, 2]
            "gt":         gt[0].cpu().numpy(),                # [pred, 2]
            "preds":      pred_trajs[0].cpu().numpy(),        # [num_k, pred, 2]
            "best_k":     best_k,
            "min_ade":    ade_k[0, best_k].item(),
            "min_fde":    norms[0, best_k, -1].item(),
            "agent_type": agent_label(batch_idx),
        })

n_samples = len(results)
print(f"Collected predictions for {n_samples} scenarios.")

# ── Per-agent visual style ────────────────────────────────────────────────────
STYLES = {
    "Robot": {
        "obs_color":  "#2980b9",   # blue        – observed history
        "gt_color":   "#1abc9c",   # teal        – ground-truth future
        "pred_color": "#e74c3c",   # red         – predicted trajectories
    },
    "Obstacle 1": {
        "obs_color":  "#e67e22",   # orange      – observed history
        "gt_color":   "#27ae60",   # green       – ground-truth future
        "pred_color": "#8e44ad",   # purple      – predicted trajectories
    },
    "Obstacle 2": {
        "obs_color":  "#c0392b",   # dark red    – observed history
        "gt_color":   "#f39c12",   # amber       – ground-truth future
        "pred_color": "#16a085",   # dark teal   – predicted trajectories
    },
}

# ── Helper: plot a single trajectory panel ────────────────────────────────────
def plot_panel(ax, r, idx):
    obs   = r["obs"]    # [obs, 2]
    gt    = r["gt"]     # [pred, 2]
    preds = r["preds"]  # [num_k, pred, 2]
    best  = r["best_k"]

    style = STYLES.get(r["agent_type"], STYLES["Obstacle 1"])
    obs_c  = style["obs_color"]
    gt_c   = style["gt_color"]
    pred_c = style["pred_color"]

    # -- all predicted trajectories (thin, translucent)
    for k in range(preds.shape[0]):
        ax.plot(preds[k, :, 0], preds[k, :, 1],
                color=pred_c, alpha=0.25, linewidth=0.8, zorder=2)

    # -- best predicted trajectory
    best_pred = np.concatenate([obs[-1:], preds[best]], axis=0)
    ax.plot(best_pred[:, 0], best_pred[:, 1],
            color=pred_c, linewidth=2.0, zorder=4, linestyle="--")
    ax.scatter(preds[best, -1, 0], preds[best, -1, 1],
               color=pred_c, s=40, zorder=5, marker="^")

    # -- ground-truth future
    gt_full = np.concatenate([obs[-1:], gt], axis=0)
    ax.plot(gt_full[:, 0], gt_full[:, 1],
            color=gt_c, linewidth=2.0, zorder=3)
    ax.scatter(gt[-1, 0], gt[-1, 1], color=gt_c, s=40, zorder=5, marker="s")

    # -- observed trajectory
    ax.plot(obs[:, 0], obs[:, 1],
            color=obs_c, linewidth=2.0, zorder=3)
    ax.scatter(obs[0, 0],  obs[0, 1],  color=obs_c, marker="o", s=50, zorder=6)
    ax.scatter(obs[-1, 0], obs[-1, 1], color=obs_c, marker="*", s=80, zorder=6)

    ax.set_title(
        f"Scenario {idx + 1} [{r['agent_type']}]\nADE={r['min_ade']:.3f}  FDE={r['min_fde']:.3f}",
        fontsize=8
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=6)


# ── Shared legend ─────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D

legend_handles = [
    # observed history (solid line)
    Line2D([0], [0], color="#2980b9", linewidth=2, label="Robot – observed"),
    Line2D([0], [0], color="#e67e22", linewidth=2, label="Obstacle 1 – observed"),
    Line2D([0], [0], color="#c0392b", linewidth=2, label="Obstacle 2 – observed"),
    # ground-truth future
    Line2D([0], [0], color="#1abc9c", linewidth=2, label="Robot – ground truth"),
    Line2D([0], [0], color="#27ae60", linewidth=2, label="Obstacle 1 – ground truth"),
    Line2D([0], [0], color="#f39c12", linewidth=2, label="Obstacle 2 – ground truth"),
    # best prediction (dashed)
    Line2D([0], [0], color="#e74c3c", linewidth=2, linestyle="--",
           label="Robot – best pred"),
    Line2D([0], [0], color="#8e44ad", linewidth=2, linestyle="--",
           label="Obstacle 1 – best pred"),
    Line2D([0], [0], color="#16a085", linewidth=2, linestyle="--",
           label="Obstacle 2 – best pred"),
]

# ── Build figure pages ────────────────────────────────────────────────────────
cols       = args.cols
rows_page  = 3                              # rows per saved figure
page_size  = cols * rows_page
n_pages    = math.ceil(n_samples / page_size)

for page in range(n_pages):
    start = page * page_size
    end   = min(start + page_size, n_samples)
    batch = results[start:end]
    n     = len(batch)

    actual_rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        actual_rows, cols,
        figsize=(cols * 3.5, actual_rows * 3.5),
    )
    axes = np.array(axes).reshape(-1)   # always 1-D

    for i, r in enumerate(batch):
        plot_panel(axes[i], r, start + i)

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"TUTR 'all' model  ·  twoPeople_test  ·  page {page + 1}/{n_pages}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    out_path = os.path.join(args.output_dir, f"trajectories_page{page + 1:02d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Summary statistics ────────────────────────────────────────────────────────
all_ade = [r["min_ade"] for r in results]
all_fde = [r["min_fde"] for r in results]
print("\n" + "-" * 40)
print(f"Scenarios visualised : {n_samples}")
print(f"Mean ADE (best-of-K) : {np.mean(all_ade):.4f}")
print(f"Mean FDE (best-of-K) : {np.mean(all_fde):.4f}")
print(f"Plots saved to       : {os.path.abspath(args.output_dir)}")
