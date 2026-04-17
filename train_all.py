import argparse
import random
import numpy as np
import torch
import os
import importlib
import pickle

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader, ConcatDataset
from model import TrajectoryModel
from torch import optim
import torch.nn.functional as F
from utils import get_motion_modes

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['eth', 'hotel', 'univ', 'zara01', 'zara02', 'sdd'], # Training dataset names
                    help='List of datasets to train on')
parser.add_argument('--hp_config', type=str, default='config/all.py')
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
parser.add_argument('--patience', type=int, default=20,
                    help='Early stopping: stop if mean ADE+FDE does not improve for this many epochs')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

print(f"Training on datasets: {args.datasets}")


class CombinedDataset(ConcatDataset):
    """Wraps ConcatDataset to expose the coll_fn and scenario_list needed by TUTR."""

    def __init__(self, datasets):
        super().__init__(datasets)
        # All datasets share identical coll_fn logic and same obs_len/dist_threshold
        self._ref = datasets[0]

    def coll_fn(self, batch):
        return self._ref.coll_fn(batch)

    @property
    def scenario_list(self):
        combined = []
        for d in self.datasets:
            combined.extend(d.scenario_list)
        return combined


# ── Build combined train dataset ─────────────────────────────────────────────
train_datasets = []
for name in args.datasets:
    pkl = os.path.join(args.dataset_path, f"{name}_train.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Missing {pkl} — run get_data_pkl.py for '{name}' first.")
    train_datasets.append(
        TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=name,
                          dataset_type='train', translation=True, rotation=True,
                          scaling=True, obs_len=args.obs_len,
                          dist_threshold=hp_config.dist_threshold, smooth=False)
    )

combined_train = CombinedDataset(train_datasets)
print(f"Total training trajectories: {len(combined_train)}")

# ── Motion modes from the combined training set ───────────────────────────────
combined_name = '_'.join(args.datasets)
motion_modes_file = os.path.join(args.dataset_path, f"{combined_name}_motion_modes.pkl")

if not os.path.exists(motion_modes_file):
    print("Generating motion modes from combined dataset ...")
    motion_modes = get_motion_modes(
        combined_train, args.obs_len, args.pred_len,
        hp_config.n_clusters, args.dataset_path, combined_name,
        smooth_size=hp_config.smooth_size,
        random_rotation=hp_config.random_rotation,
        traj_seg=hp_config.traj_seg
    )
else:
    print("Loading motion modes ...")
    with open(motion_modes_file, 'rb') as f:
        motion_modes = pickle.load(f)

motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

# ── Build per-dataset test loaders ───────────────────────────────────────────
test_loaders = {}
for name in args.datasets:
    pkl = os.path.join(args.dataset_path, f"{name}_test.pkl")
    if not os.path.exists(pkl):
        print(f"Warning: missing {pkl}, skipping test for '{name}'")
        continue
    ds = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=name,
                           dataset_type='test', translation=True, rotation=True,
                           scaling=False, obs_len=args.obs_len)
    test_loaders[name] = DataLoader(ds, collate_fn=ds.coll_fn,
                                    batch_size=hp_config.batch_size,
                                    shuffle=False, num_workers=0)

train_loader = DataLoader(combined_train, collate_fn=combined_train.coll_fn,
                          batch_size=hp_config.batch_size,
                          shuffle=True, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model = TrajectoryModel(
    in_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
    embed_size=hp_config.model_hidden_dim,
    enc_num_layers=2, int_num_layers_list=[1, 1],
    heads=4, forward_expansion=2
)
model = model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp_config.lr)
reg_criterion = torch.nn.SmoothL1Loss().cuda()
cls_criterion = torch.nn.CrossEntropyLoss().cuda()


def get_cls_label(gt, motion_modes):
    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)
    mm = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)
    distance = torch.norm(gt - mm, dim=-1)
    soft_label = F.softmax(-distance, dim=-1)
    closest_mode_indices = torch.argmin(distance, dim=-1)
    return soft_label, closest_mode_indices


def train_epoch(model, optimizer, loader, motion_modes):
    model.train()
    losses = []
    for ped, neis, mask in loader:
        ped, neis, mask = ped.cuda(), neis.cuda(), mask.cuda()

        scale = (torch.randn(ped.shape[0]) * 0.05 + 1).cuda().reshape(-1, 1, 1)
        ped = ped * scale
        neis = neis * scale.reshape(-1, 1, 1, 1)

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt, motion_modes)

        optimizer.zero_grad()
        pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices)
        # pred_trajs: [B, num_train_k, pred_len*2]
        # Pick the prediction closest to GT for the regression loss (best-of-k)
        reg_label = gt.reshape(gt.shape[0], -1)                                         # [B, pred_len*2]
        distances = torch.norm(pred_trajs - reg_label.unsqueeze(1), dim=-1)             # [B, num_train_k]
        best_idx = distances.argmin(dim=-1)                                             # [B]
        best_pred = pred_trajs[torch.arange(pred_trajs.shape[0]).cuda(), best_idx]      # [B, pred_len*2]
        reg_loss = reg_criterion(best_pred, reg_label)
        clf_loss = cls_criterion(scores.squeeze(), soft_label)
        loss = reg_loss + clf_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def test_epoch(model, loader, motion_modes):
    model.eval()
    ade_total, fde_total, num_traj = 0, 0, 0
    with torch.no_grad():
        for ped, neis, mask in loader:
            ped, neis, mask = ped.cuda(), neis.cuda(), mask.cuda()
            ped_obs = ped[:, :args.obs_len]
            gt = ped[:, args.obs_len:]
            neis_obs = neis[:, :, :args.obs_len]

            pred_trajs, _ = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

            norm_ = torch.norm(pred_trajs - gt.unsqueeze(1), p=2, dim=-1)
            min_ade, _ = torch.min(torch.mean(norm_, dim=-1), dim=-1)
            min_fde, _ = torch.min(norm_[:, :, -1], dim=-1)

            ade_total += torch.sum(min_ade).item()
            fde_total += torch.sum(min_fde).item()
            num_traj += ped_obs.shape[0]

    return ade_total / num_traj, fde_total / num_traj, num_traj


# ── Training loop ─────────────────────────────────────────────────────────────
checkpoint_dir = os.path.join(args.checkpoint, 'all')
os.makedirs(checkpoint_dir, exist_ok=True)

best_score = float('inf')
epochs_no_improve = 0

for ep in range(hp_config.epoch):
    avg_loss = train_epoch(model, optimizer, train_loader, motion_modes)

    # Evaluate on every test set
    results = {}
    for name, loader in test_loaders.items():
        ade, fde, n = test_epoch(model, loader, motion_modes)
        results[name] = (ade, fde, n)

    mean_ade = np.mean([v[0] for v in results.values()])
    mean_fde = np.mean([v[1] for v in results.values()])
    combined_score = mean_ade + mean_fde

    if combined_score < best_score:
        best_score = combined_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
    else:
        epochs_no_improve += 1

    detail = '  '.join(f"{n}: ADE={v[0]:.3f} FDE={v[1]:.3f}" for n, v in results.items())
    print(f"Epoch {ep:3d} | loss={avg_loss:.4f} | mean ADE={mean_ade:.3f} FDE={mean_fde:.3f} | {detail}")

    if args.patience > 0 and epochs_no_improve >= args.patience:
        print(f'Early stopping at epoch {ep}: no improvement for {args.patience} epochs.')
        break
