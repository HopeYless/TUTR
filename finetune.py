"""
Fine-tune a pretrained TUTR model.

Only the output-side layers are trained; everything else is frozen:
  social_decoder  (model.social_decoder) - cross-attention to neighbors
  traj_head       (model.reg_head)        - outputs trajectory coordinates
  prob_head       (model.cls_head)        - outputs mode probabilities

Usage example:
  python finetune.py \
    --dataset_name twoPeople \
    --hp_config config/all.py \
    --pretrained checkpoint/all/best.pth \
    --checkpoint checkpoint/ \
    --gpu 0
"""

import argparse
import random
import importlib
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from dataset import TrajectoryDataset
from model import TrajectoryModel
from utils import get_motion_modes

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path',   type=str, default='./dataset_own/')
parser.add_argument('--dataset_name',   type=str, default='twoPeople')
parser.add_argument('--hp_config',      type=str, default='config/all.py',
                    help='Hyper-parameter config file (reuse a base config or supply a custom one)')
parser.add_argument('--pretrained',     type=str, required=True,
                    help='Path to the pretrained checkpoint (.pth) to fine-tune from')
parser.add_argument('--checkpoint',     type=str, default='./checkpoint/')
parser.add_argument('--lr',             type=float, default=None,
                    help='Override learning rate from config')
parser.add_argument('--epochs',         type=int,  default=None,
                    help='Override number of epochs from config')
parser.add_argument('--patience',       type=int,  default=20,
                    help='Early stopping patience (epochs without improvement)')
parser.add_argument('--obs_len',        type=int,  default=8)
parser.add_argument('--pred_len',       type=int,  default=12)
parser.add_argument('--data_scaling',   type=list, default=[1.9, 0.4])
parser.add_argument('--num_works',      type=int,  default=0)
parser.add_argument('--seed',           type=int,  default=1)
parser.add_argument('--gpu',            type=str,  default='0')

args = parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

# ── Hyper-parameter config ────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

lr     = args.lr     if args.lr     is not None else hp_config.lr
epochs = args.epochs if args.epochs is not None else hp_config.epoch

# ── Datasets ──────────────────────────────────────────────────────────────────
train_dataset = TrajectoryDataset(
    dataset_path=args.dataset_path, dataset_name=args.dataset_name,
    dataset_type='train', translation=True, rotation=True,
    scaling=True, obs_len=args.obs_len,
    dist_threshold=hp_config.dist_threshold, smooth=False,
)
test_dataset = TrajectoryDataset(
    dataset_path=args.dataset_path, dataset_name=args.dataset_name,
    dataset_type='test', translation=True, rotation=True,
    scaling=False, obs_len=args.obs_len,
)

train_loader = DataLoader(train_dataset, collate_fn=train_dataset.coll_fn,
                          batch_size=hp_config.batch_size, shuffle=True,
                          num_workers=args.num_works)
test_loader  = DataLoader(test_dataset,  collate_fn=test_dataset.coll_fn,
                          batch_size=hp_config.batch_size, shuffle=False,
                          num_workers=args.num_works)

# ── Motion modes ──────────────────────────────────────────────────────────────
motion_modes_file = args.dataset_path + args.dataset_name + '_motion_modes.pkl'

if not os.path.exists(motion_modes_file):
    print('Motion modes not found — generating ...')
    motion_modes = get_motion_modes(
        train_dataset, args.obs_len, args.pred_len, hp_config.n_clusters,
        args.dataset_path, args.dataset_name,
        smooth_size=hp_config.smooth_size,
        random_rotation=hp_config.random_rotation,
        traj_seg=hp_config.traj_seg,
    )
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

if os.path.exists(motion_modes_file):
    print('Motion modes loading ...')
    with open(motion_modes_file, 'rb') as f:
        motion_modes = pickle.load(f)
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

# ── Model ─────────────────────────────────────────────────────────────────────
model = TrajectoryModel(
    in_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
    embed_size=hp_config.model_hidden_dim,
    enc_num_layers=2, int_num_layers_list=[1, 1],
    heads=4, forward_expansion=2,
)
model = model.cuda()

# Load pretrained weights
print(f'Loading pretrained weights from: {args.pretrained}')
state_dict = torch.load(args.pretrained, map_location='cuda')
model.load_state_dict(state_dict)
print('Pretrained weights loaded.')

# ── Freeze / unfreeze ─────────────────────────────────────────────────────────
# Map conceptual fine-tune names to actual model attribute names.
#   "social_decoder" -> model.social_decoder  (cross-attention to neighbors)
#   "traj_head"      -> model.reg_head         (outputs trajectory coordinates)
#   "prob_head"      -> model.cls_head         (outputs mode probabilities)
FINETUNE_MODULES = {
    "social_decoder": model.social_decoder,
    "traj_head":      model.reg_head,
    "prob_head":      model.cls_head,
}

# First freeze everything
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze only the chosen layers
for name, module in FINETUNE_MODULES.items():
    for param in module.parameters():
        param.requires_grad = True
    print(f'Unfrozen : {name}')

# Report parameter counts
total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {trainable_params:,} trainable / {total_params:,} total '
      f'({100*trainable_params/total_params:.1f}%)')

# ── Optimiser & losses ────────────────────────────────────────────────────────
optimizer     = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
reg_criterion = torch.nn.SmoothL1Loss().cuda()
cls_criterion = torch.nn.CrossEntropyLoss().cuda()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_cls_label(gt, motion_modes):
    gt           = gt.reshape(gt.shape[0], -1).unsqueeze(1)          # [B 1 pred_len*2]
    modes        = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance     = torch.norm(gt - modes, dim=-1)                     # [B K]
    soft_label   = F.softmax(-distance, dim=-1)                       # [B K]
    closest_idx  = torch.argmin(distance, dim=-1)                     # [B]
    return soft_label, closest_idx


def train_one_epoch(model, optimizer, loader, motion_modes):
    model.train()
    losses = []
    for ped, neis, mask in loader:
        ped  = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()

        # Random scale augmentation (same as train.py)
        scale = (torch.randn(ped.shape[0]) * 0.05 + 1).cuda()
        ped  = ped  * scale.reshape(-1, 1, 1)
        neis = neis * scale.reshape(-1, 1, 1, 1)

        ped_obs  = ped[:, :args.obs_len]
        gt       = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            soft_label, closest_idx = get_cls_label(gt, motion_modes)

        optimizer.zero_grad()
        pred_traj, scores = model(ped_obs, neis_obs, motion_modes, mask, closest_idx)
        reg_loss = reg_criterion(pred_traj, gt.reshape(pred_traj.shape))
        clf_loss = cls_criterion(scores.squeeze(), soft_label)
        loss = reg_loss + clf_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, loader, motion_modes):
    model.eval()
    ade, fde, n_traj = 0.0, 0.0, 0

    for ped, neis, mask in loader:
        ped  = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()

        ped_obs  = ped[:, :args.obs_len]
        gt       = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        pred_trajs, _ = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

        gt_     = gt.unsqueeze(1)
        norm_   = torch.norm(pred_trajs - gt_, p=2, dim=-1)
        ade_    = torch.mean(norm_, dim=-1)
        fde_    = norm_[:, :, -1]
        min_ade = torch.min(ade_, dim=-1).values
        min_fde = torch.min(fde_, dim=-1).values

        ade    += min_ade.sum().item()
        fde    += min_fde.sum().item()
        n_traj += ped_obs.shape[0]

    return ade / n_traj, fde / n_traj, n_traj


# ── Fine-tuning loop ──────────────────────────────────────────────────────────
save_dir = os.path.join(args.checkpoint, args.dataset_name + '_finetuned')
os.makedirs(save_dir, exist_ok=True)

best_ade, best_fde = 99.0, 99.0
epochs_no_improve  = 0
best_epoch         = 0

for ep in range(epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader, motion_modes)
    ade, fde, n_traj = evaluate(model, test_loader, motion_modes)

    improved = (ade + fde) < (best_ade + best_fde)
    if improved:
        best_ade, best_fde = ade, fde
        best_epoch = ep
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
    else:
        epochs_no_improve += 1

    print(
        f'epoch: {ep:4d}  loss: {train_loss:.4f}  '
        f'ade: {ade:.4f}  fde: {fde:.4f}  '
        f'best_ade: {best_ade:.4f}  best_fde: {best_fde:.4f}  '
        f'(best epoch: {best_epoch})  n_traj: {n_traj}'
    )

    if args.patience > 0 and epochs_no_improve >= args.patience:
        print(f'Early stopping at epoch {ep}: no improvement for {args.patience} epochs.')
        break

print(f'\nFine-tuning complete. Best checkpoint saved to: {save_dir}/best.pth')
