import argparse
import random
import numpy as np
import torch
import os
import importlib
import pickle

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from model import TrajectoryModel

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str, default='eth')
parser.add_argument('--hp_config', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
parser.add_argument('--motion_modes_name', type=str, default=None,
                    help='Base name used for the motion modes file. Defaults to dataset_name. '
                         'Set to the combined name (e.g. eth_hotel_univ_zara1_zara2_sdd) '
                         'when evaluating a model trained with train_all.py.')
parser.add_argument('--motion_modes_dir', type=str, default=None,
                    help='Directory containing the motion modes pkl. '
                         'Defaults to --dataset_path. Override when the motion modes '
                         'live in a different folder than the test data (e.g. ./dataset/).')
parser.add_argument('--checkpoint_name', type=str, default=None,
                    help='Subdirectory under --checkpoint containing best.pth. '
                         'Defaults to dataset_name. Set to "all" for a combined model.')
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

# Resolve defaults for optional name overrides
motion_modes_name = args.motion_modes_name or args.dataset_name
motion_modes_dir  = args.motion_modes_dir  or args.dataset_path
checkpoint_name   = args.checkpoint_name   or args.dataset_name

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Load hyperparameter config
spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

# Load test dataset
test_dataset = TrajectoryDataset(
    dataset_path=args.dataset_path,
    dataset_name=args.dataset_name,
    dataset_type='test',
    translation=True, rotation=True, scaling=False,
    obs_len=args.obs_len
)
test_loader = DataLoader(
    test_dataset, collate_fn=test_dataset.coll_fn,
    batch_size=hp_config.batch_size, shuffle=False, num_workers=0
)

# Load motion modes
motion_modes_file = os.path.join(motion_modes_dir, motion_modes_name + '_motion_modes.pkl')
if not os.path.exists(motion_modes_file):
    raise FileNotFoundError(
        f"Motion modes file not found: {motion_modes_file}\n"
        "Run train.py (or train_all.py) first to generate it."
    )
with open(motion_modes_file, 'rb') as f:
    motion_modes = pickle.load(f)
motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

# Build model and load checkpoint
model = TrajectoryModel(
    in_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
    embed_size=hp_config.model_hidden_dim,
    enc_num_layers=2, int_num_layers_list=[1, 1],
    heads=4, forward_expansion=2
)
model = model.cuda()

checkpoint_path = os.path.join(args.checkpoint, checkpoint_name, 'best.pth')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path))
print(f"Loaded checkpoint: {checkpoint_path}")

# Evaluate
model.eval()
ade_total = 0
fde_total = 0
num_traj = 0

with torch.no_grad():
    for (ped, neis, mask) in test_loader:
        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

        gt_ = gt.unsqueeze(1)
        norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
        ade_ = torch.mean(norm_, dim=-1)
        fde_ = norm_[:, :, -1]

        min_ade, _ = torch.min(ade_, dim=-1)
        min_fde, _ = torch.min(fde_, dim=-1)

        ade_total += torch.sum(min_ade).item()
        fde_total += torch.sum(min_fde).item()
        num_traj += ped_obs.shape[0]

ade = ade_total / num_traj
fde = fde_total / num_traj

print(f"\nDataset : {args.dataset_name}")
print(f"Trajectories: {num_traj}")
print(f"ADE : {ade:.4f}")
print(f"FDE : {fde:.4f}")
