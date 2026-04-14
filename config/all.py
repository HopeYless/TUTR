# Combined config for training on all ETH-UCY + SDD datasets
OB_RADIUS = 2
OB_HORIZON = 8
PRED_HORIZON = 12
INCLUSIVE_GROUPS = []
model_hidden_dim = 128
n_clusters = 50
smooth_size = 3
random_rotation = True
traj_seg = False

# training
lr = 1e-4
batch_size = 128
dist_threshold = 2
epoch = 100
EPOCH_BATCHES = 100
TEST_SINCE = 500

# testing
PRED_SAMPLES = 20

# evaluation
WORLD_SCALE = 1
