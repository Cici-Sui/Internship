# Plot the latent factors' repersentation in the space using PCA
# Plot 2018论文里的 figure 3，隐变量和各种rotation
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from lfads_torch.model import LFADS
from lfads_torch.tuples import SessionBatch


import os
import re
subject_id = "sub-10"
base_dir = Path("runs/lfads-torch-example") / f"my_data_{subject_id}"
save_dir = Path(r"C:\UM_Files\Internship\Result Plot")
save_dir2 = Path(r"C:\UM_Files\Internship\Result_PCA")

train_folders = [
    f for f in os.listdir(base_dir)
    if os.path.isdir(base_dir / f) and re.match(r"^\d+_", f)
]

if not train_folders:
    raise FileNotFoundError(f"No training folders found for {subject_id}")

# Sort by numeric prefix
train_folders.sort(key=lambda x: int(x.split('_')[0]), reverse=True)
training_folder = train_folders[0]
print(f"✔ Found training folder: {training_folder}")

# ---- Step 2: Find first numbered checkpoint file ----
ckpt_dir = base_dir / training_folder / "lightning_checkpoints"
ckpt_files = [
    f for f in os.listdir(ckpt_dir)
    if re.search(r"\d", f) and f.endswith(".ckpt")
]

if not ckpt_files:
    raise FileNotFoundError(f"No numbered .ckpt files found in {ckpt_dir}")

ckpt_file = sorted(ckpt_files)[0]
print(f"✔ Found checkpoint: {ckpt_file}")

# ---- Final path ----
ckpt_path = ckpt_dir / ckpt_file
print(f"Full checkpoint path: {ckpt_path}")

# Load the trained model, taking one session as an example
# cwd == C:\UM_Files\Internship, relitive path
initialize(config_path="../configs", job_name="load")
cfg = compose(
    config_name="single",
    overrides=[
        "model=my_model",        
        "datamodule=my_datamodule"
    ],
)
#Override the datamodule's file‐pattern to the real path
#(this will replace whatever was in the YAML)
cfg.datamodule.datafile_pattern = os.path.join(
    os.getcwd(),           # C:\UM_Files\Internship
    "features", "final_data", f"{subject_id}.h5"
)

# Instantiate *all* modules (LFADS + its read-in/read-out/etc.) exactly as during training
model: LFADS = instantiate(cfg.model)
model.eval()



ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt["state_dict"])

# Load all data without spliting into training/validating
dm = instantiate(cfg.datamodule)
dm.setup("fit")
# debug prints
print("Pattern I’m using:", dm.hparams.datafile_pattern)
from glob import glob
files = sorted(glob(dm.hparams.datafile_pattern))
print("Files matched:", files)
print("Number of train_ds sessions:", len(dm.train_ds))
print("Number of valid_ds sessions:", len(dm.valid_ds))


train_ds = dm.train_ds[0]
val_ds   = dm.valid_ds[0]
print("train_ds type:", type(train_ds))
print("val_ds   type:", type(val_ds))

from torch.utils.data import ConcatDataset, DataLoader
full_ds = ConcatDataset([train_ds, val_ds])
print("full_ds type:", type(full_ds))

full_loader = DataLoader(
    full_ds,
    batch_size=32,    # your training batch size
    shuffle=False,
    drop_last=False,
)
# print("✔ full_loader is a", type(full_loader))  # should be DataLoader
# Sanity check right before the loop

# Run the model, collecting latent factors
# all_factors = [] 这里多写了一行
all_factors = []
with torch.no_grad():
    for session_batch, _extras in full_loader:
        # run the model
        outs = model(session_batch, sample_posteriors=False, output_means=False)
        # extract the (B,T,F) tensor
        fac = outs[0].factors
        all_factors.append(fac.cpu().numpy())

# Concatenate → shape (N_total_windows, 100, fac_dim)
factors = np.concatenate(all_factors, axis=0)  


# Stitch data back to 2D for applying PCA
def stitch_windows(windows: np.ndarray, shift: int) -> np.ndarray:
    # windows.shape == (n_w, window_len, fac_dim)
    n_w, T, F   = windows.shape
    overlap     = T - shift
    L           = T + (n_w - 1) * shift
    full        = np.zeros((L, F), dtype=windows.dtype)
    full[:T]    = windows[0]
    for i in range(1, n_w):
        start = overlap + (i - 1) * shift
        tail  = windows[i, overlap:]
        full[start:start+shift] = tail
    return full

# e.g. with shift=10:
continuous = stitch_windows(factors, shift=10)
# continuous.shape == (L, fac_dim)

# Run PCA with all components
pca_full = PCA()
pca_full.fit(continuous)

# Plot variance explained
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Find how many components explain at least 90%
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
print(f"Number of components to explain 90% variance: {n_components_90}")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# --- (A) Scree plot: each PC’s variance ---
axs[0].plot(np.arange(1, len(explained_var)+1), explained_var, marker='o')
axs[0].set_title(f"{subject_id}: Explained Variance per Component")
axs[0].set_xlabel("Principal Component")
axs[0].set_ylabel("Explained Variance Ratio")
axs[0].grid(True)

# Add percentage labels for first 5 PCs
for i in range(5):
    val = explained_var[i]
    axs[0].text(i+1, val + 0.005, f"{val*100:.1f}%", ha='center', fontsize=9)

# --- (B) Cumulative explained variance ---
axs[1].plot(np.arange(1, len(cumulative_var)+1), cumulative_var, marker='o')
axs[1].axhline(0.90, color='r', linestyle='--', label="90% threshold")
axs[1].axvline(n_components_90, color='r', linestyle='--')
axs[1].set_title(f"{subject_id}: Cumulative Explained Variance")
axs[1].set_xlabel("Number of Components")
axs[1].set_ylabel("Cumulative Variance Ratio")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(save_dir / f"{subject_id}_PCA_variance.png", dpi=300, bbox_inches="tight")
plt.show()



# Apply PCA (only the first two)
pca = PCA(n_components=2)
pc2 = pca.fit_transform(continuous)    # shape (L, 2)



#Plot the all single trials PCA result

words_path = Path(fr"C:/UM_Files/Internship/features/{subject_id}_procWords.npy")

sample_labels = np.load(words_path, allow_pickle=True)

L = min(len(sample_labels), pc2.shape[0])
sample_labels = sample_labels[:L]
pc2           = pc2[:L]

# 截取单词窗口
segments_fixed = []
segments_variable = []

offset = 0
fixed_len = 193
margin = 0 # for the individual plots (10 sub plots)

# ---- First: variable-length segmentation ----
in_seg = False
for i, lbl in enumerate(sample_labels):
    if not in_seg and lbl not in (b"", "", None):
        in_seg = True
        seg_start = i
        current_word = lbl
    elif in_seg and lbl in (b"", "", None):
        segments_variable.append((seg_start, i - 1, current_word))
        in_seg = False
if in_seg:
    # finalize the last word
    segments_variable.append((seg_start, len(sample_labels) - 1, current_word))

# ---- Then: fixed-length segmentation based on variable ones ----
for start, _end, word in segments_variable:
    t0 = start - offset
    t1 = t0 + fixed_len
    if t0 < 0 or t1 > len(pc2):
        continue  # skip if out of bounds
    segments_fixed.append((t0, t1, word))

print("\n--- segments_fixed (for averaging) ---")
for i, (t0, t1, word) in enumerate(segments_fixed):
    w = word.decode() if isinstance(word, (bytes, bytearray)) else str(word)
    print(f"{i:3d}: {w:<20} | length: {t1 - t0}")

print("\n--- segments_variable (for per-word plots) ---")
for i, (start, end, word) in enumerate(segments_variable):
    w = word.decode() if isinstance(word, (bytes, bytearray)) else str(word)
    print(f"{i:3d}: {w:<20} | length: {end - start + 1}")


# 新的画图
resampled_trajs = []

for t0, t1, _ in segments_fixed:
    seg = pc2[t0:t1]  # should be shape (193, 2)
    if seg.shape[0] != 193:
        print(f"Warning: segment from {t0} to {t1} has wrong shape")
        continue
    resampled_trajs.append(seg)

import matplotlib as mpl
from matplotlib.collections import LineCollection
# Stack and average
all_trajs = np.stack(resampled_trajs, axis=0)  # shape: (n_trials, seg_length, 2)
avg_traj = np.mean(all_trajs, axis=0)          # shape: (seg_length, 2)

#save the calculated averaged pca
np.save(save_dir2 / f"{subject_id}_avg_traj.npy", avg_traj)


n_points = all_trajs.shape[1]
time_vals = np.arange(n_points)
norm = mpl.colors.Normalize(0, n_points - 1)
cmap = plt.get_cmap("plasma")

# ====== 1. Plot individual trials ======
fig1, ax1 = plt.subplots(figsize=(6, 6))
for traj in all_trajs:
    points = traj.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(time_vals)
    lc.set_linewidth(1)
    ax1.add_collection(lc)

ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.set_title(f"{subject_id} - Single-trial PCA trajectories")
ax1.autoscale()
ax1.set_aspect('equal')
fig1.colorbar(lc, ax=ax1, label="Time (samples)")
plt.tight_layout()
plt.show()
# Save F1
fig1.savefig(save_dir / f"{subject_id}_F1.png", dpi=300, bbox_inches="tight")


# ====== 2. Plot averaged trajectory ======
fig2, ax2 = plt.subplots(figsize=(6, 6))
points = avg_traj.reshape(-1, 1, 2)
plot_segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc_avg = LineCollection(plot_segments, cmap=cmap, norm=norm)
lc_avg.set_array(time_vals)
lc_avg.set_linewidth(2.5)
ax2.add_collection(lc_avg)

ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_title(f"{subject_id} - Averaged PCA trajectory across all trials")
ax2.autoscale()
ax2.set_aspect('equal')
fig2.colorbar(lc_avg, ax=ax2, label="Time (samples)")
plt.tight_layout()
plt.show()
# Save F2
fig2.savefig(save_dir / f"{subject_id}_F2.png", dpi=300, bbox_inches="tight")
# Plot setup
# fig, axs = plt.subplots(2, 5, figsize=(15, 6))
# axs = axs.flatten()
# # norm = mpl.colors.Normalize(0, 219)
# cmap = plt.get_cmap("plasma")
# time_vals = np.arange(220)
# 这些都是之前固定长度的10个词的图像
# for ax, (t0, t1, word) in zip(axs, selected_segments):
#     seg = pc2[t0:t1]
#     if seg.shape[0] != 220:
#         print(f"Skipping word {word} due to wrong length ({seg.shape[0]})")
#         ax.axis('off')
#         continue

#     points = seg.reshape(-1, 1, 2)
#     segments_plot = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments_plot, cmap=cmap, norm=norm)
#     lc.set_array(time_vals)
#     lc.set_linewidth(2)
#     ax.add_collection(lc)

#     # Formatting
#     ax.set_title(word.decode() if isinstance(word, (bytes, bytearray)) else str(word), fontsize=10)
#     ax.set_aspect('equal')
#     ax.autoscale()
#     ax.set_xticks([])
#     ax.set_yticks([])

# # Shared colorbar
# fig.subplots_adjust(right=0.88)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.set_label("Time (samples)")

# plt.suptitle("Latent PCA Trajectories for First 5 and Last 5 Words", fontsize=14)
# plt.tight_layout(rect=[0, 0, 0.88, 0.95])
# plt.show()

# 画十个示例
# Use first 5 and last 5 variable-length segments
selected_segments = segments_variable[:5] + segments_variable[-5:]

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
cmap = plt.get_cmap("plasma")

for ax, (start, end, word) in zip(axs, selected_segments):
    t0 = max(0, start - margin)
    t1 = min(len(pc2), end + margin + 1)
    seg = pc2[t0:t1]
    length = seg.shape[0]
    print(f"Word: {word}, segment length: {length} samples")

    if length < 2:
        ax.axis('off')
        continue

    points = seg.reshape(-1, 1, 2)
    plot_segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mpl.colors.Normalize(0, length - 1)
    lc = LineCollection(plot_segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(length))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Proper labels
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    w = word.decode() if isinstance(word, (bytes, bytearray)) else str(word)
    ax.set_title(f"{w} ({length} pts)", fontsize=10)
    ax.autoscale()
    ax.set_aspect('equal')

# Optional shared colorbar — symbolic only
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 100))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Time (samples — varies by word)")

plt.suptitle(f"{subject_id} - Per-Word PCA Trajectories (First 5 + Last 5 Words)", fontsize=14)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()
# Save F3
fig.savefig(save_dir / f"{subject_id}_F3.png", dpi=300, bbox_inches="tight")

print("Finished")




