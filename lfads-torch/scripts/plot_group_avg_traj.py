from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl

# Folders
avg_dir = Path(r"C:\UM_Files\Internship\Result_PCA")
save_dir = Path(r"C:\UM_Files\Internship\Result Plot")

# Load each subject's average PCA trajectory
avg_trajs = []
for i in range(1, 11):
    sid = f"sub-{i:02d}"
    path = avg_dir / f"{sid}_avg_traj.npy"
    if path.exists():
        avg_trajs.append(np.load(path))
    else:
        print(f"Missing file: {path}")

if len(avg_trajs) != 10:
    print(f"Only {len(avg_trajs)} subjects loaded — averaging anyway.")

# Compute group average
all_avg = np.stack(avg_trajs, axis=0)        # shape: (10, 193, 2)
group_avg = np.mean(all_avg, axis=0)         # shape: (193, 2)
np.save(avg_dir / "group_avg_traj.npy", group_avg)

# Plot with time color
n_points = group_avg.shape[0]
time_vals = np.arange(n_points)
norm = mpl.colors.Normalize(0, n_points - 1)
cmap = plt.get_cmap("plasma")
points = group_avg.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(figsize=(6,6))
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(time_vals)
lc.set_linewidth(2.5)
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Group-Averaged PCA Trajectory (10 subjects)")
fig.colorbar(lc, ax=ax, label="Time (samples)")
plt.tight_layout()
plt.savefig(save_dir / "group_avg_traj.png", dpi=300, bbox_inches="tight")
plt.show()


# ======== Leave-one-out group averaging (10 runs) ========
for excluded_idx in range(1, 11):
    excluded_id = f"sub-{excluded_idx:02d}"
    print(f"🔄 Excluding {excluded_id}")

    # Load remaining subjects
    partial_trajs = []
    for i in range(1, 11):
        sid = f"sub-{i:02d}"
        if sid == excluded_id:
            continue
        path = avg_dir / f"{sid}_avg_traj.npy"
        if path.exists():
            partial_trajs.append(np.load(path))
        else:
            print(f"⚠️ Missing file: {path}")

    if len(partial_trajs) < 9:
        print(f"⚠️ Skipping {excluded_id}: not enough data")
        continue

    # Compute mean of 9 subjects
    partial_avg = np.mean(np.stack(partial_trajs, axis=0), axis=0)
    np.save(avg_dir / f"group_avg_wo_{excluded_id}.npy", partial_avg)

    # Plot
    n_points = partial_avg.shape[0]
    time_vals = np.arange(n_points)
    norm = mpl.colors.Normalize(0, n_points - 1)
    points = partial_avg.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(6,6))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(time_vals)
    lc.set_linewidth(2.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Group-Averaged PCA (without {excluded_id})")
    fig.colorbar(lc, ax=ax, label="Time (samples)")
    plt.tight_layout()
    plt.savefig(save_dir / f"group_avg_wo_{excluded_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

