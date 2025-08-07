# Use the Procrustes check the similarity between the averaged PCA plot for subjects

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from pathlib import Path


# Directories
data_dir = Path(r"C:\UM_Files\Internship\Result_PCA")
save_dir = Path(r"C:\UM_Files\Internship\Result Plot")
save_dir.mkdir(parents=True, exist_ok=True)

# Reference
reference_id = "sub-06"
reference_traj = np.load(data_dir / f"{reference_id}_avg_traj.npy")

# Store disparities
disparities = {}

# Loop through subjects
for i in range(1, 11):
    subject_id = f"sub-{i:02d}"
    if subject_id == reference_id:
        disparities[subject_id] = 0.0  # assign 0 for the reference
        continue

    # Load this subject's average trajectory
    traj_path = data_dir / f"{subject_id}_avg_traj.npy"
    if not traj_path.exists():
        print(f"{subject_id}: file not found, skipping.")
        continue
    traj = np.load(traj_path)

    # Check shape
    if traj.shape != reference_traj.shape:
        print(f"{subject_id}: shape mismatch ({traj.shape} vs {reference_traj.shape}), skipping.")
        continue

    # Run Procrustes
    mtx1, mtx2, disparity = procrustes(reference_traj, traj)
    disparities[subject_id] = disparity
    print(f"{subject_id}: Procrustes disparity vs sub-06 = {disparity:.6f}")

    # collect aligned trajectories
    if "aligned_trajs" not in locals():
        aligned_trajs = {reference_id: mtx1}
    aligned_trajs[subject_id] = mtx2

#Plot the 10 PCA pairs with same refference frame
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.figure(figsize=(8, 8))
ax = plt.gca()
cmap = cm.get_cmap("tab10")

for idx, (subj, traj) in enumerate(aligned_trajs.items()):
    color = cmap(idx % 10)
    x, y = traj[:, 0], traj[:, 1]

    # Break trajectory into segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Plot the smooth trajectory line
    lc = LineCollection(
        segments,
        color=color,
        linewidth=2,
        linestyle='solid',
        zorder=1
    )
    ax.add_collection(lc)

    # Add arrows along the trajectory
    from matplotlib.patches import FancyArrowPatch
    arrow_step = 40
    start_idx = len(x) // 4  # quarter-way point

    for i in range(start_idx, len(x) - 1, arrow_step):
        arrow = FancyArrowPatch(
            (x[i], y[i]), (x[i+1], y[i+1]),
            arrowstyle='->',
            color=color,
            mutation_scale=14,   # larger arrowhead
            linewidth=1.2,
            zorder=10,
        )
        ax.add_patch(arrow)




    ax.plot([], [], color=color, label=subj)  # dummy for legend

ax.set_xlabel("PC 1 (aligned)")
ax.set_ylabel("PC 2 (aligned)")
ax.set_title("All Aligned PCA Trajectories with Arrows")
ax.axis("equal")
ax.legend()
plt.tight_layout()
plt.savefig(save_dir / "all_subjects_aligned_arrows.png", dpi=300, bbox_inches="tight")
plt.show()



# Plot summary bar chart
subjects = list(disparities.keys())
disparity_vals = [disparities[s] for s in subjects]

plt.figure(figsize=(10,6))
plt.bar(subjects, disparity_vals, color='skyblue', edgecolor='k')
plt.ylabel("Procrustes Disparity vs sub-06")
plt.xlabel("Subject")
plt.title("Shape differences across subjects vs sub-06")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(save_dir / "disparities_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()


print("Done comparing all subjects to sub-06.")


# ===== Compute group average from aligned trajectories =====
aligned_sub_ids = list(aligned_trajs.keys())
aligned_trajs_arr = np.stack([aligned_trajs[sub] for sub in aligned_sub_ids])  # shape: (10, 193, 2)
group_avg = np.mean(aligned_trajs_arr, axis=0)
np.save(data_dir / "group_avg_aligned.npy", group_avg)

# Plot group average
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib as mpl

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
ax.set_xlabel("PC 1 (aligned)")
ax.set_ylabel("PC 2 (aligned)")
ax.set_title("Group-Averaged Aligned PCA Trajectory (10 subjects)")
fig.colorbar(lc, ax=ax, label="Time (samples)")
plt.tight_layout()
plt.savefig(save_dir / "group_avg_traj_aligned.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== Leave-one-out versions =====
for excluded_id in aligned_sub_ids:
    trajs_wo = [aligned_trajs[s] for s in aligned_sub_ids if s != excluded_id]
    if len(trajs_wo) < 2:
        continue
    group_wo = np.mean(np.stack(trajs_wo), axis=0)
    np.save(data_dir / f"group_avg_aligned_wo_{excluded_id}.npy", group_wo)

    points = group_wo.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(6,6))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(time_vals)
    lc.set_linewidth(2.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel("PC 1 (aligned)")
    ax.set_ylabel("PC 2 (aligned)")
    ax.set_title(f"Group Avg Aligned Trajectory (wo {excluded_id})")
    fig.colorbar(lc, ax=ax, label="Time (samples)")
    plt.tight_layout()
    plt.savefig(save_dir / f"group_avg_traj_aligned_wo_{excluded_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Save disparities to file so plot_r.py can access them
import json
with open("C:/UM_Files/Internship/Result_PCA/procrustes_disparities.json", "w") as f:
    json.dump(disparities, f, indent=2)
