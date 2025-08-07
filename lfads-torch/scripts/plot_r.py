# plot_subject_r_by_sub.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 之前画sub2-6的准确率图
# def load_subject_r(subject):
#     """Loads train/valid pearson r per epoch (drops epoch 0)."""
#     csv = os.path.join(
#         "C:/UM_Files/Internship/runs/lfads-torch-example",
#         f"my_data_{subject}",
#         "250526_exampleSingle/csv_logs/metrics.csv"
#     )
#     df = pd.read_csv(csv)
#     # grab the per-epoch r's and drop the first row (epoch 0)
#     tr = df["train/pearson_r_epoch"].dropna().values[1:]
#     vr = df["valid/pearson_r_epoch"].dropna().values[1:]
#     return tr, vr

# def main():
#     subjects = [f"sub-{i:02d}" for i in range(2,6)]  # sub-02 … sub-06
#     n = len(subjects)
#     cmap = plt.get_cmap("tab10")  # 10 distinct colors

#     # prepare storage
#     train_stats = []
#     valid_stats = []
#     train_epochs = []
#     valid_epochs = []

#     for i, subj in enumerate(subjects):
#         tr, vr = load_subject_r(subj)
#         train_stats.append((tr.mean(), tr.std()))
#         valid_stats.append((vr.mean(), vr.std()))
#         train_epochs.append(tr)
#         valid_epochs.append(vr)

#     x = np.arange(n)
#     fig, (ax_t, ax_v) = plt.subplots(1,2, figsize=(12,5), sharey=True, tight_layout=True)

#     # ---- TRAIN PANEL ----
#     for i, subj in enumerate(subjects):
#         mean, std = train_stats[i]
#         color = cmap(i)
#         # bar
#         ax_t.bar(x[i], mean, yerr=std, capsize=5, color=color)
#         # dots
#         ax_t.scatter(np.full_like(train_epochs[i], x[i]), train_epochs[i],
#                      color=color, edgecolor="k", alpha=0.8, s=40)

#     ax_t.set_xticks(x)
#     ax_t.set_xticklabels(subjects, rotation=45, ha="right")
#     ax_t.set_ylabel("Pearson r")
#     ax_t.set_title("Training Pearson r (epochs 1–end)")

#     # ---- VALIDATION PANEL ----
#     for i, subj in enumerate(subjects):
#         mean, std = valid_stats[i]
#         color = cmap(i)
#         ax_v.bar(x[i], mean, yerr=std, capsize=5, color=color)
#         ax_v.scatter(np.full_like(valid_epochs[i], x[i]), valid_epochs[i],
#                      color=color, edgecolor="k", alpha=0.8, s=40)

#     ax_v.set_xticks(x)
#     ax_v.set_xticklabels(subjects, rotation=45, ha="right")
#     ax_v.set_title("Validation Pearson r (epochs 1–end)")

#     plt.savefig("pearson_r_sub2-6.png", dpi=300)
#     plt.show()

# if __name__=="__main__":
#     main()





# 第一版 出来的图像全是一个颜色
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def load_subject_r(subject):
#     """
#     Given a subject ID like 'sub-02', load its metrics.csv
#     and return (train_r, valid_r) as 1D numpy arrays.
#     """
#     base = os.path.join(
#         "C:/UM_Files/Internship/runs/lfads-torch-example",
#         f"my_data_{subject}",
#         "250526_exampleSingle",
#         "csv_logs"
#     )
#     csv_path = os.path.join(base, "metrics.csv")
#     if not os.path.isfile(csv_path):
#         raise FileNotFoundError(f"Cannot find metrics for {subject} at {csv_path}")
#     df = pd.read_csv(csv_path)
#     train_r = df["train/pearson_r_epoch"].dropna().values
#     valid_r = df["valid/pearson_r_epoch"].dropna().values
#     return train_r, valid_r

# def main():
#     subjects = [f"sub-{i:02d}" for i in range(2, 7)]  # sub-02 … sub-06

#     train_means, train_stds, train_vals = [], [], []
#     valid_means, valid_stds, valid_vals = [], [], []

#     # Load & compute stats
#     for subj in subjects:
#         tr, vr = load_subject_r(subj)
#         train_means.append(tr.mean())
#         train_stds .append(tr.std())
#         train_vals .append(tr)

#         valid_means.append(vr.mean())
#         valid_stds .append(vr.std())
#         valid_vals .append(vr)

#     x = np.arange(len(subjects))

#     # Plot
#     fig, (ax0, ax1) = plt.subplots(
#         1, 2, figsize=(12,5), sharey=True,
#         gridspec_kw={"wspace":0.3}
#     )

#     # --- Left: Train Pearson r ---
#     ax = ax0
#     ax.bar(x, train_means, yerr=train_stds, capsize=5, color="C0", label="Mean ± SD")
#     # overlay per-epoch points
#     for i, vals in enumerate(train_vals):
#         ax.scatter(
#             np.full_like(vals, x[i]), vals,
#             color="k", alpha=0.7, s=20, label="_nolegend_"
#         )
#     ax.set_xticks(x)
#     ax.set_xticklabels(subjects, rotation=45, ha="right")
#     ax.set_ylabel("Pearson r")
#     ax.set_title("Train Pearson r per Subject")

#     # --- Right: Valid Pearson r ---
#     ax = ax1
#     ax.bar(x, valid_means, yerr=valid_stds, capsize=5, color="C1", label="Mean ± SD")
#     for i, vals in enumerate(valid_vals):
#         ax.scatter(
#             np.full_like(vals, x[i]), vals,
#             color="k", alpha=0.7, s=20, label="_nolegend_"
#         )
#     ax.set_xticks(x)
#     ax.set_xticklabels(subjects, rotation=45, ha="right")
#     ax.set_title("Valid Pearson r per Subject")

#     plt.tight_layout()
#     out_png = "pearson_r_by_subject.png"
#     plt.savefig(out_png, dpi=300)
#     print(f"Saved plot to {out_png}")
#     plt.show()

# if __name__ == "__main__":
#     main()




def find_latest_subfolder(base_path):
    """
    Finds the latest (largest prefix) folder under base_path.
    For folders like '250411_exampleSingle' → picks highest '250411'.
    """
    entries = [e for e in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, e))]
    date_folders = [(int(e.split('_')[0]), e) for e in entries]

    # pick folder with largest prefix
    date_folders.sort(reverse=True)
    return date_folders[0][1]


def load_subject_r(subject):
    """
    Loads train/valid pearson r per epoch for given subject.
    Finds the latest numeric folder automatically.
    """
    base = os.path.join(
        "C:/UM_Files/Internship/runs/lfads-torch-example",
        f"my_data_{subject}"
    )
    latest_folder = find_latest_subfolder(base)
    csv_path = os.path.join(base, latest_folder, "csv_logs", "metrics.csv")
    df = pd.read_csv(csv_path)
    tr = df["train/pearson_r_epoch"].dropna().values[1:]  # drop epoch 0
    vr = df["valid/pearson_r_epoch"].dropna().values[1:]
    return tr, vr

def main():
    subjects = [f"sub-{i:02d}" for i in range(1,11)]  # sub-01 to sub-10
    n = len(subjects)
    cmap = plt.get_cmap("tab10")

    train_stats, valid_stats = [], []
    train_epochs, valid_epochs = [], []

    for i, subj in enumerate(subjects):
        tr, vr = load_subject_r(subj)
        train_stats.append((tr.mean(), tr.std()))
        valid_stats.append((vr.mean(), vr.std()))
        train_epochs.append(tr)
        valid_epochs.append(vr)

    x = np.arange(n)
    fig, (ax_t, ax_v) = plt.subplots(1,2, figsize=(14,5), sharey=True, tight_layout=True)

    # ---- TRAIN PANEL ----
    for i, subj in enumerate(subjects):
        mean, std = train_stats[i]
        color = cmap(i % 10)
        ax_t.bar(x[i], mean, yerr=std, capsize=5, color=color)
        ax_t.scatter(np.full_like(train_epochs[i], x[i]), train_epochs[i],
                     color=color, edgecolor="k", alpha=0.8, s=40)
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(subjects, rotation=45, ha="right")
    ax_t.set_ylabel("Pearson r")
    ax_t.set_title("Training Pearson r (epochs 1–end)")

    # ---- VALIDATION PANEL ----
    for i, subj in enumerate(subjects):
        mean, std = valid_stats[i]
        color = cmap(i % 10)
        ax_v.bar(x[i], mean, yerr=std, capsize=5, color=color)
        ax_v.scatter(np.full_like(valid_epochs[i], x[i]), valid_epochs[i],
                     color=color, edgecolor="k", alpha=0.8, s=40)
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(subjects, rotation=45, ha="right")
    ax_v.set_title("Validation Pearson r (epochs 1–end)")

    plt.savefig("C:/UM_Files/Internship/Result Plot/pearson_r_sub1-10.png", dpi=300, bbox_inches="tight")
    plt.show()

    return subjects, train_stats, valid_stats

if __name__ == "__main__":
    subjects, train_stats, valid_stats = main()


# 后加的 画完PCA轨迹以后对二者间关系的分析
import json
from scipy.stats import pearsonr

# ======== Load Procrustes disparity from saved file ========
with open("C:/UM_Files/Internship/Result_PCA/procrustes_disparities.json", "r") as f:
    disparities = json.load(f)  # dict of {sub-01: val, ...}

# ======== Prepare values ========
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
cmap = plt.get_cmap("tab10")

train_means = [train_stats[i][0] for i in range(10)]
valid_means = [valid_stats[i][0] for i in range(10)]
disparity_vals = [disparities[subj] for subj in subjects]

# ======== Correlation calculation ========
train_rval, train_pval = pearsonr(disparity_vals, train_means)
valid_rval, valid_pval = pearsonr(disparity_vals, valid_means)

# ======== Plot the correlation scatter ========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, subj in enumerate(subjects):
    color = cmap(i % 10)
    # Training
    ax1.scatter(disparity_vals[i], train_means[i], color=color, edgecolor='k', s=60)
    ax1.text(disparity_vals[i]+0.005, train_means[i], subj, fontsize=9)

    # Validation
    ax2.scatter(disparity_vals[i], valid_means[i], color=color, edgecolor='k', s=60)
    ax2.text(disparity_vals[i]+0.005, valid_means[i], subj, fontsize=9)

ax1.set_title(f"Train r vs Disparity\nr={train_rval:.2f}, p={train_pval:.4f}")
ax1.set_xlabel("Procrustes Disparity (vs sub-06)")
ax1.set_ylabel("Mean Training Pearson r")

ax2.set_title(f"Valid r vs Disparity\nr={valid_rval:.2f}, p={valid_pval:.4f}")
ax2.set_xlabel("Procrustes Disparity (vs sub-06)")

plt.tight_layout()
plt.savefig("C:/UM_Files/Internship/Result Plot/correlation_disparity_vs_r.png", dpi=300)
plt.show()
