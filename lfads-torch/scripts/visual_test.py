#检查不同人说的词
import h5py
import numpy as np
from pathlib import Path
from collections import Counter

def word_labels_per_window(sample_labels, n_windows, win_len, hop):
    labels_win = []
    for i in range(n_windows):
        chunk = sample_labels[i*hop : i*hop + win_len]
        if chunk.size == 0:
            labels_win.append(b"")
        else:
            lbl, _ = Counter(chunk).most_common(1)[0]
            labels_win.append(lbl)
    return np.array(labels_win, dtype=sample_labels.dtype)

def inspect_subject(subj):
    prefix = f"[{subj}]"
    print(f"\n{prefix} ===== Inspecting {subj} =====")

    # Paths
    base_run  = Path("C:/UM_Files/Internship/runs/lfads-torch-example") / f"my_data_{subj}" / "250526_exampleSingle"
    h5_file   = base_run / f"lfads_output_{subj}.h5"
    words_file= Path("C:/UM_Files/Internship/features") / f"{subj}_procWords.npy"

    # 1) HDF5 exists?
    if not h5_file.exists():
        print(f"{prefix} ERROR: HDF5 not found: {h5_file}")
        return
    print(f"{prefix} HDF5 file:           {h5_file}")

    # 2) .npy exists?
    if not words_file.exists():
        print(f"{prefix} ERROR: procWords.npy not found: {words_file}")
        return
    print(f"{prefix} Word-label file:     {words_file}")

    # 3) Load spectrogram windows
    with h5py.File(h5_file, "r") as f:
        valid_recon = f["valid_recon_data"][()]
    n_windows, win_len, freq_bins = valid_recon.shape
    print(f"{prefix} valid_recon_data:    windows={n_windows}, win_len={win_len}, freq_bins={freq_bins}")

    # 4) Load sample-level labels
    samples = np.load(words_file, allow_pickle=True)
    print(f"{prefix} sample_labels.shape: {samples.shape}")
    print(f"{prefix} first 7 samples:    {samples[:7].tolist()}")

    # 5) Convert to per-window labels
    hop = 10
    win_labels = word_labels_per_window(samples, n_windows, win_len, hop)
    print(f"{prefix} window_labels.shape: {win_labels.shape}")
    print(f"{prefix} first 7 windows:    {[w.decode() for w in win_labels[:7]]}")

    # 6) Unique non-empty words
    uniq, counts = np.unique(win_labels, return_counts=True)
    filtered = [(u.decode(), c) for u, c in zip(uniq, counts) if u and u != b""]
    print(f"{prefix} unique words ({len(filtered)}):")
    for w, c in filtered:
        print(f"{prefix}    {w!r}: {c} windows")

if __name__ == "__main__":
    for subj in ("sub-02", "sub-06"):
        inspect_subject(subj)




# 检查words的
# assign one word‐label per spectrogram window
# from collections import Counter

# def word_labels_per_window(sample_labels, n_windows, win_len, hop):
#     """
#     sample_labels: 1D array of length >= (n_windows-1)*hop + win_len
#     returns: list of length n_windows, where each entry is the most common
#              label in that window's sample range
#     """
#     lbls = []
#     for i in range(n_windows):
#         start = i * hop
#         end   = start + win_len
#         w_lbls = sample_labels[start:end]
#         if len(w_lbls)==0:
#             lbls.append(b"")          # or ""
#         else:
#             # pick the most common (mode)
#             lbl, _ = Counter(w_lbls).most_common(1)[0]
#             lbls.append(lbl)
#     return np.array(lbls, dtype=sample_labels.dtype)

# # DEBUG script
# import h5py, numpy as np
# from pathlib import Path
# from collections import Counter

# # 1) load windows
# h5 = Path(r"C:/UM_Files/Internship/runs/lfads-torch-example/my_data_sub-02/250526_exampleSingle/lfads_output_sub-02.h5")
# with h5py.File(h5,"r") as f:
#     valid_orig = f["valid_recon_data"][()]
# n_windows, win_len, n_bins = valid_orig.shape
# print("n_windows, win_len, n_bins:", valid_orig.shape)

# # 2) load sample-level labels
# words_all = np.load(r"C:/UM_Files/Internship/features/sub-02_procWords.npy", allow_pickle=True)
# print("sample labels length:", words_all.shape)

# # 3) compute window-level labels
# hop = 10
# labels_win = word_labels_per_window(words_all, n_windows, win_len, hop)
# print("window-level labels shape:", labels_win.shape)
# print("unique window labels & counts:")
# uniq, cnts = np.unique(labels_win, return_counts=True)
# for u, c in zip(uniq, cnts):
#     print(u, c)

# # pick first 5 non-empty labels for example
# examples = [w for w in uniq if w]  # drop b'' if present
# examples = examples[:5]
# print("plotting these words:", examples)

# # now you can slice windows by label:
# idxs = np.where(labels_win == examples[0])[0]
# print(f"windows for word {examples[0]!r}:", idxs[:10], "... total", len(idxs))

