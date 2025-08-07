# 画音频图谱结果图像，老师论文里的图五
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

subject_id = "sub-01"
save_dir = Path(r"C:\UM_Files\Internship\Result Plot")

def stitch_windows(data: np.ndarray, shift: int) -> np.ndarray:
    """
    Stitch overlapped windows (n_w, T, F) → (time, freq):
    take full first window, then append each window’s 'shift' tail.
    """
    n_w, T, F = data.shape
    if n_w == 0:
        raise ValueError("No windows to stitch!")
    overlap = T - shift
    L       = T + (n_w - 1) * shift
    full    = np.zeros((L, F), dtype=data.dtype)
    full[:T] = data[0]
    for i in range(1, n_w):
        start = T + (i - 1) * shift
        tail  = data[i, overlap:]
        full[start:start+shift] = tail
    return full  # shape (time, freq)

def word_labels_per_window(sample_labels: np.ndarray,
                           n_windows: int,
                           win_len: int,
                           hop: int) -> np.ndarray:
    """
    Map sample‐level labels to one label per spectrogram window
    by taking the mode of each window's sample labels.
    """
    labels_win = []
    for i in range(n_windows):
        chunk = sample_labels[i*hop : i*hop + win_len]
        if chunk.size == 0:
            labels_win.append(b"")
        else:
            lbl, _ = Counter(chunk).most_common(1)[0]
            labels_win.append(lbl)
    return np.array(labels_win, dtype=sample_labels.dtype)

def main():
    # ——— Configuration ———
    h5_path    = Path(r"C:/UM_Files/Internship/"
                      "runs/lfads-torch-example/"
                      "my_data_sub-01/"
                      "250606_exampleSingle/"
                      "lfads_output_sub-01.h5")
    words_path = Path(r"C:/UM_Files/Internship/features/sub-01_procWords.npy")
    hop        = 10       # frame‐shift in time‐bins
    
    margin     = 5        # extra windows before/after each word for context

    # ——— Load spectrogram windows ———
    with h5py.File(h5_path, "r") as f:
        orig_wins = f["valid_recon_data"][()]    # (599, win_len, freq_bins)
        rec_wins  = f["valid_output_params"][()]

    n_windows, win_len, freq_bins = orig_wins.shape
    print(f"[DEBUG] windows: {n_windows}, win_len: {win_len}, freq_bins: {freq_bins}")

    # ——— Load sample‐level labels & compute per‐window labels ———
    sample_labels = np.load(words_path, allow_pickle=True)
    labels_win    = word_labels_per_window(sample_labels, n_windows, win_len, hop)

    # ——— Identify unique non‐empty words & pick first n_words ———
    uniq, counts  = np.unique(labels_win, return_counts=True)
    uniq_words    = [w for w in uniq if w and w != b""][:5]
    print(f"[DEBUG] plotting words: {[w.decode() for w in uniq_words]}")

    # ——— For each word: select indices ± margin, stitch & collect ———
    orig_segs = []
    rec_segs  = []
    seg_lengths = []
    for w in uniq_words:
        idxs = np.where(labels_win == w)[0]
        start, end = idxs.min(), idxs.max()
        s0 = max(0,    start - margin)
        s1 = min(n_windows, end + margin + 1)
        win_idx = np.arange(s0, s1)
        print(f"[DEBUG] word {w.decode()}: windows {s0}→{s1-1}")

        seg_o = stitch_windows(orig_wins[win_idx], hop)  # (time, freq)
        seg_r = stitch_windows(rec_wins[win_idx],  hop)
        orig_segs.append(seg_o.T)  # store as (freq × time)
        rec_segs .append(seg_r.T)
        seg_lengths.append(seg_o.shape[0])

    # ——— Concatenate all segments horizontally ———
    orig_full = np.hstack(orig_segs)  # (freq_bins × total_time)
    rec_full  = np.hstack(rec_segs)
    total_time = orig_full.shape[1]
    print(f"[DEBUG] stitched shapes → orig: {orig_full.shape}, rec: {rec_full.shape}")

    # ——— Compute xtick centers for word labels ———
    edges = np.cumsum([0] + seg_lengths)
    centers = (edges[:-1] + edges[1:]) / 2

    # ——— Plot original (top) & reconstruction (bottom) ———
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"hspace":0.1}
    )
    im = ax1.imshow(orig_full, origin="lower", aspect="auto")
    ax1.set_ylabel("Freq bin")
    ax1.set_title((f"{subject_id} - Origional Spectrogram"))
    ax1.set_xticks([])

    ax2.imshow(rec_full, origin="lower", aspect="auto")
    ax2.set_ylabel("Freq bin")
    ax2.set_xticks(centers)
    ax2.set_xticklabels([w.decode() for w in uniq_words], rotation=45, ha="right")
    ax2.set_xlabel("Word")
    ax2.set_title(f"{subject_id} - Reconstructed Spectrogram")
    # shared colorbar
    cbar = fig.colorbar(im, ax=(ax1, ax2), orientation="vertical", fraction=0.02)
    cbar.set_label("Spectral power")

    plt.tight_layout()
    plt.show()
    fig.savefig(save_dir / f"{subject_id}_Spectro.png", dpi=300, bbox_inches="tight")
if __name__ == "__main__":
    main()