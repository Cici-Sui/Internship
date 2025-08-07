import os
import numpy as np
from scipy.stats import pearsonr
# from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# (Waveform synthesis still here if need it; otherwise remove createAudio)
# import lfads_torch.reconstructWave as rW   
import lfads_torch.MelFilterBank as mel


def predict_spectrogram_from_factors(
    factors_np:     np.ndarray,  # [B/n_windows, T, D]
    spectrogram_np: np.ndarray,  # [B/n_windows, T, F]
    # n_folds:        int = 10,
    num_comps:      int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the spectrogram from the latent factors using PCA + Linear Regression.

    Args:
        factors_np: Latent factors of shape [B, T, D]
        spectrogram_np: Ground truth spectrogram of shape [B, T, F]
        num_comps: Number of PCA components to use

    Returns:
        rec_spec: Predicted spectrogram of shape [B, T, F]
        rs: Pearson r for each frequency bin, shape [F]
    """

    B, T, D = factors_np.shape
    _, _, F = spectrogram_np.shape

    # Initialize an empty array for the predicted spectrogram
    rec_spec = np.zeros_like(spectrogram_np)
    rs = np.zeros(F)  # Store Pearson correlation for each frequency bin

    # Apply PCA and LR for each window (B)
    for i in range(B):
        # Extract the latent factors and the true spectrogram for this window
        X = factors_np[i]  # Shape: [T, D]
        Y = spectrogram_np[i]  # Shape: [T, F]

        # Perform PCA on the factors for this window
        pca = PCA(n_components=num_comps)
        Z = pca.fit_transform(X)  # Shape: [T, num_comps]

        # Fit Linear Regression to predict the spectrogram
        lr = LinearRegression(n_jobs=-1)
        lr.fit(Z, Y)  # Fit LR to predict the spectrogram from PCA-transformed factors

        # Make predictions for the current window
        Y_pred_flat = lr.predict(Z)  # Shape: [T, F]
        rec_spec[i] = Y_pred_flat  # Store the prediction for this window

        # Compute Pearson correlation for each frequency bin
        for b in range(F):
            r, _ = pearsonr(Y[:, b], Y_pred_flat[:, b])  # Pearson r for frequency bin b
            rs[b] = r

    return rec_spec, rs

# # —————— 分割线 ——————
#     # Perform PCA on the factors
#     pca = PCA(n_components=num_comps)
#     Z = pca.fit_transform(X)  # Shape: [B*T, num_comps]

#     # Fit Linear Regression to predict the spectrogram
#     lr = LinearRegression(n_jobs=-1)
#     lr.fit(Z, Y)  # Predicting the spectrogram from PCA-transformed factors

#     # Make predictions on the entire data (train + test)
#     Y_pred_flat = lr.predict(Z)  # Shape: [B*T, F]

#     # Reshape the predicted spectrogram back to [B, T, F]
#     rec_spec = Y_pred_flat.reshape(B, T, F)

#     # Compute Pearson correlation for each frequency bin
#     rs = np.zeros(F)
#     for b in range(F):
#         r, _ = pearsonr(Y[:, b], Y_pred_flat[:, b])  # Compute Pearson r for each frequency bin
#         rs[b] = r

#     return rec_spec, rs
# 分割线
#     # Preallocate
#     rec_spec      = np.zeros((n_w, T, F), dtype=spectrogram_np.dtype)
#     rs            = np.zeros((n_folds, F), dtype=float)
#     explained_var = np.zeros(n_folds, dtype=float)

#     # split on window‐index
#     kf = KFold(n_splits=n_folds, shuffle=False)
#     for fold, (train_w, test_w) in enumerate(kf.split(np.arange(n_w))):
#         # flatten train windows → [#train*T, D] & [#train*T, F]
#         X_train = factors_np[train_w].reshape(-1, D)
#         Y_train = spectrogram_np[train_w].reshape(-1, F)

#         # flatten test windows → [#test*T, D] & [#test*T, F]
#         X_test  = factors_np[test_w ].reshape(-1, D)
#         Y_test  = spectrogram_np[test_w ].reshape(-1, F)

#         # PCA on training data
#         pca = PCA(n_components=num_comps)
#         Z_train = pca.fit_transform(X_train)
#         Z_test  = pca.transform(X_test)
#         explained_var[fold] = pca.explained_variance_ratio_[:num_comps].sum()

#         # Linear regression
#         lr = LinearRegression(n_jobs=-1)
#         lr.fit(Z_train, Y_train)

#         # Predict on test, then reshape back into windows
#         Y_pred_flat = lr.predict(Z_test)  # shape [#test*T, F]
#         rec_spec[test_w] = Y_pred_flat.reshape(len(test_w), T, F)

#         # Pearson r per frequency bin (across all flattened test points)
#         for b in range(F):
#             rs[fold, b], _ = pearsonr(Y_test[:, b], Y_pred_flat[:, b])

#     return rec_spec, rs, explained_var


# def createAudio(
#     spectrogram: np.ndarray,
#     audiosr:     int     = 16000,
#     winLength:   float   = 0.05,
#     frameshift:  float   = 0.01
# ):
#     """
#     (Optional) unchanged: log‐mel spectrogram → waveform via Griffin–Lim.
#     """
#     mfb = mel.MelFilterBank(int((audiosr * winLength) / 2 + 1),
#                             spectrogram.shape[1],
#                             audiosr)
#     linear_spec = mfb.fromLogMels(spectrogram)
#     nfolds, hop = 10, int(linear_spec.shape[0] / 10)
#     rec_audio = np.array([])
#     for w in range(0, linear_spec.shape[0], hop):
#         seg = linear_spec[w : min(w+hop, linear_spec.shape[0]), :]
#         # rec = rW.reconstructWavFromSpectrogram(...)  # re‑enable if you import rW
#         # rec_audio = np.append(rec_audio, rec)
#     # scaled = np.int16(rec_audio/np.max(np.abs(rec_audio))*32767)
#     # return scaled

#     # if you're not using this right now, you can comment out the body
#     raise NotImplementedError("createAudio is not currently used in LFADS readout.")


# # you can leave this stub if you want to test standalone
# if __name__ == "__main__":
#     # Example:
#     # factors = np.load("all_factors.npy")    # shape [2394,100,D]
#     # specs   = np.load("all_specs.npy")      # shape [2394,100,23]
#     # rec_spec, rs, ev = predict_spectrogram_from_factors(factors, specs)
#     pass
