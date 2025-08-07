import torch
import torch.nn as nn

class AudioSpectrogramReadout(nn.Module):
    """
    LFADS readout module: takes latent factors [B, T, D] and returns
    a predicted spectrogram [B, T, F, 1], exactly as LFADS expects.
    Uses a single linear layer (no PCA, no CV).

    Args:
        in_features (int): dimensionality D of each latent factor vector.
        out_features (int): number F of spectral bins to predict.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # expose this so LFADS can verify fac_dim
        self.in_features = in_features
        # tells LFADS there's 1 distribution parameter per bin
        self.n_params = 1
        # single linear layer: D -> F
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        # factors: [B, T, D]
        # apply linear mapping to last dim -> [B, T, F]
        y = self.linear(factors)
        # add final param dimension -> [B, T, F, 1]
        return y #.unsqueeze(-1) 这里先注释掉一下




















# import torch
# import torch.nn as nn
# import h5py
# import numpy as np
# from lfads_torch.reconstruction_minimal_1 import predict_spectrogram_from_factors

# class AudioSpectrogramReadout(nn.Module):
#     """
#     LFADS readout module: takes latent factors [B,T,D] and returns
#     a predicted spectrogram [B,T,F,1], exactly as LFADS expects.
#     """
#     def __init__(self,
#                  in_features:      int,
#                  spectrogram_h5:   str,  # ground‑truth spectrogram 
#                  h5_key:           str,
#                  # n_folds:          int = 10,
#                  num_comps:        int = 50,
#                  ):
#         super().__init__()
#         # expose this so LFADS can check fac_dim
#         self.in_features = in_features
#         self.n_params = 1            # tells LFADS there’s 1 param per freq‑bin
#         # Store full dataset’s spectrogram here so the CV function can see it:
#         # self.spectrogram_np = spectrogram_np 注释掉 因为my_model里这行也注释掉了
#         # self.n_folds        = n_folds
#         self.num_comps      = num_comps

#         with h5py.File(spectrogram_h5, "r") as f:
#             # this grabs the dataset
#             data = f[h5_key][:]
#         # ensure it's a 2D numpy array [n_samples, n_bins] 这里不对吧我的数据是3D的
#         self.spectrogram_np = np.array(data)

#     def forward(self, factors: torch.Tensor):
        
#         B, T, D = factors.shape # 这里还是正常的（32,100,100）
#         fac_np   = factors.detach().cpu().numpy().reshape(-1, D)
#         # 这里就有bug了！这里fac_np 就是3200,100， 似乎因为（32*100，D）
#         # 2) Run exact PCA+LR 10‑fold routine
#         rec_flat, rs, explained_var = predict_spectrogram_from_factors(
#             fac_np,
#             self.spectrogram_np,
#             # n_folds=self.n_folds,
#             num_comps=self.num_comps
#         )

#         # 3) Reshape back to [B, T, F]
#         F    = self.spectrogram_np.shape[1]
#         rec  = rec_flat.reshape(B, T, F)

#         # 4) Add the final channel dim → [B, T, F, 1]
#         return torch.from_numpy(rec).to(factors.device).unsqueeze(-1)
