import h5py
from pathlib import Path

def inspect_shapes(h5_path):
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"No such file: {h5_path!r}")

    # Datasets to inspect
    keys_to_check = [
        "train_output_params",
        "valid_output_params",
        "train_recon_data",    # original train spectrogram windows
        "valid_recon_data",    # original valid spectrogram windows
    ]

    with h5py.File(h5_path, "r") as f:
        print(f"Inspecting HDF5: {h5_path.name}\n")
        for key in keys_to_check:
            if key in f:
                ds = f[key]
                print(f"{key!r}:")
                print(f"  shape = {tuple(ds.shape)}")
                print(f"  dtype = {ds.dtype}\n")
            else:
                print(f"Dataset {key!r} not found in file.\n")

if __name__ == "__main__":
    # Adjust this path as needed:
    h5_file = r"C:/UM_Files/Internship/runs/lfads-torch-example/my_data_sub-02/250526_exampleSingle/lfads_output_sub-02.h5"
    inspect_shapes(h5_file)








# import h5py

# filename = "features/final_data/sub-01.h5"  # Update with file path



# with h5py.File(filename, 'r') as f:
    
#     data = f["train_encod_data"]
#     first10 = data[:30, 0, 0]
#     print("First 10 elements in its first column:", first10)

# with h5py.File("path/to/your_file.h5", "r") as f:
#     # 2) List top‑level datasets so you can confirm the name
#     print("Available datasets:", list(f.keys()))

#     # 3) Grab the train_encod_data dataset
#     data = f["train_encod_data"]
#     print("Shape of train_encod_data:", data.shape)
#     #    → say it prints: (N, M, P)

#     # 4) Slice out the first 10 entries in the first column.
#     #    Here we assume “column” means index 0 along axis 1, and
#     #    you’re picking the very first slice along axis 2 as well:
#     first10 = data[:10, 0, 0]
#     print("First 10 elements in its first column:", first10)