import numpy as np
import os
import glob

# Path where features are stored
path_features = "features"

# Get all participant files
participant_files = glob.glob(os.path.join(path_features, "*_spec.npy"))

for spec_file in participant_files:
    # Extract participant ID from filename
    participant_id = os.path.basename(spec_file).split("_")[0]

    # Load spectrogram data
    spec = np.load(spec_file)

    # Compute binary speech vs. non-speech labels
    m = np.median(np.mean(spec, axis=1))
    binary_labels = (np.mean(spec, axis=1) > m).astype(int)  # Convert to 0/1

    # Save binary labels
    binary_file = os.path.join(path_features, f"{participant_id}_binary.npy")
    np.save(binary_file, binary_labels)

    print(f"Processed {participant_id}, saved binary labels to {binary_file}")

print("All participants processed successfully!")
