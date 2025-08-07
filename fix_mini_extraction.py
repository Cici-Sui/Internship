import os
import h5py

# Path to your existing multi-subject HDF5 file.
input_file = "features/processed_dataset.h5"  # Update with your file's actual location

# Define the output directory (new directory "final_form" inside "features").
output_dir = os.path.join("features", "final_form")
os.makedirs(output_dir, exist_ok=True)

# Open the input file in read mode.
with h5py.File(input_file, 'r') as f_in:
    # Iterate through each subject group in the input file.
    for subject in f_in.keys():
        # Define the output file path for this subject.
        subject_file = os.path.join(output_dir, f"{subject}.h5")
        # Open a new HDF5 file for writing the subject's data.
        with h5py.File(subject_file, 'w') as f_out:
            # Copy the entire subject group from the input file into the new file.
            f_in.copy(subject, f_out)
        print(f"Subject {subject} saved as {subject_file}")

print("Extraction complete. Check the 'final_form' directory under 'features'.")
