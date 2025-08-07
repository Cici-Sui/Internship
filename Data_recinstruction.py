import os
import numpy as np
import h5py

# Directory where the .npy files are stored
data_dir = r'./features'

# Parameters in seconds
window_length_sec = 1.0    # 1 second window
frame_shift_sec = 0.1      # 0.1 second frame shift
sampling_rate = 100        # 100 Hz (each data point is 0.01 seconds)

# function to extract neuro data by overlaping
def extract_windows_2d(data, window_length_sec, frame_shift_sec, sampling_rate):
    """
    Extract overlapping windows from a 2D data array and return a 3D array.

    Parameters:
        data (np.array): 2D input data array of shape (t, channels).
        window_length_sec (float): Length of each window in seconds.
        frame_shift_sec (float): Time shift between successive windows in seconds.
        sampling_rate (int): in Hz 

    Returns:
        np.array: A 3D array with shape (num_windows, window_length, channels),
                  where each "window" is an extracted segment from the data.
    """
    # Convert window length and frame shift from seconds to data points
    window_length = int(window_length_sec * sampling_rate)
    frame_shift = int(frame_shift_sec * sampling_rate)
    
    num_samples = data.shape[0]
    windows = []
    
    # Slide the window across the data
    for start in range(0, num_samples - window_length + 1, frame_shift):
        # Extract a 2D window (t x channels)
        window = data[start: start + window_length, :]
        windows.append(window)
    
    # Convert list of 2D windows to a 3D numpy array
    return np.array(windows)


# Prepare the 3D matrix dataset
# Loop over participants sub-01 to sub-10
# output_file = 'features/processed_dataset.h5'
output_dir = os.path.join("features", "final_data")
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 11):
        subject = f"sub-{i:02d}"
    
        # Define the file paths for features, words, and binary information
        feat_file = os.path.join(data_dir, f"{subject}_feat.npy")
        # Define the file path for the audio spectrogram
        spec_file = os.path.join(data_dir, f"{subject}_spec.npy")

        # Load the arrays
        data = np.load(feat_file)  # Expected shape: (t, 127)
        spec_data = np.load(spec_file)

        # Extract overlapping windows
        windows_3d = extract_windows_2d(data, window_length_sec, frame_shift_sec, sampling_rate) #这个还存吗
        spec_3d = extract_windows_2d(spec_data, window_length_sec, frame_shift_sec, sampling_rate)

        n_windows = windows_3d.shape[0]
        split_index = int(n_windows * 0.8)

        # Split the data into training and validation sets
        train_encod_data = windows_3d[:split_index, :, :]
        valid_encod_data = windows_3d[split_index:, :, :]
        train_recon_data = spec_3d[:split_index, :, :]
        valid_recon_data = spec_3d[split_index:, :, :]


        # Create a group for the subject in the HDF5 file 之前的办法
        # grp = f.create_group(subject)
        # grp.create_dataset('train_encod_data', data=train_encod_data)
        # grp.create_dataset('valid_encod_data', data=valid_encod_data)
        # grp.create_dataset('train_recon_data', data=train_recon_data)
        # grp.create_dataset('valid_recon_data', data=valid_recon_data)

        # 新办法，全都分开存档
        output_file = os.path.join(output_dir, f"{subject}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('train_encod_data', data=train_encod_data)
            f.create_dataset('valid_encod_data', data=valid_encod_data)
            f.create_dataset('train_recon_data', data=train_recon_data)
            f.create_dataset('valid_recon_data', data=valid_recon_data)

        print(f"{subject}: Saved file {output_file}")
        print(f"{subject}: train_encod_data shape = {train_encod_data.shape}, "
              f"valid_encod_data shape = {valid_encod_data.shape}\n"
              f"train_recon_data shape = {train_recon_data.shape}, "
              f"valid_recon_data shape = {valid_recon_data.shape}")

print("All data saved to", output_dir)

# Output the shape of the extracted windows
# Expected shape: (number_of_windows, window_length_in_points, number_of_features)
# print("Shape of the extracted windows:", windows_3d.shape)
   


    # 保存数据
    # np.save(os.path.join(data_dir,f'{subject}_windows3d.npy'), windows_3d) 



# get the training set and validation set
 



  