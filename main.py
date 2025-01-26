import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points_file, ecog_data_file):
    trial_points = pd.read_csv(trial_points_file, header=None)
    if trial_points.shape[1] != 3:
        raise ValueError("trial_points file must have exactly three columns: start, peak, and finger.")
    trial_points = trial_points.astype(int)
    trial_points.columns = ['start', 'peak', 'finger']

    # Load ECoG data
    ecog_data = pd.read_csv(ecog_data_file, header=None)
    if ecog_data.shape[1] != 1:
        raise ValueError("ecog_data file must have exactly one column of data.")
    ecog_data = ecog_data.iloc[:, 0].values  

    # Initialize output matrix
    num_fingers = 5
    block_length = 1201  
    fingers_erp_mean = np.zeros((num_fingers, block_length))

    # Process each finger
    for finger in range(1, num_fingers + 1):
        finger_trials = trial_points[trial_points['finger'] == finger]
        
        # Extract segments and average them
        all_segments = []
        for _, row in finger_trials.iterrows():
            start_idx = row['start'] - 200 
            end_idx = row['start'] + 1000 
            if start_idx >= 0 and end_idx < len(ecog_data):
                segment = ecog_data[start_idx:end_idx + 1] 
                if len(segment) == block_length:  
                    all_segments.append(segment)

        if all_segments: 
            fingers_erp_mean[finger - 1, :] = np.mean(all_segments, axis=0)
        else:
            print(f"Warning: No valid trials found for finger {finger}.")

    # Plot the averaged brain response for each finger
    time_axis = np.linspace(-200, 1000, block_length)
    for finger in range(1, num_fingers + 1):
        plt.plot(time_axis, fingers_erp_mean[finger - 1, :], label=f'Finger {finger}')

    plt.xlabel('Time (ms)')
    plt.ylabel('Brain Signal (µV)')
    plt.title('Average Brain Response per Finger')
    plt.legend()
    plt.grid(True)
    plt.show()

    return fingers_erp_mean

fingers_erp_mean = calc_mean_erp("events_file_ordered.csv", "brain_data_channel_one.csv")
print("hi")