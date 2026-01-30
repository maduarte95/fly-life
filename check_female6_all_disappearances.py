"""
Check ALL Female 6 disappearances, even short ones.
"""

import pandas as pd
import numpy as np
import yaml

CONFIG = yaml.safe_load(open('config.yaml'))
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']

traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

frame_rate = len(trajectories) / trajectories['time'].iloc[-1]

female6_x = trajectories['x6'].values
female6_y = trajectories['y6'].values
female6_missing = np.isnan(female6_x) | np.isnan(female6_y)

print("ALL Female 6 disappearance events (any duration) in 9-11 minute window:")
print("="*80)

# Focus on 9-11 minute window
start_time = 9 * 60
end_time = 11 * 60
mask = (trajectories['time'] >= start_time) & (trajectories['time'] <= end_time)
window_indices = np.where(mask)[0]

# Find all disappearances in window
in_missing = False
start_idx = None

for i in window_indices:
    if female6_missing[i] and not in_missing:
        in_missing = True
        start_idx = i
    elif not female6_missing[i] and in_missing:
        duration_frames = i - start_idx
        duration_sec = duration_frames / frame_rate
        start_time_sec = trajectories['time'].iloc[start_idx]
        end_time_sec = trajectories['time'].iloc[i-1]

        print(f"\nDisappearance:")
        print(f"  Frames: {start_idx} - {i}")
        print(f"  Time: {start_time_sec:.1f}s - {end_time_sec:.1f}s ({start_time_sec/60:.2f} - {end_time_sec/60:.2f} min)")
        print(f"  Duration: {duration_sec:.1f}s")

        in_missing = False

if in_missing:
    duration_frames = window_indices[-1] - start_idx + 1
    duration_sec = duration_frames / frame_rate
    start_time_sec = trajectories['time'].iloc[start_idx]
    end_time_sec = trajectories['time'].iloc[window_indices[-1]]

    print(f"\nDisappearance (extends beyond window):")
    print(f"  Frames: {start_idx} - {window_indices[-1]+1}")
    print(f"  Time: {start_time_sec:.1f}s - {end_time_sec:.1f}s ({start_time_sec/60:.2f} - {end_time_sec/60:.2f} min)")
    print(f"  Duration: {duration_sec:.1f}s (to end of window)")
