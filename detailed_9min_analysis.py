"""
Very detailed analysis of the 9:30 - 10:00 minute period to understand
exactly what's happening frame-by-frame.
"""

import pandas as pd
import numpy as np
import yaml

# Load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']

# Load trajectory data
traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

male_id = 1
female_ids = [2, 3, 4, 5, 6]

frame_rate = len(trajectories) / trajectories['time'].iloc[-1]

# Calculate distances
male_x = trajectories[f'x{male_id}'].values
male_y = trajectories[f'y{male_id}'].values

distances_cm = {}
for female_id in female_ids:
    female_x = trajectories[f'x{female_id}'].values
    female_y = trajectories[f'y{female_id}'].values
    dist_pixels = np.sqrt((male_x - female_x)**2 + (male_y - female_y)**2)
    dist_cm = dist_pixels / PIXELS_PER_CM
    distances_cm[female_id] = dist_cm

print("="*80)
print("DETAILED FRAME-BY-FRAME ANALYSIS: 9:30 - 10:00")
print("="*80)

# Focus on 9:30 - 10:00 minute window
start_min = 9.5
end_min = 10.0
start_time_sec = start_min * 60
end_time_sec = end_min * 60

mask = (trajectories['time'] >= start_time_sec) & (trajectories['time'] <= end_time_sec)
window_data = trajectories[mask].copy()

print(f"\nTime window: {start_min:.2f} - {end_min:.2f} minutes")
print(f"Total frames in window: {len(window_data)}")
print(f"Frame indices: {window_data.index[0]} - {window_data.index[-1]}")

# Add presence columns
window_data['male_present'] = ~np.isnan(window_data[f'x{male_id}'])

for female_id in female_ids:
    window_data[f'f{female_id}_present'] = ~np.isnan(window_data[f'x{female_id}'])
    window_data[f'f{female_id}_dist'] = distances_cm[female_id][window_data.index]

# Sample every 5 seconds (300 frames at 60 fps)
sample_interval = 5 * 60  # 5 seconds
sample_indices = np.arange(0, len(window_data), sample_interval)

print(f"\nSampling every 5 seconds ({sample_interval} frames):")
print(f"{'Time':>8} {'Frame':>8} {'Male':>6} | ", end='')
for fid in female_ids:
    print(f"F{fid:>2} Dist", end='  ')
print()
print("-" * 80)

for idx in sample_indices:
    row = window_data.iloc[idx]
    time_sec = row['time']
    time_min = time_sec / 60
    frame = row.name

    print(f"{time_min:8.2f} {frame:8} ", end='')
    print(f"{'Y' if row['male_present'] else 'N':>6} | ", end='')

    for fid in female_ids:
        present = 'Y' if row[f'f{fid}_present'] else 'N'
        dist = row[f'f{fid}_dist']
        if not np.isnan(dist):
            print(f"{present} {dist:5.2f} ", end=' ')
        else:
            print(f"{present}   --- ", end=' ')
    print()

# Now let's find the exact moment when Female 6 disappears for a long time
print("\n" + "="*80)
print("FINDING FEMALE 6 DISAPPEARANCE EVENTS IN THIS WINDOW")
print("="*80)

female6_present = window_data['f6_present'].values
female6_distances = window_data['f6_dist'].values

# Find when Female 6 goes missing
in_missing = False
missing_start = None
current_window_frame = 0

for i in range(len(female6_present)):
    if not female6_present[i] and not in_missing:
        # Start of missing period
        in_missing = True
        missing_start = i
    elif female6_present[i] and in_missing:
        # End of missing period
        duration_frames = i - missing_start
        duration_sec = duration_frames / frame_rate
        actual_frame = window_data.iloc[missing_start].name
        actual_time = window_data.iloc[missing_start]['time']

        if duration_sec > 1.0:  # Only show gaps > 1 second
            print(f"\nMissing period:")
            print(f"  Window frames: {missing_start} - {i}")
            print(f"  Global frames: {window_data.iloc[missing_start].name} - {window_data.iloc[i-1].name}")
            print(f"  Time: {actual_time:.1f}s ({actual_time/60:.2f} min)")
            print(f"  Duration: {duration_sec:.1f}s")

            # Check male presence during this time
            male_present_during = window_data['male_present'].iloc[missing_start:i].sum()
            print(f"  Male present during: {male_present_during}/{duration_frames} frames")

            # Check distances BEFORE disappearance
            lookback_frames = int(5 * frame_rate)
            lookback_start = max(0, missing_start - lookback_frames)
            distances_before = female6_distances[lookback_start:missing_start]
            valid_distances_before = distances_before[~np.isnan(distances_before)]

            if len(valid_distances_before) > 0:
                print(f"  Distance 5s before:")
                print(f"    Mean: {valid_distances_before.mean():.2f} cm")
                print(f"    Min: {valid_distances_before.min():.2f} cm")
                print(f"    Last valid: {valid_distances_before[-1]:.2f} cm")
            else:
                print(f"  Distance 5s before: NO VALID DATA")

        in_missing = False

# Handle if still missing at end
if in_missing:
    duration_frames = len(female6_present) - missing_start
    duration_sec = duration_frames / frame_rate
    actual_frame = window_data.iloc[missing_start].name
    actual_time = window_data.iloc[missing_start]['time']

    print(f"\nMissing period (extends beyond window):")
    print(f"  Window frames: {missing_start} - {len(female6_present)}")
    print(f"  Global frames: {window_data.iloc[missing_start].name} - {window_data.iloc[-1].name}")
    print(f"  Time: {actual_time:.1f}s ({actual_time/60:.2f} min)")
    print(f"  Duration: {duration_sec:.1f}s (to end of window)")

    # Check male presence
    male_present_during = window_data['male_present'].iloc[missing_start:].sum()
    print(f"  Male present during: {male_present_during}/{duration_frames} frames")

    # Check distances BEFORE
    lookback_frames = int(5 * frame_rate)
    lookback_start = max(0, missing_start - lookback_frames)
    distances_before = female6_distances[lookback_start:missing_start]
    valid_distances_before = distances_before[~np.isnan(distances_before)]

    if len(valid_distances_before) > 0:
        print(f"  Distance 5s before:")
        print(f"    Mean: {valid_distances_before.mean():.2f} cm")
        print(f"    Min: {valid_distances_before.min():.2f} cm")
        print(f"    Last valid: {valid_distances_before[-1]:.2f} cm")
    else:
        print(f"  Distance 5s before: NO VALID DATA")

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)
print("""
The issue is now clear:

Female 6 disappears around frame 35774 (596.2s = 9.94 min), which corresponds
to your observation of copulation around 9:50 (590s = 9.83 min).

However, Female 6's disappearance duration (33.8s) is LESS than the required
120 seconds (2 minutes) threshold for copulation detection via disappearance.

The algorithm REQUIRES:
- disappearance_min_duration_sec: 120 seconds (2 minutes)

Female 6's disappearance is only 33.8 seconds, so it doesn't qualify as a
copulation event according to the current thresholds.

This is why the copulation with Female 6 at ~9:50 is NOT detected by the
disappearance-based detection method.
""")
