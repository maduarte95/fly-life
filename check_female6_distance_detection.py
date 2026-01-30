"""
Check if Female 6 copulation should be detected by distance-based method.
"""

import pandas as pd
import numpy as np
import yaml
from scipy import ndimage

# Load config
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

COPULATION_DISTANCE_CM = CONFIG['behavior_thresholds']['copulation']['distance_cm']
COPULATION_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['min_duration_sec']
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']

# Load trajectory data
traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

male_id = 1
frame_rate = len(trajectories) / trajectories['time'].iloc[-1]

# Calculate Female 6 distance
male_x = trajectories[f'x{male_id}'].values
male_y = trajectories[f'y{male_id}'].values
female6_x = trajectories['x6'].values
female6_y = trajectories['y6'].values

dist_pixels = np.sqrt((male_x - female6_x)**2 + (male_y - female6_y)**2)
dist_cm = dist_pixels / PIXELS_PER_CM

print("="*80)
print("FEMALE 6 DISTANCE-BASED COPULATION DETECTION")
print("="*80)
print(f"Copulation distance threshold: {COPULATION_DISTANCE_CM} cm")
print(f"Copulation min duration: {COPULATION_MIN_DURATION} s")
print(f"Frames required: {int(COPULATION_MIN_DURATION * frame_rate)}")

# Apply distance-based detection
within_distance = (dist_cm < COPULATION_DISTANCE_CM).astype(int)
copulation_frames_threshold = int(COPULATION_MIN_DURATION * frame_rate)

# Use scipy.ndimage.label to find connected components
labeled_array, num_features = ndimage.label(within_distance)
copulation = np.zeros(len(within_distance), dtype=bool)

print(f"\nFound {num_features} regions where distance < {COPULATION_DISTANCE_CM} cm")

copulation_events = []
for region_label in range(1, num_features + 1):
    region_mask = labeled_array == region_label
    region_length = region_mask.sum()
    region_indices = np.where(region_mask)[0]
    start_frame = region_indices[0]
    end_frame = region_indices[-1]
    start_time = trajectories['time'].iloc[start_frame]
    end_time = trajectories['time'].iloc[end_frame]
    duration_sec = region_length / frame_rate

    if region_length >= copulation_frames_threshold:
        copulation[region_mask] = True
        copulation_events.append((start_frame, end_frame, duration_sec))
        print(f"\n  Region {region_label}: QUALIFIES as copulation")
    else:
        print(f"\n  Region {region_label}: too short")

    print(f"    Frames: {start_frame} - {end_frame} ({region_length} frames)")
    print(f"    Time: {start_time:.1f}s - {end_time:.1f}s ({start_time/60:.2f} - {end_time/60:.2f} min)")
    print(f"    Duration: {duration_sec:.1f}s ({duration_sec/60:.2f} min)")

print(f"\nTotal copulation events for Female 6: {len(copulation_events)}")
print(f"Total copulation time: {copulation.sum() / frame_rate / 60:.2f} minutes")

# Now check specifically around 9:30 - 10:00
print("\n" + "="*80)
print("CHECKING 9:30 - 10:00 MINUTE WINDOW")
print("="*80)

start_min = 9.5
end_min = 10.0
start_time_sec = start_min * 60
end_time_sec = end_min * 60

mask = (trajectories['time'] >= start_time_sec) & (trajectories['time'] <= end_time_sec)
window_distances = dist_cm[mask]
window_within = within_distance[mask]

valid_distances = window_distances[~np.isnan(window_distances)]

print(f"\nWindow: {start_min:.2f} - {end_min:.2f} minutes")
print(f"Valid distance measurements: {len(valid_distances)}/{mask.sum()}")

if len(valid_distances) > 0:
    print(f"Distance statistics:")
    print(f"  Mean: {valid_distances.mean():.2f} cm")
    print(f"  Min: {valid_distances.min():.2f} cm")
    print(f"  Max: {valid_distances.max():.2f} cm")
    print(f"  Median: {np.median(valid_distances):.2f} cm")

    within_threshold = valid_distances < COPULATION_DISTANCE_CM
    print(f"\nFrames within copulation distance: {within_threshold.sum()} ({within_threshold.sum()/len(valid_distances)*100:.1f}%)")

    # Find longest contiguous sequence within threshold
    window_indices = np.where(mask)[0]
    within_in_window = within_distance[mask]

    labeled, num = ndimage.label(within_in_window)
    if num > 0:
        print(f"\nContiguous sequences within {COPULATION_DISTANCE_CM} cm:")
        for i in range(1, num + 1):
            seq_mask = labeled == i
            seq_length = seq_mask.sum()
            seq_indices = np.where(seq_mask)[0]
            seq_start_frame = window_indices[seq_indices[0]]
            seq_end_frame = window_indices[seq_indices[-1]]
            seq_start_time = trajectories['time'].iloc[seq_start_frame]
            seq_duration = seq_length / frame_rate

            print(f"  Sequence {i}: {seq_length} frames ({seq_duration:.1f}s) at {seq_start_time/60:.2f} min")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Based on this analysis, we can determine:

1. If Female 6 shows very close distances (<0.3 cm) for extended periods
   around 9:45-9:50, the distance-based detection SHOULD catch it.

2. If the distances are close but the duration is less than 120 seconds,
   it won't be detected as copulation.

3. The fact that both male and female are missing during much of this
   period means there are no valid distance measurements, so distance-based
   detection CANNOT work during those times.

This is the root cause: the copulation happens but tracking is lost,
so neither distance-based NOR disappearance-based detection can properly
identify it (disappearance is too short, distance can't be measured when
IDs are lost).
""")
