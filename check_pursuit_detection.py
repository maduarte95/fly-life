"""
Check why Female 6 wasn't flagged for pursuit around 9:45-9:50.
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

PURSUIT_DISTANCE_CM = CONFIG['behavior_thresholds']['pursuit']['distance_cm']
PURSUIT_MIN_DURATION = CONFIG['behavior_thresholds']['pursuit']['min_duration_sec']
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
print("FEMALE 6 PURSUIT DETECTION ANALYSIS")
print("="*80)
print(f"Pursuit distance threshold: {PURSUIT_DISTANCE_CM} cm")
print(f"Pursuit min duration: {PURSUIT_MIN_DURATION} s")
print(f"Frames required: {int(PURSUIT_MIN_DURATION * frame_rate)}")

# Apply pursuit detection
within_distance = (dist_cm < PURSUIT_DISTANCE_CM).astype(int)
pursuit_frames_threshold = int(PURSUIT_MIN_DURATION * frame_rate)

# Use scipy.ndimage.label to find connected components
labeled_array, num_features = ndimage.label(within_distance)
pursuit = np.zeros(len(within_distance), dtype=bool)

print(f"\nFound {num_features} regions where distance < {PURSUIT_DISTANCE_CM} cm")

pursuit_events = []
for region_label in range(1, num_features + 1):
    region_mask = labeled_array == region_label
    region_length = region_mask.sum()
    region_indices = np.where(region_mask)[0]
    start_frame = region_indices[0]
    end_frame = region_indices[-1]
    start_time = trajectories['time'].iloc[start_frame]
    end_time = trajectories['time'].iloc[end_frame]
    duration_sec = region_length / frame_rate

    qualifies = region_length >= pursuit_frames_threshold

    if qualifies:
        pursuit[region_mask] = True
        pursuit_events.append((start_frame, end_frame, duration_sec))

print(f"\nTotal pursuit events that qualify (>= {PURSUIT_MIN_DURATION}s): {len(pursuit_events)}")
print(f"Total pursuit time: {pursuit.sum() / frame_rate / 60:.2f} minutes")

# Focus on 9:00 - 10:30 minute window
print("\n" + "="*80)
print("DETAILED ANALYSIS: 9:00 - 10:30 MINUTE WINDOW")
print("="*80)

start_min = 9.0
end_min = 10.5
start_time_sec = start_min * 60
end_time_sec = end_min * 60

mask = (trajectories['time'] >= start_time_sec) & (trajectories['time'] <= end_time_sec)
window_distances = dist_cm[mask]
window_within = within_distance[mask]
window_indices = np.where(mask)[0]

# Check presence
male_present = ~np.isnan(male_x[mask])
female_present = ~np.isnan(female6_x[mask])
both_present = male_present & female_present

print(f"\nWindow: {start_min:.2f} - {end_min:.2f} minutes")
print(f"Total frames: {mask.sum()}")
print(f"Both present: {both_present.sum()} frames ({both_present.sum()/mask.sum()*100:.1f}%)")
print(f"Male missing: {(~male_present).sum()} frames ({(~male_present).sum()/mask.sum()*100:.1f}%)")
print(f"Female missing: {(~female_present).sum()} frames ({(~female_present).sum()/mask.sum()*100:.1f}%)")

# Distance statistics when both present
valid_distances = window_distances[both_present]
if len(valid_distances) > 0:
    print(f"\nDistance statistics (when both present):")
    print(f"  Mean: {valid_distances.mean():.2f} cm")
    print(f"  Min: {valid_distances.min():.2f} cm")
    print(f"  Max: {valid_distances.max():.2f} cm")
    print(f"  Median: {np.median(valid_distances):.2f} cm")

    within_pursuit = valid_distances < PURSUIT_DISTANCE_CM
    print(f"\nFrames within pursuit distance (<{PURSUIT_DISTANCE_CM}cm) when both present:")
    print(f"  {within_pursuit.sum()}/{len(valid_distances)} frames ({within_pursuit.sum()/len(valid_distances)*100:.1f}%)")

# Find contiguous pursuit sequences IN THIS WINDOW
window_within_all = np.zeros(len(window_within), dtype=int)
window_within_all[both_present] = window_within[both_present]

labeled, num = ndimage.label(window_within_all)
print(f"\nContiguous sequences within {PURSUIT_DISTANCE_CM} cm in this window:")

if num > 0:
    for i in range(1, num + 1):
        seq_mask = labeled == i
        seq_length = seq_mask.sum()
        seq_indices = np.where(seq_mask)[0]
        seq_start_window_idx = seq_indices[0]
        seq_end_window_idx = seq_indices[-1]

        # Get global frame indices
        seq_start_frame = window_indices[seq_start_window_idx]
        seq_end_frame = window_indices[seq_end_window_idx]

        seq_start_time = trajectories['time'].iloc[seq_start_frame]
        seq_end_time = trajectories['time'].iloc[seq_end_frame]
        seq_duration = seq_length / frame_rate

        qualifies = seq_length >= pursuit_frames_threshold
        status = "QUALIFIES" if qualifies else "too short"

        print(f"\n  Sequence {i}: {status}")
        print(f"    Frames: {seq_start_frame} - {seq_end_frame} ({seq_length} frames)")
        print(f"    Time: {seq_start_time:.1f}s - {seq_end_time:.1f}s ({seq_start_time/60:.2f} - {seq_end_time/60:.2f} min)")
        print(f"    Duration: {seq_duration:.1f}s")
        print(f"    Required: {PURSUIT_MIN_DURATION:.1f}s")

        # Show distances in this sequence
        seq_distances = dist_cm[seq_start_frame:seq_end_frame+1]
        valid_seq_distances = seq_distances[~np.isnan(seq_distances)]
        if len(valid_seq_distances) > 0:
            print(f"    Distance range: {valid_seq_distances.min():.2f} - {valid_seq_distances.max():.2f} cm (mean: {valid_seq_distances.mean():.2f})")
else:
    print("  No sequences found")

# Now check for GAPS in the close distance periods
print("\n" + "="*80)
print("INVESTIGATING GAPS IN PROXIMITY")
print("="*80)

print(f"\nLooking at frames where distance would be < {PURSUIT_DISTANCE_CM} cm IF both were visible:")

# Sample every second in the window
sample_interval = int(frame_rate)  # 1 second
sample_indices = np.arange(0, len(window_distances), sample_interval)

print(f"\n{'Time(min)':>10} {'Frame':>8} {'Male':>5} {'Fem6':>5} {'Dist(cm)':>10} {'<1cm?':>6}")
print("-" * 60)

for idx in sample_indices:
    if idx >= len(window_distances):
        break

    global_frame = window_indices[idx]
    time_sec = trajectories['time'].iloc[global_frame]
    time_min = time_sec / 60

    male_vis = 'Y' if male_present[idx] else 'N'
    fem_vis = 'Y' if female_present[idx] else 'N'
    dist = window_distances[idx]

    if not np.isnan(dist):
        within = 'YES' if dist < PURSUIT_DISTANCE_CM else 'no'
        print(f"{time_min:>10.2f} {global_frame:>8} {male_vis:>5} {fem_vis:>5} {dist:>10.2f} {within:>6}")
    else:
        print(f"{time_min:>10.2f} {global_frame:>8} {male_vis:>5} {fem_vis:>5} {'---':>10} {'---':>6}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
The issue is likely one of these:

1. BOTH flies are missing during the copulation period, so distance cannot be measured
   - No valid distance = no pursuit detection possible

2. The periods when both ARE visible and close might be interrupted by tracking loss
   - Each contiguous period might be < {PURSUIT_MIN_DURATION}s even though cumulatively they're close

3. Female 6 might be slightly farther than {PURSUIT_DISTANCE_CM} cm during courtship
   - She could be at 1.1-1.5 cm, which wouldn't trigger pursuit detection

From the data above, you can see which scenario applies.
""")
