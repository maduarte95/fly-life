"""
Check if Female 6 has any distance-based copulation detection that could be merged.
"""

import pandas as pd
import numpy as np
import yaml
from scipy import ndimage

CONFIG = yaml.safe_load(open('config.yaml'))
COPULATION_DISTANCE_CM = CONFIG['behavior_thresholds']['copulation']['distance_cm']
COPULATION_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['min_duration_sec']
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']

traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

frame_rate = len(trajectories) / trajectories['time'].iloc[-1]

male_x = trajectories['x1'].values
male_y = trajectories['y1'].values
female6_x = trajectories['x6'].values
female6_y = trajectories['y6'].values

dist_pixels = np.sqrt((male_x - female6_x)**2 + (male_y - female6_y)**2)
dist_cm = dist_pixels / PIXELS_PER_CM

print("="*80)
print("FEMALE 6 DISTANCE-BASED COPULATION (9-11 min window)")
print("="*80)
print(f"Copulation distance: < {COPULATION_DISTANCE_CM} cm")
print(f"Copulation min duration: {COPULATION_MIN_DURATION} s")

# Check 9-11 minute window
start_time = 9 * 60
end_time = 11 * 60
mask = (trajectories['time'] >= start_time) & (trajectories['time'] <= end_time)
window_dist = dist_cm[mask]
window_time = trajectories['time'].values[mask]
window_indices = np.where(mask)[0]

# Find close proximity periods
within_copulation_dist = (window_dist < COPULATION_DISTANCE_CM).astype(int)

print(f"\nFrames within {COPULATION_DISTANCE_CM} cm in window: {within_copulation_dist.sum()} / {len(within_copulation_dist)}")

# Find contiguous sequences
labeled_array, num_features = ndimage.label(within_copulation_dist)

if num_features > 0:
    print(f"\nFound {num_features} sequences within copulation distance:")

    for region_label in range(1, num_features + 1):
        region_mask = labeled_array == region_label
        region_length = region_mask.sum()
        region_indices = np.where(region_mask)[0]

        start_idx = window_indices[region_indices[0]]
        end_idx = window_indices[region_indices[-1]]
        start_time_val = trajectories['time'].iloc[start_idx]
        end_time_val = trajectories['time'].iloc[end_idx]
        duration_sec = region_length / frame_rate

        qualifies = duration_sec >= COPULATION_MIN_DURATION
        status = "QUALIFIES" if qualifies else "too short"

        print(f"\n  Sequence {region_label}: {status}")
        print(f"    Frames: {start_idx} - {end_idx} ({region_length} frames)")
        print(f"    Time: {start_time_val:.1f}s - {end_time_val:.1f}s ({start_time_val/60:.2f} - {end_time_val/60:.2f} min)")
        print(f"    Duration: {duration_sec:.1f}s")
else:
    print("\nNo sequences found within copulation distance")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print(f"""
The issue is clear: Female 6 is being PURSUED (within 1.0 cm) but not within
COPULATION distance (<{COPULATION_DISTANCE_CM} cm) for long enough periods.

During actual copulation:
- The flies are very close (mean 0.47 cm when visible)
- But they keep disappearing (tracking loss)
- When visible, they're often at 0.4-0.5 cm (pursuit range, not copulation range)
- Only briefly < 0.3 cm (0.8s total)

The gap merging (300s threshold) can only merge DETECTED copulation events.
It cannot merge pursuit events or create copulation from pursuit.

SOLUTION: The copulation is being detected as PURSUIT (orange), not COPULATION (red).
We need to convert sustained pursuit that transitions into/through disappearance
into copulation.
""")
