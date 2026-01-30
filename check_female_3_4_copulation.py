"""
Check why Females 3 and 4 are being assigned copulation.
"""

import pandas as pd
import numpy as np
import yaml

CONFIG = yaml.safe_load(open('config.yaml'))
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']
COPULATION_DISAPPEARANCE_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['disappearance_min_duration_sec']

traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

male_id = 1
female_ids = [2, 3, 4, 5, 6]
frame_rate = len(trajectories) / trajectories['time'].iloc[-1]

male_x = trajectories[f'x{male_id}'].values
male_y = trajectories[f'y{male_id}'].values
male_missing = np.isnan(male_x)

# Calculate distances
distances_cm = {}
for female_id in female_ids:
    female_x = trajectories[f'x{female_id}'].values
    female_y = trajectories[f'y{female_id}'].values
    dist_pixels = np.sqrt((male_x - female_x)**2 + (male_y - female_y)**2)
    dist_cm = dist_pixels / PIXELS_PER_CM
    distances_cm[female_id] = dist_cm

print("="*80)
print("MALE DISAPPEARANCE EVENTS AND ASSIGNMENTS")
print("="*80)

# Find male disappearance events
disappearance_frames_threshold = int(COPULATION_DISAPPEARANCE_MIN_DURATION * frame_rate)
in_missing = False
start_idx = 0
male_disappearance_events = []

for i in range(len(male_missing)):
    if male_missing[i] and not in_missing:
        in_missing = True
        start_idx = i
    elif not male_missing[i] and in_missing:
        duration_frames = i - start_idx
        if duration_frames >= disappearance_frames_threshold:
            male_disappearance_events.append((start_idx, i, duration_frames))
        in_missing = False

if in_missing:
    duration_frames = len(male_missing) - start_idx
    if duration_frames >= disappearance_frames_threshold:
        male_disappearance_events.append((start_idx, len(male_missing), duration_frames))

print(f"\nFound {len(male_disappearance_events)} male disappearance events")

for event_idx, (start, end, dur) in enumerate(male_disappearance_events):
    dur_sec = dur / frame_rate
    start_time = trajectories['time'].iloc[start]
    end_time = trajectories['time'].iloc[end-1] if end < len(trajectories) else trajectories['time'].iloc[-1]

    print(f"\n{'='*80}")
    print(f"Male Disappearance Event {event_idx + 1}")
    print(f"{'='*80}")
    print(f"Time: {start_time:.1f}s - {end_time:.1f}s ({start_time/60:.2f} - {end_time/60:.2f} min)")
    print(f"Duration: {dur_sec:.1f}s")
    print(f"Frames: {start} - {end}")

    # Check which female was closest before
    lookback_frames = int(5 * frame_rate)
    lookback_start = max(0, start - lookback_frames)

    print(f"\nDistances 5s before (frames {lookback_start}-{start}):")
    avg_distances = {}
    for female_id in female_ids:
        distances_before = distances_cm[female_id][lookback_start:start]
        valid_distances = distances_before[~np.isnan(distances_before)]
        if len(valid_distances) > 0:
            avg_distances[female_id] = valid_distances.mean()
            print(f"  Female {female_id}: {valid_distances.mean():.2f} cm (avg)")
        else:
            avg_distances[female_id] = np.inf
            print(f"  Female {female_id}: No valid data")

    if len(avg_distances) > 0:
        closest_female = min(avg_distances, key=avg_distances.get)
        print(f"\n-> ASSIGNED TO: Female {closest_female} ({avg_distances[closest_female]:.2f} cm)")

        # Highlight if this is Female 3 or 4
        if closest_female in [3, 4]:
            print(f"*** THIS IS WHY FEMALE {closest_female} SHOWS COPULATION ***")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Females 3 and 4 are being assigned copulation via MALE DISAPPEARANCE events,
not female disappearance events. The male disappears for >=60s, and the algorithm
looks back 5 seconds to see which female was closest, then assigns the entire
male disappearance period to that female as copulation.

These assignments do NOT have the "prior pursuit" check because that check was
only added to the FEMALE disappearance detection, not the MALE disappearance
detection.
""")
