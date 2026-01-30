"""
Debug script to specifically check female disappearance events around 9 minutes.
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

# Parameters from config
COPULATION_DISAPPEARANCE_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['disappearance_min_duration_sec']
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']

# Load trajectory data
traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

# Extract male ID from metadata
male_id = 1
female_ids = [2, 3, 4, 5, 6]

# Calculate frame rate
frame_rate = len(trajectories) / trajectories['time'].iloc[-1]

# Male coordinates
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
print("FEMALE DISAPPEARANCE EVENTS AROUND 9-10 MINUTES")
print("="*80)

# Focus on 9-10.5 minute window
start_min = 9.0
end_min = 10.5
start_time_sec = start_min * 60
end_time_sec = end_min * 60

mask = (trajectories['time'] >= start_time_sec) & (trajectories['time'] <= end_time_sec)
frame_indices = np.where(mask)[0]

disappearance_frames_threshold = int(COPULATION_DISAPPEARANCE_MIN_DURATION * frame_rate)

for female_id in female_ids:
    print(f"\n{'='*80}")
    print(f"Female {female_id}")
    print(f"{'='*80}")

    female_x = trajectories[f'x{female_id}'].values
    female_y = trajectories[f'y{female_id}'].values
    female_missing = np.isnan(female_x) | np.isnan(female_y)

    # Find contiguous regions where female is missing (GLOBALLY)
    in_missing = False
    start_idx = 0
    female_disappearance_events = []

    for i in range(len(female_missing)):
        if female_missing[i] and not in_missing:
            in_missing = True
            start_idx = i
        elif not female_missing[i] and in_missing:
            duration_frames = i - start_idx
            if duration_frames >= disappearance_frames_threshold:
                female_disappearance_events.append((start_idx, i, duration_frames))
            in_missing = False

    # Handle case where missing extends to end
    if in_missing:
        duration_frames = len(female_missing) - start_idx
        if duration_frames >= disappearance_frames_threshold:
            female_disappearance_events.append((start_idx, len(female_missing), duration_frames))

    if len(female_disappearance_events) > 0:
        print(f"Found {len(female_disappearance_events)} disappearance events (>= {COPULATION_DISAPPEARANCE_MIN_DURATION}s) GLOBALLY:")

        for event_idx, (start, end, dur) in enumerate(female_disappearance_events):
            event_duration = dur / frame_rate
            event_time = trajectories['time'].iloc[start]
            end_time = trajectories['time'].iloc[end-1] if end < len(trajectories) else trajectories['time'].iloc[-1]

            # Check if this event overlaps with our window of interest
            event_start_min = event_time / 60
            event_end_min = end_time / 60

            overlaps_window = (event_end_min >= start_min and event_start_min <= end_min)

            if overlaps_window:
                print(f"\n  ** Event {event_idx + 1} (OVERLAPS WITH 9-10.5min WINDOW) **")
            else:
                print(f"\n  Event {event_idx + 1}:")

            print(f"    Time: {event_time:.1f}s - {end_time:.1f}s ({event_start_min:.2f} - {event_end_min:.2f} min)")
            print(f"    Duration: {event_duration:.1f}s ({event_duration/60:.2f} min)")
            print(f"    Frames: {start} - {end}")

            # Check if male is present during this time
            male_present_during = ~male_missing[start:end]
            male_present_pct = (male_present_during.sum() / len(male_present_during)) * 100

            print(f"    Male present during: {male_present_during.sum()}/{len(male_present_during)} frames ({male_present_pct:.1f}%)")

            if male_present_during.any():
                print(f"    -> Male IS present during disappearance")
                print(f"    -> Algorithm WILL assign copulation to Female {female_id}")

                # Check lookback distances
                lookback_frames = int(5 * frame_rate)
                lookback_start = max(0, start - lookback_frames)

                print(f"    Distances 5s before (frames {lookback_start}-{start}):")
                distances_before = distances_cm[female_id][lookback_start:start]
                valid_distances = distances_before[~np.isnan(distances_before)]
                if len(valid_distances) > 0:
                    print(f"      Mean: {valid_distances.mean():.2f} cm")
                    print(f"      Min: {valid_distances.min():.2f} cm")
                    print(f"      Max: {valid_distances.max():.2f} cm")
                else:
                    print(f"      No valid data before disappearance")
            else:
                print(f"    -> Male ALSO missing during disappearance")
                print(f"    -> Algorithm will SKIP this (already handled in male disappearance)")
    else:
        print(f"No disappearance events (>= {COPULATION_DISAPPEARANCE_MIN_DURATION}s)")

print("\n" + "="*80)
print("SUMMARY OF THE ISSUE")
print("="*80)
print("""
Based on the analysis:

1. Around 9:45-9:50 (9.75-9.83 min):
   - Male is MISSING for a long period (started disappearing around frame 35227)
   - Female 6 is also MISSING during much of this time
   - Female 6 is CLOSE to male when both are present (mean 0.47 cm, min 0.02 cm)

2. The algorithm's male disappearance detection (Event 1):
   - Male missing: 11.10 - 13.52 min (666.1s - 811.2s)
   - This is AFTER the 9:45 time you mentioned
   - The algorithm assigned this to Female 4 (closest 5s before: 3.85 cm)
   - But Female 6 had no valid data in the lookback period

3. The female disappearance detection should catch Female 6's disappearance
   starting around frame 35774 (596.2s = 9.94 min), which is close to your
   observed copulation time of 9:50 (590s).

The issue is likely:
- The algorithm looks 5 seconds BEFORE the disappearance starts
- If Female 6 is already missing or far from male in that lookback window,
  she won't be selected as the copulation partner
- The male disappearance event happens LATER (at 11.10 min), not at 9:45
""")
