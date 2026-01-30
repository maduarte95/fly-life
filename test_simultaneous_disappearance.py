"""
Test if simultaneous disappearance detection would catch the Female 6 copulation.
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
COPULATION_DISAPPEARANCE_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['disappearance_min_duration_sec']

# Load trajectory data
traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

male_id = 1
female_ids = [2, 3, 4, 5, 6]
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
print("TESTING SIMULTANEOUS DISAPPEARANCE DETECTION")
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

print(f"\nFound {len(male_disappearance_events)} male disappearance events (>= {COPULATION_DISAPPEARANCE_MIN_DURATION}s)")

# For each male disappearance, check for simultaneous female disappearances
SIMULTANEOUS_WINDOW_SEC = 10  # ±10 seconds

for event_idx, (start, end, dur) in enumerate(male_disappearance_events):
    dur_sec = dur / frame_rate
    start_time = trajectories['time'].iloc[start]
    end_time = trajectories['time'].iloc[end-1] if end < len(trajectories) else trajectories['time'].iloc[-1]

    print(f"\n{'='*80}")
    print(f"Male Disappearance Event {event_idx + 1}")
    print(f"{'='*80}")
    print(f"Time: {start_time:.1f}s - {end_time:.1f}s ({start_time/60:.2f} - {end_time/60:.2f} min)")
    print(f"Duration: {dur_sec:.1f}s ({dur_sec/60:.2f} min)")
    print(f"Frames: {start} - {end}")

    # Check which females also disappear within ±10 seconds of the male's disappearance START
    simultaneous_window_frames = int(SIMULTANEOUS_WINDOW_SEC * frame_rate)

    print(f"\nChecking for females who disappear within ±{SIMULTANEOUS_WINDOW_SEC}s of male disappearance start:")
    print(f"  Window: frames {max(0, start - simultaneous_window_frames)} - {min(len(male_missing), start + simultaneous_window_frames)}")

    co_disappearing_females = []

    for female_id in female_ids:
        female_x = trajectories[f'x{female_id}'].values
        female_y = trajectories[f'y{female_id}'].values
        female_missing = np.isnan(female_x) | np.isnan(female_y)

        # Check if female is missing during a significant portion of the male's disappearance
        female_missing_during = female_missing[start:end].sum()
        overlap_pct = (female_missing_during / dur) * 100

        # Also check if female's disappearance STARTS near the male's disappearance start
        # Find when this female goes missing around the male's disappearance time
        window_start = max(0, start - simultaneous_window_frames)
        window_end = min(len(female_missing), start + simultaneous_window_frames)

        # Find the first missing frame in this window
        female_disappearance_start = None
        for i in range(window_start, window_end):
            if female_missing[i]:
                # Check if this is the start of a disappearance
                if i == 0 or not female_missing[i-1]:
                    female_disappearance_start = i
                    break
                # Or if female was already missing, find when it started
                if i == window_start:
                    # Walk backwards to find the start
                    for j in range(i, -1, -1):
                        if not female_missing[j]:
                            female_disappearance_start = j + 1
                            break
                        if j == 0:
                            female_disappearance_start = 0
                    break

        if female_disappearance_start is not None:
            time_diff = abs(female_disappearance_start - start) / frame_rate
            female_start_time = trajectories['time'].iloc[female_disappearance_start]

            print(f"\n  Female {female_id}:")
            print(f"    Disappearance starts: frame {female_disappearance_start}, time {female_start_time:.1f}s ({female_start_time/60:.2f} min)")
            print(f"    Time difference from male: {time_diff:.1f}s")
            print(f"    Overlap with male disappearance: {female_missing_during}/{dur} frames ({overlap_pct:.1f}%)")

            if time_diff <= SIMULTANEOUS_WINDOW_SEC:
                print(f"    -> CO-DISAPPEARS with male (within {SIMULTANEOUS_WINDOW_SEC}s window)")

                # Get last valid distance before EITHER disappeared
                last_valid_frame = min(start, female_disappearance_start) - 1
                lookback_start = max(0, last_valid_frame - int(5 * frame_rate))

                distances_before = distances_cm[female_id][lookback_start:last_valid_frame+1]
                valid_distances = distances_before[~np.isnan(distances_before)]

                if len(valid_distances) > 0:
                    avg_dist = valid_distances.mean()
                    min_dist = valid_distances.min()
                    last_valid_dist = valid_distances[-1] if len(valid_distances) > 0 else np.nan

                    print(f"    Distance before disappearance (5s lookback from frame {last_valid_frame}):")
                    print(f"      Mean: {avg_dist:.2f} cm")
                    print(f"      Min: {min_dist:.2f} cm")
                    print(f"      Last valid: {last_valid_dist:.2f} cm")

                    co_disappearing_females.append({
                        'female_id': female_id,
                        'time_diff': time_diff,
                        'overlap_pct': overlap_pct,
                        'avg_distance': avg_dist,
                        'min_distance': min_dist,
                        'last_distance': last_valid_dist
                    })
                else:
                    print(f"    Distance before disappearance: NO VALID DATA")
                    co_disappearing_females.append({
                        'female_id': female_id,
                        'time_diff': time_diff,
                        'overlap_pct': overlap_pct,
                        'avg_distance': np.inf,
                        'min_distance': np.inf,
                        'last_distance': np.inf
                    })
            else:
                print(f"    -> Does not co-disappear (time diff > {SIMULTANEOUS_WINDOW_SEC}s)")
        else:
            print(f"\n  Female {female_id}: Not missing in simultaneous window")

    # Now make decision
    print(f"\n{'='*80}")
    print(f"DECISION for Male Disappearance Event {event_idx + 1}:")
    print(f"{'='*80}")

    if len(co_disappearing_females) > 0:
        print(f"Found {len(co_disappearing_females)} co-disappearing female(s): {[f['female_id'] for f in co_disappearing_females]}")
        print(f"\nSelecting based on closest distance before disappearance:")

        # Choose the closest one
        best_female = min(co_disappearing_females, key=lambda x: x['avg_distance'])
        print(f"\n  SELECTED: Female {best_female['female_id']}")
        print(f"    Average distance: {best_female['avg_distance']:.2f} cm")
        print(f"    Time difference: {best_female['time_diff']:.1f}s")
        print(f"    Overlap: {best_female['overlap_pct']:.1f}%")
    else:
        print("No co-disappearing females found. Falling back to original method (closest in 5s before male disappearance).")

        # Original method
        lookback_frames = int(5 * frame_rate)
        lookback_start = max(0, start - lookback_frames)

        avg_distances = {}
        for female_id in female_ids:
            distances_before = distances_cm[female_id][lookback_start:start]
            valid_distances = distances_before[~np.isnan(distances_before)]
            if len(valid_distances) > 0:
                avg_distances[female_id] = valid_distances.mean()
            else:
                avg_distances[female_id] = np.inf

        closest_female = min(avg_distances, key=avg_distances.get)
        print(f"  SELECTED (original method): Female {closest_female} ({avg_distances[closest_female]:.2f} cm)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If simultaneous disappearance detection is implemented:
- It will check for females who disappear within ±10 seconds of the male
- For the first male disappearance event (11.10 min), we'll see if Female 6
  is identified as co-disappearing even though her actual copulation was earlier
""")
