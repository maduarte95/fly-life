"""
Analyze the false positive copulations for Females 2 and 4 with new thresholds.
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
COPULATION_DISAPPEARANCE_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['disappearance_min_duration_sec']
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']
PURSUIT_DISTANCE_CM = CONFIG['behavior_thresholds']['pursuit']['distance_cm']
PURSUIT_MIN_DURATION = CONFIG['behavior_thresholds']['pursuit']['min_duration_sec']

print(f"Current thresholds:")
print(f"  Copulation distance: {COPULATION_DISTANCE_CM} cm, min duration: {COPULATION_MIN_DURATION}s")
print(f"  Disappearance min duration: {COPULATION_DISAPPEARANCE_MIN_DURATION}s")
print(f"  Pursuit distance: {PURSUIT_DISTANCE_CM} cm, min duration: {PURSUIT_MIN_DURATION}s")

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

print("\n" + "="*80)
print("ANALYZING FEMALE 2 AND 4 DISAPPEARANCE EVENTS")
print("="*80)

disappearance_frames_threshold = int(COPULATION_DISAPPEARANCE_MIN_DURATION * frame_rate)

for female_id in [2, 4]:
    print(f"\n{'='*80}")
    print(f"FEMALE {female_id}")
    print(f"{'='*80}")

    female_x = trajectories[f'x{female_id}'].values
    female_y = trajectories[f'y{female_id}'].values
    female_missing = np.isnan(female_x) | np.isnan(female_y)

    # Find disappearance events
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

    if in_missing:
        duration_frames = len(female_missing) - start_idx
        if duration_frames >= disappearance_frames_threshold:
            female_disappearance_events.append((start_idx, len(female_missing), duration_frames))

    print(f"Found {len(female_disappearance_events)} disappearance events (>= {COPULATION_DISAPPEARANCE_MIN_DURATION}s)")

    for event_idx, (start, end, dur) in enumerate(female_disappearance_events):
        event_duration = dur / frame_rate
        event_time = trajectories['time'].iloc[start]
        end_time = trajectories['time'].iloc[end-1] if end < len(trajectories) else trajectories['time'].iloc[-1]

        print(f"\n  Event {event_idx + 1}:")
        print(f"    Time: {event_time:.1f}s - {end_time:.1f}s ({event_time/60:.2f} - {end_time/60:.2f} min)")
        print(f"    Duration: {event_duration:.1f}s ({event_duration/60:.2f} min)")
        print(f"    Frames: {start} - {end}")

        # Check if male is present during this time
        male_present_during = ~male_missing[start:end]
        male_present_pct = (male_present_during.sum() / len(male_present_during)) * 100

        print(f"    Male present during: {male_present_during.sum()}/{len(male_present_during)} frames ({male_present_pct:.1f}%)")

        if male_present_during.any():
            print(f"    -> Algorithm WILL assign copulation (male present)")

            # Check pursuit before disappearance
            lookback_frames = int(10 * frame_rate)  # Look back 10 seconds
            lookback_start = max(0, start - lookback_frames)

            # Was this female in pursuit before disappearing?
            distances_before = distances_cm[female_id][lookback_start:start]
            valid_distances = distances_before[~np.isnan(distances_before)]

            if len(valid_distances) > 0:
                avg_dist = valid_distances.mean()
                min_dist = valid_distances.min()

                # Check if within pursuit distance
                within_pursuit = valid_distances < PURSUIT_DISTANCE_CM
                pursuit_pct = (within_pursuit.sum() / len(valid_distances)) * 100

                print(f"    Distance 10s before disappearance:")
                print(f"      Mean: {avg_dist:.2f} cm")
                print(f"      Min: {min_dist:.2f} cm")
                print(f"      Within pursuit distance (<{PURSUIT_DISTANCE_CM}cm): {within_pursuit.sum()}/{len(valid_distances)} frames ({pursuit_pct:.1f}%)")

                # Detect if there was continuous pursuit right before
                within_pursuit_array = (distances_before < PURSUIT_DISTANCE_CM).astype(int)
                labeled_array, num_features = ndimage.label(within_pursuit_array)

                # Check if pursuit extends to the disappearance
                if start > 0 and within_pursuit_array[-1] == 1:
                    # Find this pursuit event
                    last_label = labeled_array[-1]
                    pursuit_mask = labeled_array == last_label
                    pursuit_duration = pursuit_mask.sum() / frame_rate

                    print(f"      ACTIVE PURSUIT leading into disappearance: {pursuit_duration:.1f}s")

                    if pursuit_duration >= PURSUIT_MIN_DURATION:
                        print(f"      -> This female WAS BEING COURTED before disappearing")
                        print(f"      -> LIKELY TRUE COPULATION")
                    else:
                        print(f"      -> Pursuit too brief (< {PURSUIT_MIN_DURATION}s)")
                        print(f"      -> LIKELY FALSE POSITIVE")
                else:
                    print(f"      -> NO active pursuit leading into disappearance")
                    print(f"      -> LIKELY FALSE POSITIVE (just tracking loss)")
            else:
                print(f"    Distance before disappearance: NO VALID DATA")
                print(f"    -> LIKELY FALSE POSITIVE")
        else:
            print(f"    -> Male also missing, algorithm will SKIP")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
To avoid false positives while catching true copulations:

SOLUTION: Only classify disappearance as copulation if there was ACTIVE PURSUIT
immediately before the disappearance.

Logic:
1. Female disappears for >= threshold duration (e.g., 60s)
2. Male is present during disappearance
3. Female was within pursuit distance (<1cm) for >= 2s IMMEDIATELY before disappearing
   -> Then classify as copulation
   -> Otherwise, it's just tracking loss (mark as unknown)

This makes biological sense:
- True copulation: male courts female -> female accepts -> they copulate -> IDs lost
- False positive: female just wanders into untrackable area without prior courtship

This would:
- Catch Female 6's copulation (she was being courted before disappearing)
- Reject False positives for Females 2 and 4 (no prior courtship)
""")
