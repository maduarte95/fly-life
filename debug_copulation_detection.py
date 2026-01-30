"""
Debug script to analyze copulation detection at specific time points.
Focus on the 9-10 minute period and 24 minute period.
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
COPULATION_DISTANCE_CM = CONFIG['behavior_thresholds']['copulation']['distance_cm']
COPULATION_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['min_duration_sec']
COPULATION_USE_ID_DISAPPEARANCE = CONFIG['behavior_thresholds']['copulation']['use_id_disappearance']
COPULATION_DISAPPEARANCE_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['disappearance_min_duration_sec']
PIXELS_PER_CM = CONFIG['pixels_per_cm']['dataset_1']['CamB']

# Load trajectory data
traj_path = 'data/1M/dataset_1/trajectories/DGRP375_CamB_2026-01-12T08_31_36_trajectories.csv'
trajectories = pd.read_csv(traj_path)

# Extract male ID from metadata
male_id = 1
female_ids = [2, 3, 4, 5, 6]

print("="*80)
print(f"COPULATION DETECTION DEBUG")
print("="*80)
print(f"Male ID: {male_id}")
print(f"Female IDs: {female_ids}")
print(f"Pixels per cm: {PIXELS_PER_CM}")
print(f"Copulation distance threshold: {COPULATION_DISTANCE_CM} cm")
print(f"Copulation min duration: {COPULATION_MIN_DURATION} s")
print("="*80)

# Calculate frame rate
frame_rate = len(trajectories) / trajectories['time'].iloc[-1]
print(f"\nFrame rate: {frame_rate:.2f} fps")

# Male coordinates
male_x = trajectories[f'x{male_id}'].values
male_y = trajectories[f'y{male_id}'].values

# Calculate distances
distances_cm = {}
for female_id in female_ids:
    female_x = trajectories[f'x{female_id}'].values
    female_y = trajectories[f'y{female_id}'].values
    dist_pixels = np.sqrt((male_x - female_x)**2 + (male_y - female_y)**2)
    dist_cm = dist_pixels / PIXELS_PER_CM
    distances_cm[female_id] = dist_cm

# Function to analyze a specific time window
def analyze_time_window(start_min, end_min, window_name):
    print(f"\n{'='*80}")
    print(f"{window_name}: {start_min:.2f} - {end_min:.2f} minutes")
    print(f"{'='*80}")

    start_time_sec = start_min * 60
    end_time_sec = end_min * 60

    # Find frames in this window
    mask = (trajectories['time'] >= start_time_sec) & (trajectories['time'] <= end_time_sec)
    window_frames = trajectories[mask]
    frame_indices = np.where(mask)[0]

    print(f"Frames in window: {len(window_frames)} (indices {frame_indices[0]} to {frame_indices[-1]})")

    # Check male presence
    male_present = ~np.isnan(male_x[mask])
    male_missing_pct = (1 - male_present.sum() / len(male_present)) * 100
    print(f"\nMale presence: {male_present.sum()}/{len(male_present)} frames ({100-male_missing_pct:.1f}%)")
    if male_missing_pct > 0:
        print(f"Male MISSING: {(~male_present).sum()} frames ({male_missing_pct:.1f}%)")

    # For each female, check:
    # 1. Distance to male
    # 2. Presence/absence
    # 3. Copulation criteria
    print(f"\nFemale analysis:")
    for female_id in female_ids:
        print(f"\n  Female {female_id}:")
        female_x = trajectories[f'x{female_id}'].values[mask]
        female_y = trajectories[f'y{female_id}'].values[mask]

        female_present = ~np.isnan(female_x)
        female_missing_pct = (1 - female_present.sum() / len(female_present)) * 100

        print(f"    Presence: {female_present.sum()}/{len(female_present)} frames ({100-female_missing_pct:.1f}%)")
        if female_missing_pct > 0:
            print(f"    MISSING: {(~female_present).sum()} frames ({female_missing_pct:.1f}%)")

        # Distance analysis (only when both present)
        distances = distances_cm[female_id][mask]
        both_present = male_present & female_present

        if both_present.sum() > 0:
            valid_distances = distances[both_present]
            print(f"    Distance (when both present):")
            print(f"      Mean: {valid_distances.mean():.2f} cm")
            print(f"      Min: {valid_distances.min():.2f} cm")
            print(f"      Max: {valid_distances.max():.2f} cm")

            # Check copulation distance threshold
            within_cop_dist = valid_distances < COPULATION_DISTANCE_CM
            if within_cop_dist.sum() > 0:
                pct_within = (within_cop_dist.sum() / len(valid_distances)) * 100
                print(f"      Within copulation distance (<{COPULATION_DISTANCE_CM}cm): {within_cop_dist.sum()} frames ({pct_within:.1f}%)")

        # Check for disappearance events (female missing)
        if female_missing_pct > 0:
            # Find contiguous missing regions
            female_missing = ~female_present
            in_missing = False
            start_idx = 0
            missing_events = []

            for i in range(len(female_missing)):
                if female_missing[i] and not in_missing:
                    in_missing = True
                    start_idx = i
                elif not female_missing[i] and in_missing:
                    duration_frames = i - start_idx
                    missing_events.append((start_idx, i, duration_frames))
                    in_missing = False

            if in_missing:
                duration_frames = len(female_missing) - start_idx
                missing_events.append((start_idx, len(female_missing), duration_frames))

            print(f"    Missing events in window: {len(missing_events)}")
            for idx, (start, end, dur) in enumerate(missing_events):
                dur_sec = dur / frame_rate
                actual_start_idx = frame_indices[start]
                actual_end_idx = frame_indices[end-1] if end > 0 else frame_indices[-1]
                actual_start_time = trajectories['time'].iloc[actual_start_idx]
                print(f"      Event {idx+1}: frames {start}-{end} (global: {actual_start_idx}-{actual_end_idx}), duration: {dur_sec:.1f}s, starts at {actual_start_time:.1f}s ({actual_start_time/60:.2f}min)")

                if dur_sec >= COPULATION_DISAPPEARANCE_MIN_DURATION:
                    print(f"        -> QUALIFIES for copulation (>= {COPULATION_DISAPPEARANCE_MIN_DURATION}s)")

                    # Check what happens before this event
                    lookback_frames = int(5 * frame_rate)
                    lookback_start = max(0, actual_start_idx - lookback_frames)

                    print(f"        -> Checking 5s before (frames {lookback_start}-{actual_start_idx}):")
                    distances_before = distances_cm[female_id][lookback_start:actual_start_idx]
                    valid_dist_before = distances_before[~np.isnan(distances_before)]
                    if len(valid_dist_before) > 0:
                        print(f"           Mean distance before: {valid_dist_before.mean():.2f} cm")
                        print(f"           This female would be assigned copulation by disappearance rule")

# Analyze the specific time windows mentioned
print("\n" + "="*80)
print("ANALYZING SPECIFIC TIME WINDOWS")
print("="*80)

# Around 9 minutes (you mentioned 9:45 specifically, so let's look 9-10 minutes)
analyze_time_window(9.0, 10.5, "WINDOW 1: Around 9 minutes (suspected copulation with Female 6)")

# Around 24 minutes
analyze_time_window(23.5, 25.0, "WINDOW 2: Around 24 minutes (detected copulation with Female 5)")

# Let's also check what the algorithm detected
print("\n" + "="*80)
print("CHECKING MALE DISAPPEARANCE EVENTS (ENTIRE RECORDING)")
print("="*80)

male_missing = np.isnan(male_x)
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

print(f"\nFound {len(male_disappearance_events)} male disappearance events (>= {COPULATION_DISAPPEARANCE_MIN_DURATION}s):")

for idx, (start, end, dur) in enumerate(male_disappearance_events):
    dur_sec = dur / frame_rate
    start_time = trajectories['time'].iloc[start]
    end_time = trajectories['time'].iloc[end-1] if end < len(trajectories) else trajectories['time'].iloc[-1]

    print(f"\n  Event {idx+1}:")
    print(f"    Time: {start_time:.1f}s - {end_time:.1f}s ({start_time/60:.2f} - {end_time/60:.2f} min)")
    print(f"    Duration: {dur_sec:.1f}s ({dur_sec/60:.2f} min)")
    print(f"    Frames: {start} - {end}")

    # Check which female was closest before
    lookback_frames = int(5 * frame_rate)
    lookback_start = max(0, start - lookback_frames)

    print(f"    Distances 5s before (frames {lookback_start}-{start}):")
    avg_distances = {}
    for female_id in female_ids:
        distances_before = distances_cm[female_id][lookback_start:start]
        valid_distances = distances_before[~np.isnan(distances_before)]
        if len(valid_distances) > 0:
            avg_distances[female_id] = valid_distances.mean()
            print(f"      Female {female_id}: {valid_distances.mean():.2f} cm (avg)")
        else:
            avg_distances[female_id] = np.inf
            print(f"      Female {female_id}: No valid data")

    if len(avg_distances) > 0:
        closest_female = min(avg_distances, key=avg_distances.get)
        print(f"    -> Closest female: {closest_female} ({avg_distances[closest_female]:.2f} cm)")
        print(f"    -> Algorithm would assign copulation to Female {closest_female}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
