import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PARAMETERS - Easy to modify
# ============================================================================

# File paths
TRAJECTORIES_PATH_1 = 'data/trajectories/0_trajectories/Copy of DGRP375_CamAVideo1_2026-01-12T08_31_36_trajectories.csv'
TRAJECTORIES_PATH_2 = 'data/trajectories/0_trajectories/Copy of DGRP375_CamAVideo2_2026-01-12T08_31_36_trajectories.csv'
AREAS_PATH = 'data/trajectories/0_trajectories/Copy of DGRP375_CamAVideo1_2026-01-12T08_31_36_areas.csv'

# Experimental parameters
RECORDING_TIME = 7200  # seconds (2 hours)
PIXELS_PER_CM_CAMA = 208  # CamA conversion: 1cm = 208 pixels
PIXELS_PER_CM_CAMB = 203  # CamB conversion: 1cm = 203 pixels
PIXELS_PER_CM = PIXELS_PER_CM_CAMA  # Currently using CamA

# Behavior detection thresholds
COPULATION_DISTANCE_CM = 0.3  # Distance threshold for copulation (cm)
COPULATION_MIN_DURATION = 120  # Minimum duration for copulation (seconds)
COPULATION_USE_ID_DISAPPEARANCE = True  # Also detect copulation when male disappears
COPULATION_DISAPPEARANCE_MIN_DURATION = 120  # Min duration when male is missing (seconds)

PURSUIT_DISTANCE_CM = 1  # Distance threshold for pursuit (cm)
PURSUIT_MIN_DURATION = 2.0  # Minimum duration for pursuit (seconds)

# Plotting parameters
PLOT_TIME_UNIT = 'minutes'  # 'seconds' or 'minutes'
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 8

# ============================================================================
# LOAD DATA
# ============================================================================
#%%
print("Loading data...")

# Load areas to identify male
areas = pd.read_csv(AREAS_PATH)
print(f"\nAreas DataFrame shape: {areas.shape}")
print(areas)

# Identify male (smallest mean area, 0-indexed in areas file)
male_index_areas = areas['mean'].idxmin()
male_id_trajectories = male_index_areas + 1  # Convert to 1-indexed for trajectories
print(f"\nMale fly: index {male_index_areas} in areas file")
print(f"Male fly: ID {male_id_trajectories} in trajectories file")

# Load and concatenate trajectories
trajectories_1 = pd.read_csv(TRAJECTORIES_PATH_1)
trajectories_2 = pd.read_csv(TRAJECTORIES_PATH_2)

# Offset Video 2 timestamps to continue from Video 1
video1_duration = trajectories_1['time'].iloc[-1]
trajectories_2['time'] = trajectories_2['time'] + video1_duration

trajectories = pd.concat([trajectories_1, trajectories_2], ignore_index=True)

print(f"\nTrajectories DataFrame shape: {trajectories.shape}")
print(f"Recording length: {trajectories['time'].iloc[-1]:.2f} seconds")

# Calculate frame rate
frame_rate = len(trajectories) / RECORDING_TIME
print(f"Calculated frame rate: {frame_rate:.2f} fps")

# ============================================================================
# CALCULATE DISTANCES FROM MALE TO EACH FEMALE
# ============================================================================
#%%
print("\nCalculating distances from male to each female...")

# Extract male coordinates
male_x_col = f'x{male_id_trajectories}'
male_y_col = f'y{male_id_trajectories}'
male_x = trajectories[male_x_col].values
male_y = trajectories[male_y_col].values

# Get all fly IDs (1 to 6)
all_fly_ids = range(1, 7)
female_ids = [fid for fid in all_fly_ids if fid != male_id_trajectories]

print(f"Female IDs: {female_ids}")

# Calculate distances in pixels, then convert to cm
distances_cm = {}
for female_id in female_ids:
    female_x = trajectories[f'x{female_id}'].values
    female_y = trajectories[f'y{female_id}'].values

    # Euclidean distance
    dist_pixels = np.sqrt((male_x - female_x)**2 + (male_y - female_y)**2)
    dist_cm = dist_pixels / PIXELS_PER_CM

    distances_cm[female_id] = dist_cm

# Create DataFrame with distances
distances_df = pd.DataFrame(distances_cm)
distances_df['time'] = trajectories['time'].values

print(f"\nDistances DataFrame shape: {distances_df.shape}")
print(distances_df.head())

# ============================================================================
# DETECT PURSUIT BEHAVIOR
# ============================================================================
#%%
print(f"\nDetecting pursuit (distance < {PURSUIT_DISTANCE_CM} cm for >= {PURSUIT_MIN_DURATION} s)...")

pursuit_frames_threshold = int(PURSUIT_MIN_DURATION * frame_rate)
print(f"Pursuit minimum frames: {pursuit_frames_threshold}")

pursuit_states = {}

for female_id in female_ids:
    # Binary array: 1 if within pursuit distance, 0 otherwise
    within_distance = (distances_df[female_id] < PURSUIT_DISTANCE_CM).astype(int)

    # Apply minimum duration filter using rolling window
    pursuit = np.zeros(len(within_distance), dtype=bool)

    # Find contiguous regions of pursuit
    in_pursuit = False
    start_idx = 0

    for i in range(len(within_distance)):
        if within_distance.iloc[i] == 1 and not in_pursuit:
            # Start of potential pursuit
            in_pursuit = True
            start_idx = i
        elif within_distance.iloc[i] == 0 and in_pursuit:
            # End of pursuit region
            duration_frames = i - start_idx
            if duration_frames >= pursuit_frames_threshold:
                pursuit[start_idx:i] = True
            in_pursuit = False

    # Handle case where pursuit extends to end
    if in_pursuit:
        duration_frames = len(within_distance) - start_idx
        if duration_frames >= pursuit_frames_threshold:
            pursuit[start_idx:] = True

    pursuit_states[female_id] = pursuit
    pursuit_percentage = (pursuit.sum() / len(pursuit)) * 100
    print(f"  Female {female_id}: {pursuit_percentage:.2f}% of time in pursuit")

pursuit_df = pd.DataFrame(pursuit_states)
pursuit_df['time'] = trajectories['time'].values

# ============================================================================
# DETECT COPULATION BEHAVIOR
# ============================================================================
#%%
print(f"\nDetecting copulation (distance < {COPULATION_DISTANCE_CM} cm for >= {COPULATION_MIN_DURATION} s)...")

copulation_frames_threshold = int(COPULATION_MIN_DURATION * frame_rate)
print(f"Copulation minimum frames: {copulation_frames_threshold}")

copulation_states = {}

for female_id in female_ids:
    # Binary array: 1 if within copulation distance, 0 otherwise
    within_distance = (distances_df[female_id] < COPULATION_DISTANCE_CM).astype(int)

    # Apply minimum duration filter
    copulation = np.zeros(len(within_distance), dtype=bool)

    # Find contiguous regions of copulation
    in_copulation = False
    start_idx = 0

    for i in range(len(within_distance)):
        if within_distance.iloc[i] == 1 and not in_copulation:
            # Start of potential copulation
            in_copulation = True
            start_idx = i
        elif within_distance.iloc[i] == 0 and in_copulation:
            # End of copulation region
            duration_frames = i - start_idx
            if duration_frames >= copulation_frames_threshold:
                copulation[start_idx:i] = True
            in_copulation = False

    # Handle case where copulation extends to end
    if in_copulation:
        duration_frames = len(within_distance) - start_idx
        if duration_frames >= copulation_frames_threshold:
            copulation[start_idx:] = True

    copulation_states[female_id] = copulation
    copulation_percentage = (copulation.sum() / len(copulation)) * 100
    print(f"  Female {female_id}: {copulation_percentage:.2f}% of time in copulation")

copulation_df = pd.DataFrame(copulation_states)
copulation_df['time'] = trajectories['time'].values

# ============================================================================
# DETECT COPULATION BY ID DISAPPEARANCE (Approach B)
# ============================================================================
#%%
if COPULATION_USE_ID_DISAPPEARANCE:
    print(f"\nDetecting copulation via ID disappearance (male missing for >= {COPULATION_DISAPPEARANCE_MIN_DURATION} s)...")

    disappearance_frames_threshold = int(COPULATION_DISAPPEARANCE_MIN_DURATION * frame_rate)
    print(f"Disappearance minimum frames: {disappearance_frames_threshold}")

    # Detect when male is missing
    male_missing = np.isnan(male_x)

    # Find contiguous regions where male is missing
    in_missing = False
    start_idx = 0
    disappearance_events = []  # List of (start_idx, end_idx) tuples

    for i in range(len(male_missing)):
        if male_missing[i] and not in_missing:
            # Start of missing period
            in_missing = True
            start_idx = i
        elif not male_missing[i] and in_missing:
            # End of missing period
            duration_frames = i - start_idx
            if duration_frames >= disappearance_frames_threshold:
                disappearance_events.append((start_idx, i))
            in_missing = False

    # Handle case where missing extends to end
    if in_missing:
        duration_frames = len(male_missing) - start_idx
        if duration_frames >= disappearance_frames_threshold:
            disappearance_events.append((start_idx, len(male_missing)))

    print(f"Found {len(disappearance_events)} disappearance events")

    # For each disappearance event, identify which female was being copulated
    for event_idx, (start, end) in enumerate(disappearance_events):
        event_duration = (end - start) / frame_rate
        event_time = trajectories['time'].iloc[start]
        print(f"\n  Event {event_idx + 1}: {event_time:.1f}s ({event_time/60:.1f}min), duration: {event_duration:.1f}s")

        # Look back before disappearance to find which female was closest
        lookback_frames = int(5 * frame_rate)  # Look back 5 seconds
        lookback_start = max(0, start - lookback_frames)

        # Find which female was closest on average during lookback period
        avg_distances = {}
        for female_id in female_ids:
            distances_before = distances_df[female_id].iloc[lookback_start:start]
            # Only consider valid (non-NaN) distances
            valid_distances = distances_before.dropna()
            if len(valid_distances) > 0:
                avg_distances[female_id] = valid_distances.mean()
            else:
                avg_distances[female_id] = np.inf

        # Identify closest female
        closest_female = min(avg_distances, key=avg_distances.get)
        closest_distance = avg_distances[closest_female]

        print(f"    Closest female before disappearance: Female {closest_female} (avg dist: {closest_distance:.3f}cm)")

        # Mark this as copulation for the closest female
        copulation_df.loc[start:end-1, closest_female] = True

    # Recalculate percentages after adding disappearance-based detection
    print("\nUpdated copulation percentages (distance + disappearance):")
    for female_id in female_ids:
        copulation_percentage = (copulation_df[female_id].sum() / len(copulation_df)) * 100
        print(f"  Female {female_id}: {copulation_percentage:.2f}% of time in copulation")

# ============================================================================
# CREATE ETHOGRAM VISUALIZATION
# ============================================================================
#%%
print("\nCreating ethogram visualization...")

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

# Convert time to desired unit
if PLOT_TIME_UNIT == 'minutes':
    time_values = trajectories['time'].values / 60
    time_label = 'Time (minutes)'
else:
    time_values = trajectories['time'].values
    time_label = 'Time (seconds)'

# Plot behaviors for each female
y_positions = {fid: i for i, fid in enumerate(female_ids)}
colors = {'no_interaction': 'lightgray', 'pursuit': 'orange', 'copulation': 'red'}

for female_id in female_ids:
    y_pos = y_positions[female_id]

    # Start with baseline (no interaction)
    behavior = np.full(len(trajectories), 0)  # 0 = no interaction

    # Add pursuit (1)
    behavior[pursuit_df[female_id]] = 1

    # Add copulation (2) - copulation overrides pursuit
    behavior[copulation_df[female_id]] = 2

    # Plot each behavior state
    for i in range(len(behavior)):
        if behavior[i] == 0:
            color = colors['no_interaction']
        elif behavior[i] == 1:
            color = colors['pursuit']
        elif behavior[i] == 2:
            color = colors['copulation']

        # Plot small vertical line for each frame
        if i == 0 or behavior[i] != behavior[i-1]:
            # Plot horizontal spans instead of individual points for efficiency
            start_idx = i
            start_time = time_values[i]

            # Find end of this behavior
            end_idx = i
            while end_idx < len(behavior) - 1 and behavior[end_idx + 1] == behavior[i]:
                end_idx += 1
            end_time = time_values[end_idx]

            ax.fill_between([start_time, end_time], y_pos - 0.4, y_pos + 0.4,
                          color=color, alpha=0.8, linewidth=0)

# Customize plot
ax.set_xlabel(time_label, fontsize=12)
ax.set_ylabel('Female Fly ID', fontsize=12)
ax.set_title(f'Ethogram: Male (ID {male_id_trajectories}) Courtship Behavior', fontsize=14, fontweight='bold')
ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels([f'Female {fid}' for fid in female_ids])
ax.set_ylim(-0.5, len(female_ids) - 0.5)
ax.grid(True, axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['no_interaction'], label='No interaction'),
    Patch(facecolor=colors['pursuit'], label=f'Pursuit (< {PURSUIT_DISTANCE_CM} cm)'),
    Patch(facecolor=colors['copulation'], label=f'Copulation (< {COPULATION_DISTANCE_CM} cm)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('ethogram_camA.png', dpi=300, bbox_inches='tight')
print("Ethogram saved as 'ethogram_camA.png'")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
#%%
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for female_id in female_ids:
    print(f"\nFemale {female_id}:")

    # Pursuit statistics
    pursuit_frames = pursuit_df[female_id].sum()
    pursuit_seconds = pursuit_frames / frame_rate
    pursuit_percent = (pursuit_frames / len(pursuit_df)) * 100
    print(f"  Pursuit: {pursuit_seconds:.1f} s ({pursuit_percent:.2f}%)")

    # Copulation statistics
    copulation_frames = copulation_df[female_id].sum()
    copulation_seconds = copulation_frames / frame_rate
    copulation_percent = (copulation_frames / len(copulation_df)) * 100
    print(f"  Copulation: {copulation_seconds:.1f} s ({copulation_percent:.2f}%)")

    # Time to first copulation
    if copulation_frames > 0:
        first_copulation_idx = np.where(copulation_df[female_id])[0][0]
        time_to_copulation = trajectories['time'].iloc[first_copulation_idx]
        if PLOT_TIME_UNIT == 'minutes':
            print(f"  Time to first copulation: {time_to_copulation/60:.2f} minutes")
        else:
            print(f"  Time to first copulation: {time_to_copulation:.2f} seconds")
    else:
        print(f"  Time to first copulation: No copulation detected")

print("\n" + "="*60)
