import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARAMETERS
# ============================================================================

# File paths
TRAJECTORIES_PATH = 'data/experiment/Setup2_CamAVideo1_2026-01-26T15_38_43_trajectories.csv'
AREAS_PATH = 'data/experiment/Setup2_CamAVideo1_2026-01-26T15_38_43_areas.csv'

TRAJECTORIES_PATH = 'data/experiment/Setup2_CamAVideo1_2026-01-26T15_38_43_trajectories.csv'
AREAS_PATH = 'data/experiment/Setup2_CamAVideo1_2026-01-26T15_38_43_areas.csv'

# Experimental parameters
RECORDING_TIME = 3600  # seconds (1 hour)
EXPECTED_FRAME_RATE = 50  # fps
EXPECTED_FRAMES = RECORDING_TIME * EXPECTED_FRAME_RATE  # 180,000 frames

# Plotting parameters
PLOT_TIME_UNIT = 'minutes'  # 'seconds' or 'minutes'
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 8

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

# Load areas to identify males (two smallest)
areas = pd.read_csv(AREAS_PATH)
print(f"\nAreas DataFrame shape: {areas.shape}")
print(areas)

# Identify two males (two smallest mean areas, 0-indexed in areas file)
male_indices_areas = areas['mean'].nsmallest(2).index.tolist()
male_ids_trajectories = [idx + 1 for idx in male_indices_areas]  # Convert to 1-indexed
print(f"\nMale flies: indices {male_indices_areas} in areas file")
print(f"Male flies: IDs {male_ids_trajectories} in trajectories file")

# Load trajectories (single file this time)
trajectories = pd.read_csv(TRAJECTORIES_PATH)

print(f"\nTrajectories DataFrame shape: {trajectories.shape}")
print(f"Recording length: {trajectories['time'].iloc[-1]:.2f} seconds")

# ============================================================================
# CHECK FRAME COUNT
# ============================================================================

print("\n" + "="*60)
print("FRAME COUNT VALIDATION")
print("="*60)

actual_frames = len(trajectories)
print(f"Expected frames (50 fps × 1 hour): {EXPECTED_FRAMES}")
print(f"Actual frames: {actual_frames}")
print(f"Difference: {actual_frames - EXPECTED_FRAMES} frames")

if actual_frames == EXPECTED_FRAMES:
    print("✓ Frame count matches expected!")
elif actual_frames == EXPECTED_FRAMES + 1:
    print("✓ Frame count is expected + 1 (includes frame 0)")
else:
    print("⚠ Frame count differs from expected!")

# Calculate actual frame rate
actual_duration = trajectories['time'].iloc[-1]
calculated_frame_rate = (len(trajectories) - 1) / actual_duration  # -1 to account for starting at 0
print(f"\nCalculated frame rate: {calculated_frame_rate:.2f} fps")

# ============================================================================
# DETECT NaN LOCATIONS FOR EACH FLY
# ============================================================================

print("\n" + "="*60)
print("NaN DETECTION")
print("="*60)

# Get all fly IDs based on columns in trajectories
fly_ids = []
for col in trajectories.columns:
    if col.startswith('x'):
        fly_id = int(col[1:])
        fly_ids.append(fly_id)

print(f"Detected fly IDs: {fly_ids}")

# Detect NaN locations for each fly
nan_states = {}
nan_percentages = {}

for fly_id in fly_ids:
    x_col = f'x{fly_id}'
    y_col = f'y{fly_id}'

    # A fly has NaN location if either x or y is NaN
    has_nan = trajectories[x_col].isna() | trajectories[y_col].isna()
    nan_states[fly_id] = has_nan.values

    nan_count = has_nan.sum()
    nan_percentage = (nan_count / len(trajectories)) * 100
    nan_percentages[fly_id] = nan_percentage

    is_male = fly_id in male_ids_trajectories
    fly_type = "MALE" if is_male else "Female"
    print(f"  Fly {fly_id} ({fly_type}): {nan_count} NaN frames ({nan_percentage:.2f}%)")

# ============================================================================
# CREATE DATA QUALITY VISUALIZATION
# ============================================================================

print("\nCreating data quality visualization...")

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

# Convert time to desired unit
if PLOT_TIME_UNIT == 'minutes':
    time_values = trajectories['time'].values / 60
    time_label = 'Time (minutes)'
else:
    time_values = trajectories['time'].values
    time_label = 'Time (seconds)'

# Plot NaN locations for each fly
y_positions = {fid: i for i, fid in enumerate(fly_ids)}

for fly_id in fly_ids:
    y_pos = y_positions[fly_id]
    nan_mask = nan_states[fly_id]

    # Find contiguous NaN segments
    in_nan = False
    start_idx = 0

    for i in range(len(nan_mask)):
        if nan_mask[i] and not in_nan:
            # Start of NaN segment
            in_nan = True
            start_idx = i
        elif not nan_mask[i] and in_nan:
            # End of NaN segment
            start_time = time_values[start_idx]
            end_time = time_values[i-1]
            ax.fill_between([start_time, end_time], y_pos - 0.4, y_pos + 0.4,
                          color='black', alpha=0.8, linewidth=0)
            in_nan = False

    # Handle case where NaN extends to end
    if in_nan:
        start_time = time_values[start_idx]
        end_time = time_values[-1]
        ax.fill_between([start_time, end_time], y_pos - 0.4, y_pos + 0.4,
                      color='black', alpha=0.8, linewidth=0)

# Customize plot
ax.set_xlabel(time_label, fontsize=12)
ax.set_ylabel('Fly ID', fontsize=12)
ax.set_title('Data Quality Inspection: NaN Coordinate Locations', fontsize=14, fontweight='bold')
ax.set_yticks(list(y_positions.values()))

# Create labels with red color for males
y_labels = []
for fid in fly_ids:
    if fid in male_ids_trajectories:
        y_labels.append(f'Fly {fid} (M)')
    else:
        y_labels.append(f'Fly {fid}')

ax.set_yticklabels(y_labels)

# Color the y-tick labels red for males
for i, (fid, label) in enumerate(zip(fly_ids, ax.get_yticklabels())):
    if fid in male_ids_trajectories:
        label.set_color('red')
        label.set_fontweight('bold')

ax.set_ylim(-0.5, len(fly_ids) - 0.5)
ax.grid(True, axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='black', label='NaN coordinates (missing data)'),
    Patch(facecolor='white', edgecolor='black', label='Valid coordinates')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('data_quality_inspection.png', dpi=300, bbox_inches='tight')
print("Data quality plot saved as 'data_quality_inspection.png'")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nRecording duration: {actual_duration:.2f} seconds ({actual_duration/60:.2f} minutes)")
print(f"Frame rate: {calculated_frame_rate:.2f} fps")
print(f"Total flies: {len(fly_ids)}")
print(f"Males (smallest area): IDs {male_ids_trajectories}")

print("\nData completeness by fly:")
for fly_id in fly_ids:
    valid_percentage = 100 - nan_percentages[fly_id]
    is_male = fly_id in male_ids_trajectories
    fly_type = "MALE" if is_male else "Female"
    print(f"  Fly {fly_id} ({fly_type}): {valid_percentage:.2f}% valid data")

print("\n" + "="*60)
