import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import ndimage
import yaml
import pickle
warnings.filterwarnings('ignore')

# Use non-interactive backend for faster plotting
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load config
CONFIG = load_config()

# ============================================================================
# PARAMETERS - Loaded from config
# ============================================================================

# Experimental parameters
RECORDING_TIME = CONFIG['recording']['duration_sec']

# Behavior detection thresholds
COPULATION_DISTANCE_CM = CONFIG['behavior_thresholds']['copulation']['distance_cm']
COPULATION_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['min_duration_sec']
COPULATION_USE_ID_DISAPPEARANCE = CONFIG['behavior_thresholds']['copulation']['use_id_disappearance']
COPULATION_DISAPPEARANCE_MIN_DURATION = CONFIG['behavior_thresholds']['copulation']['disappearance_min_duration_sec']
COPULATION_GAP_MERGE_THRESHOLD = CONFIG['behavior_thresholds']['copulation']['gap_merge_threshold_sec']

PURSUIT_DISTANCE_CM = CONFIG['behavior_thresholds']['pursuit']['distance_cm']
PURSUIT_MIN_DURATION = CONFIG['behavior_thresholds']['pursuit']['min_duration_sec']

# Plotting parameters
PLOT_TIME_UNIT = CONFIG['plotting']['time_unit']
FIGURE_WIDTH = CONFIG['plotting']['figure_width']
FIGURE_HEIGHT = CONFIG['plotting']['figure_height']
DPI = CONFIG['plotting']['dpi']

# Dataset paths
DATASET_PATHS = [
    'data/1M/dataset_1',
    'data/1M/dataset_2'
]

# Output directory
OUTPUT_DIR = Path('output/ethograms')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, ignore any existing cache and reprocess runs from raw files
FORCE_REPROCESS = CONFIG.get('force_reprocess', False) if isinstance(CONFIG, dict) else False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pixels_per_cm(dataset_name, camera, setup=None):
    """
    Get the correct pixels-per-cm conversion value based on dataset, camera, and setup.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'dataset_1', 'dataset_2')
    camera : str
        Camera name (e.g., 'CamA', 'CamB')
    setup : str, optional
        Setup name for dataset_2 (e.g., 'Setup1', 'Setup2')

    Returns:
    --------
    float
        Pixels per centimeter conversion value
    """
    pixels_config = CONFIG['pixels_per_cm']

    if dataset_name == 'dataset_1':
        return pixels_config['dataset_1'][camera]
    elif dataset_name == 'dataset_2':
        if setup is None:
            raise ValueError(f"Setup must be specified for {dataset_name}")
        return pixels_config['dataset_2'][setup][camera]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def merge_copulation_gaps(copulation_array, frame_rate, gap_threshold_seconds):
    """
    Merge copulation events if gap between them is less than threshold.
    Uses vectorized operations for speed.

    Parameters:
    -----------
    copulation_array : numpy array of bool
        Boolean array indicating copulation frames
    frame_rate : float
        Frame rate in fps
    gap_threshold_seconds : float
        Maximum gap in seconds to merge

    Returns:
    --------
    numpy array of bool
        Merged copulation array
    """
    if copulation_array.sum() == 0:
        return copulation_array

    gap_threshold_frames = int(gap_threshold_seconds * frame_rate)

    # Use morphological closing to merge small gaps
    # This is much faster than loop-based approach
    from scipy.ndimage import binary_closing
    structure = np.ones(gap_threshold_frames * 2 + 1)
    merged = binary_closing(copulation_array, structure=structure)

    return merged


def resolve_simultaneous_copulation(copulation_df, distances_df, trajectories, frame_rate):
    """
    Resolve cases where male appears to copulate with multiple females simultaneously.
    Choose the copulation that would be longest.

    Parameters:
    -----------
    copulation_df : DataFrame
        DataFrame with copulation states for each female
    distances_df : DataFrame
        DataFrame with distances to each female
    trajectories : DataFrame
        Trajectories data
    frame_rate : float
        Frame rate in fps

    Returns:
    --------
    tuple: (resolved_copulation_df, conflicts_count)
    """
    resolved = copulation_df.copy()
    female_ids = [col for col in copulation_df.columns if col != 'time']
    conflicts = []

    # Find frames where male is copulating with multiple females
    copulation_counts = copulation_df[female_ids].sum(axis=1)
    simultaneous_frames = np.where(copulation_counts > 1)[0]

    if len(simultaneous_frames) == 0:
        return resolved, []

    # Group consecutive simultaneous frames into events
    events = []
    if len(simultaneous_frames) > 0:
        current_event = [simultaneous_frames[0]]
        for frame in simultaneous_frames[1:]:
            if frame == current_event[-1] + 1:
                current_event.append(frame)
            else:
                events.append(current_event)
                current_event = [frame]
        events.append(current_event)

    # Resolve each event
    for event_frames in events:
        start_frame = event_frames[0]
        end_frame = event_frames[-1]

        # Find which females are involved
        involved_females = []
        for female_id in female_ids:
            if copulation_df[female_id].iloc[start_frame:end_frame+1].any():
                involved_females.append(female_id)

        if len(involved_females) <= 1:
            continue

        # For each involved female, calculate how long the copulation would be
        # if we kept only that female
        copulation_lengths = {}
        for female_id in involved_females:
            # Find the full copulation event that includes this conflict
            female_copulating = copulation_df[female_id].values

            # Find start of this copulation event
            event_start = start_frame
            while event_start > 0 and female_copulating[event_start - 1]:
                event_start -= 1

            # Find end of this copulation event
            event_end = end_frame
            while event_end < len(female_copulating) - 1 and female_copulating[event_end + 1]:
                event_end += 1

            copulation_lengths[female_id] = event_end - event_start + 1

        # Choose the female with longest copulation
        chosen_female = max(copulation_lengths, key=copulation_lengths.get)

        # Remove copulation for other females during this conflict period
        for female_id in involved_females:
            if female_id != chosen_female:
                resolved.loc[start_frame:end_frame, female_id] = False

        # Record conflict
        conflict_time = trajectories['time'].iloc[start_frame] if 'time' in trajectories else start_frame / frame_rate
        conflicts.append({
            'time': conflict_time,
            'frames': len(event_frames),
            'involved_females': involved_females,
            'chosen_female': chosen_female,
            'lengths': copulation_lengths
        })

    return resolved, conflicts


def detect_unknown_state_per_female(trajectories, male_id, female_id, copulation_state, pursuit_state):
    """
    Detect frames where THIS FEMALE's position is unknown (NaN)
    AND not copulating or pursuing.
    These will be marked in black.

    Parameters:
    -----------
    trajectories : DataFrame
        Trajectories data
    male_id : int
        Male fly ID
    female_id : int
        Female fly ID
    copulation_state : numpy array of bool
        Copulation state for this female
    pursuit_state : numpy array of bool
        Pursuit state for this female

    Returns:
    --------
    numpy array of bool
        Boolean array indicating unknown state frames for this female
    """
    female_x_col = f'x{female_id}'
    female_y_col = f'y{female_id}'

    # Check if THIS FEMALE's position is NaN
    female_missing = np.isnan(trajectories[female_x_col].values) | np.isnan(trajectories[female_y_col].values)

    # Unknown state: THIS FEMALE is missing AND not in copulation or pursuit
    # (If she's copulating or pursuing, we know where she is by definition)
    unknown_state = female_missing & ~copulation_state & ~pursuit_state

    return unknown_state


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_run(trajectories_path, metadata_row, dataset_name):
    """
    Process a single run and generate ethogram.

    Parameters:
    -----------
    trajectories_path : Path
        Path to concatenated trajectories file
    metadata_row : Series
        Row from metadata DataFrame with run information
    dataset_name : str
        Name of the dataset (e.g., 'dataset_1')
    """
    print(f"\n{'='*80}")
    print(f"Processing: {trajectories_path.name}")
    print(f"{'='*80}")

    # Extract information from metadata
    genotype = metadata_row['genotype']
    camera = metadata_row['camera']
    date = metadata_row['date']
    time = metadata_row['time']
    male_id = metadata_row['male_id']

    # Get setup if available (for dataset_2)
    setup = metadata_row.get('setup', None) if 'setup' in metadata_row.index else None

    # Determine pixels per cm based on dataset, camera, and setup
    try:
        pixels_per_cm = get_pixels_per_cm(dataset_name, camera, setup)
    except Exception as e:
        print(f"ERROR: Could not determine pixels_per_cm: {e}")
        return

    print(f"Genotype: {genotype}")
    print(f"Camera: {camera}")
    if setup:
        print(f"Setup: {setup}")
    print(f"Date: {date}, Time: {time}")
    print(f"Male ID: {male_id}")
    print(f"Pixels per cm: {pixels_per_cm}")

    # Load trajectories
    try:
        trajectories = pd.read_csv(trajectories_path)
    except Exception as e:
        print(f"ERROR: Could not load trajectories: {e}")
        return

    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Recording length: {trajectories['time'].iloc[-1]:.2f} seconds")

    # Calculate frame rate
    frame_rate = len(trajectories) / trajectories['time'].iloc[-1]
    print(f"Frame rate: {frame_rate:.2f} fps")

    # Extract male coordinates
    male_x_col = f'x{male_id}'
    male_y_col = f'y{male_id}'

    if male_x_col not in trajectories.columns or male_y_col not in trajectories.columns:
        print(f"ERROR: Male coordinates not found in trajectories")
        return

    male_x = trajectories[male_x_col].values
    male_y = trajectories[male_y_col].values

    # Get all fly IDs
    all_fly_ids = range(1, 7)
    female_ids = [fid for fid in all_fly_ids if fid != male_id]
    print(f"Female IDs: {female_ids}")

    # ============================================================================
    # CALCULATE DISTANCES
    # ============================================================================
    print("\nCalculating distances...")
    distances_cm = {}
    for female_id in female_ids:
        female_x = trajectories[f'x{female_id}'].values
        female_y = trajectories[f'y{female_id}'].values

        # Euclidean distance
        dist_pixels = np.sqrt((male_x - female_x)**2 + (male_y - female_y)**2)
        dist_cm = dist_pixels / pixels_per_cm

        distances_cm[female_id] = dist_cm

    distances_df = pd.DataFrame(distances_cm)
    distances_df['time'] = trajectories['time'].values

    # ============================================================================
    # DETECT PURSUIT (VECTORIZED)
    # ============================================================================
    print(f"\nDetecting pursuit (< {PURSUIT_DISTANCE_CM} cm for >= {PURSUIT_MIN_DURATION} s)...")
    pursuit_frames_threshold = int(PURSUIT_MIN_DURATION * frame_rate)

    pursuit_states = {}
    for female_id in female_ids:
        within_distance = (distances_df[female_id] < PURSUIT_DISTANCE_CM).values.astype(int)

        # Use scipy.ndimage.label to find connected components
        labeled_array, num_features = ndimage.label(within_distance)
        pursuit = np.zeros(len(within_distance), dtype=bool)

        # Check each connected component
        for region_label in range(1, num_features + 1):
            region_mask = labeled_array == region_label
            region_length = region_mask.sum()
            if region_length >= pursuit_frames_threshold:
                pursuit[region_mask] = True

        pursuit_states[female_id] = pursuit
        pursuit_percentage = (pursuit.sum() / len(pursuit)) * 100
        print(f"  Female {female_id}: {pursuit_percentage:.2f}% in pursuit")

    pursuit_df = pd.DataFrame(pursuit_states)
    pursuit_df['time'] = trajectories['time'].values

    # ============================================================================
    # DETECT COPULATION (distance-based, VECTORIZED)
    # ============================================================================
    print(f"\nDetecting copulation (< {COPULATION_DISTANCE_CM} cm for >= {COPULATION_MIN_DURATION} s)...")
    copulation_frames_threshold = int(COPULATION_MIN_DURATION * frame_rate)

    copulation_states = {}
    for female_id in female_ids:
        within_distance = (distances_df[female_id] < COPULATION_DISTANCE_CM).values.astype(int)

        # Use scipy.ndimage.label to find connected components
        labeled_array, num_features = ndimage.label(within_distance)
        copulation = np.zeros(len(within_distance), dtype=bool)

        # Check each connected component
        for region_label in range(1, num_features + 1):
            region_mask = labeled_array == region_label
            region_length = region_mask.sum()
            if region_length >= copulation_frames_threshold:
                copulation[region_mask] = True

        copulation_states[female_id] = copulation

    copulation_df = pd.DataFrame(copulation_states)
    copulation_df['time'] = trajectories['time'].values

    # ============================================================================
    # DETECT COPULATION BY ID DISAPPEARANCE
    # ============================================================================
    if COPULATION_USE_ID_DISAPPEARANCE:
        print(f"\nDetecting copulation via male disappearance (>= {COPULATION_DISAPPEARANCE_MIN_DURATION} s)...")
        disappearance_frames_threshold = int(COPULATION_DISAPPEARANCE_MIN_DURATION * frame_rate)

        male_missing = np.isnan(male_x)

        in_missing = False
        start_idx = 0
        disappearance_events = []

        for i in range(len(male_missing)):
            if male_missing[i] and not in_missing:
                in_missing = True
                start_idx = i
            elif not male_missing[i] and in_missing:
                duration_frames = i - start_idx
                if duration_frames >= disappearance_frames_threshold:
                    disappearance_events.append((start_idx, i))
                in_missing = False

        if in_missing:
            duration_frames = len(male_missing) - start_idx
            if duration_frames >= disappearance_frames_threshold:
                disappearance_events.append((start_idx, len(male_missing)))

        print(f"Found {len(disappearance_events)} disappearance events")

        for event_idx, (start, end) in enumerate(disappearance_events):
            event_duration = (end - start) / frame_rate
            event_time = trajectories['time'].iloc[start]

            lookback_frames = int(5 * frame_rate)
            lookback_start = max(0, start - lookback_frames)

            avg_distances = {}
            for female_id in female_ids:
                distances_before = distances_df[female_id].iloc[lookback_start:start]
                valid_distances = distances_before.dropna()
                if len(valid_distances) > 0:
                    avg_distances[female_id] = valid_distances.mean()
                else:
                    avg_distances[female_id] = np.inf

            closest_female = min(avg_distances, key=avg_distances.get)
            copulation_df.loc[start:end-1, closest_female] = True

        # ============================================================================
        # DETECT COPULATION BY FEMALE DISAPPEARANCE
        # ============================================================================
        print(f"\nDetecting copulation via female disappearance (>= {COPULATION_DISAPPEARANCE_MIN_DURATION} s)...")

        for female_id in female_ids:
            female_x = trajectories[f'x{female_id}'].values
            female_y = trajectories[f'y{female_id}'].values
            female_missing = np.isnan(female_x) | np.isnan(female_y)

            # Find contiguous regions where female is missing
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
                        female_disappearance_events.append((start_idx, i))
                    in_missing = False

            # Handle case where missing extends to end
            if in_missing:
                duration_frames = len(female_missing) - start_idx
                if duration_frames >= disappearance_frames_threshold:
                    female_disappearance_events.append((start_idx, len(female_missing)))

            if len(female_disappearance_events) > 0:
                print(f"  Female {female_id}: Found {len(female_disappearance_events)} disappearance events")

                for event_idx, (start, end) in enumerate(female_disappearance_events):
                    event_duration = (end - start) / frame_rate
                    event_time = trajectories['time'].iloc[start]

                    # Check if male is present during this time (not missing)
                    # If male is also missing, we already handled it in male disappearance
                    male_present_during = ~male_missing[start:end]

                    if male_present_during.any():
                        # Male is present for at least part of this female disappearance
                        # This suggests copulation
                        print(f"    Event {event_idx + 1}: {event_time:.1f}s ({event_time/60:.1f}min), duration: {event_duration:.1f}s (male present)")
                        copulation_df.loc[start:end-1, female_id] = True
                    else:
                        print(f"    Event {event_idx + 1}: {event_time:.1f}s ({event_time/60:.1f}min), duration: {event_duration:.1f}s (male also missing - skipped)")

    # ============================================================================
    # MERGE COPULATION GAPS
    # ============================================================================
    print(f"\nMerging copulation gaps (< {COPULATION_GAP_MERGE_THRESHOLD} s)...")
    for female_id in female_ids:
        original_sum = copulation_df[female_id].sum()
        copulation_df[female_id] = merge_copulation_gaps(
            copulation_df[female_id].values,
            frame_rate,
            COPULATION_GAP_MERGE_THRESHOLD
        )
        merged_sum = copulation_df[female_id].sum()
        if merged_sum > original_sum:
            added_frames = merged_sum - original_sum
            added_seconds = added_frames / frame_rate
            print(f"  Female {female_id}: Added {added_seconds:.1f}s by merging gaps")

    # ============================================================================
    # RESOLVE SIMULTANEOUS COPULATION
    # ============================================================================
    print("\nResolving simultaneous copulation conflicts...")
    copulation_df, conflicts = resolve_simultaneous_copulation(
        copulation_df, distances_df, trajectories, frame_rate
    )

    if len(conflicts) > 0:
        print(f"  WARNING: Found {len(conflicts)} simultaneous copulation conflicts!")
        for i, conflict in enumerate(conflicts, 1):
            print(f"    Conflict {i} at t={conflict['time']:.1f}s:")
            print(f"      Involved: {conflict['involved_females']}")
            print(f"      Chosen: Female {conflict['chosen_female']} (longest)")
    else:
        print("  No simultaneous copulation conflicts found.")

    # Note: Unknown state will be computed per female during plotting

    # ============================================================================
    # PRINT COPULATION STATISTICS
    # ============================================================================
    print("\nFinal copulation statistics:")
    for female_id in female_ids:
        copulation_percentage = (copulation_df[female_id].sum() / len(copulation_df)) * 100
        copulation_seconds = copulation_df[female_id].sum() / frame_rate
        print(f"  Female {female_id}: {copulation_seconds:.1f}s ({copulation_percentage:.2f}%)")

    # ============================================================================
    # CREATE ETHOGRAM
    # ============================================================================
    print("\nCreating ethogram...")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Convert time to desired unit
    if PLOT_TIME_UNIT == 'minutes':
        time_values = trajectories['time'].values / 60
        time_label = 'Time (minutes)'
    else:
        time_values = trajectories['time'].values
        time_label = 'Time (seconds)'

    # Define colors
    colors = {
        'unknown': 'black',
        'no_interaction': 'lightgray',
        'pursuit': 'orange',
        'copulation': 'red'
    }

    # Build behavior matrix for all females (much faster than individual plotting)
    behavior_matrix = np.zeros((len(female_ids), len(trajectories)))

    for i, female_id in enumerate(female_ids):
        # Detect unknown state for this specific female
        unknown_state_female = detect_unknown_state_per_female(
            trajectories, male_id, female_id,
            copulation_df[female_id].values,
            pursuit_df[female_id].values
        )

        # Start with baseline (no interaction)
        behavior = np.full(len(trajectories), 0)  # 0 = no interaction

        # Add pursuit (1)
        behavior[pursuit_df[female_id]] = 1

        # Add copulation (2) - copulation overrides pursuit
        behavior[copulation_df[female_id]] = 2

        # Add unknown state (3) - unknown overrides everything for this row
        behavior[unknown_state_female] = 3

        behavior_matrix[i, :] = behavior

    # Create custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
        colors['no_interaction'],  # 0
        colors['pursuit'],         # 1
        colors['copulation'],      # 2
        colors['unknown']          # 3
    ])

    # Plot using imshow (much faster!)
    im = ax.imshow(behavior_matrix, aspect='auto', cmap=cmap,
                   extent=[time_values[0], time_values[-1], -0.5, len(female_ids) - 0.5],
                   origin='lower', interpolation='nearest', vmin=0, vmax=3)

    # Customize plot
    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel('Female Fly ID', fontsize=12)

    # Create title with dataset and genotype info
    if 'setup' in metadata_row.index:
        setup = metadata_row['setup']
        title = f'Ethogram: {dataset_name} - {genotype} - {camera} - {setup}\n' + \
                f'Male ID {male_id} Courtship Behavior ({date} {time})'
    else:
        title = f'Ethogram: {dataset_name} - {genotype} - {camera}\n' + \
                f'Male ID {male_id} Courtship Behavior ({date} {time})'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(female_ids)))
    ax.set_yticklabels([f'Female {fid}' for fid in female_ids])
    ax.set_ylim(-0.5, len(female_ids) - 0.5)
    ax.grid(True, axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['unknown'], label='Unknown position'),
        Patch(facecolor=colors['no_interaction'], label='No interaction'),
        Patch(facecolor=colors['pursuit'], label=f'Pursuit (< {PURSUIT_DISTANCE_CM} cm)'),
        Patch(facecolor=colors['copulation'], label=f'Copulation (< {COPULATION_DISTANCE_CM} cm)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    # Instead of saving a per-run figure, return the behavior matrix and metadata
    plt.close()

    # ============================================================================
    # GENERATE ANNOTATION STATISTICS
    # ============================================================================
    print("\nGenerating annotation statistics...")

    annotations = {
        'dataset': dataset_name,
        'genotype': genotype,
        'camera': camera,
        'date': date,
        'time': time,
        'male_id': male_id,
        'pixels_per_cm': pixels_per_cm
    }

    if 'setup' in metadata_row.index:
        annotations['setup'] = metadata_row['setup']

    # Overall statistics
    all_copulation_frames = copulation_df[female_ids].any(axis=1)
    total_copulation_time = all_copulation_frames.sum() / frame_rate / 60  # minutes
    annotations['total_copulation_min'] = total_copulation_time

    all_pursuit_frames = pursuit_df[female_ids].any(axis=1)
    total_pursuit_time = all_pursuit_frames.sum() / frame_rate / 60  # minutes
    annotations['total_pursuit_min'] = total_pursuit_time

    # Find all copulation events across all females
    copulation_events = []
    for female_id in female_ids:
        female_copulating = copulation_df[female_id].values
        if female_copulating.sum() == 0:
            continue

        # Find copulation events for this female
        labeled_array, num_events = ndimage.label(female_copulating)
        for event_label in range(1, num_events + 1):
            event_mask = labeled_array == event_label
            event_indices = np.where(event_mask)[0]
            start_idx = event_indices[0]
            end_idx = event_indices[-1]
            duration = (end_idx - start_idx + 1) / frame_rate / 60  # minutes

            copulation_events.append({
                'female_id': female_id,
                'start_time_min': trajectories['time'].iloc[start_idx] / 60,
                'end_time_min': trajectories['time'].iloc[end_idx] / 60,
                'duration_min': duration
            })

    annotations['number_of_copulations'] = len(copulation_events)

    if len(copulation_events) > 0:
        longest_copulation = max(copulation_events, key=lambda x: x['duration_min'])
        annotations['longest_copulation_min'] = longest_copulation['duration_min']
        annotations['longest_copulation_start_min'] = longest_copulation['start_time_min']
        annotations['longest_copulation_end_min'] = longest_copulation['end_time_min']

        # Store all copulation events as a string
        copulation_list = "; ".join([
            f"Female {e['female_id']}: {e['start_time_min']:.2f}-{e['end_time_min']:.2f}min ({e['duration_min']:.2f}min)"
            for e in copulation_events
        ])
        annotations['copulation_events'] = copulation_list
    else:
        annotations['longest_copulation_min'] = 0
        annotations['longest_copulation_start_min'] = np.nan
        annotations['longest_copulation_end_min'] = np.nan
        annotations['copulation_events'] = ""

    # Count number of females copulated with
    females_copulated = [fid for fid in female_ids if copulation_df[fid].sum() > 0]
    annotations['number_of_females_copulated'] = len(females_copulated)
    annotations['females_copulated_ids'] = ",".join(map(str, females_copulated))

    # Find all pursuit events
    pursuit_events = []
    for female_id in female_ids:
        female_pursuing = pursuit_df[female_id].values
        if female_pursuing.sum() == 0:
            continue

        # Find pursuit events for this female
        labeled_array, num_events = ndimage.label(female_pursuing)
        for event_label in range(1, num_events + 1):
            event_mask = labeled_array == event_label
            event_indices = np.where(event_mask)[0]
            start_idx = event_indices[0]
            end_idx = event_indices[-1]
            duration = (end_idx - start_idx + 1) / frame_rate / 60  # minutes

            pursuit_events.append({
                'female_id': female_id,
                'start_time_min': trajectories['time'].iloc[start_idx] / 60,
                'end_time_min': trajectories['time'].iloc[end_idx] / 60,
                'duration_min': duration
            })

    annotations['number_of_pursuits'] = len(pursuit_events)

    if len(pursuit_events) > 0:
        longest_pursuit = max(pursuit_events, key=lambda x: x['duration_min'])
        annotations['longest_pursuit_min'] = longest_pursuit['duration_min']
        annotations['longest_pursuit_start_min'] = longest_pursuit['start_time_min']
        annotations['longest_pursuit_end_min'] = longest_pursuit['end_time_min']
    else:
        annotations['longest_pursuit_min'] = 0
        annotations['longest_pursuit_start_min'] = np.nan
        annotations['longest_pursuit_end_min'] = np.nan

    # -----------------------------
    # Additional metrics (speeds and times)
    # -----------------------------
    t = trajectories['time'].values

    # Compute instantaneous speeds (cm/s) for each fly using gradient
    speeds_cm_s = {}
    for fid in range(1, 7):
        x_col = f'x{fid}'
        y_col = f'y{fid}'
        if x_col in trajectories.columns and y_col in trajectories.columns:
            x = trajectories[x_col].values
            y = trajectories[y_col].values
            # gradients (pixels per second)
            with np.errstate(invalid='ignore'):
                dx_dt = np.gradient(x, t)
                dy_dt = np.gradient(y, t)
                speed_pix_s = np.sqrt(dx_dt**2 + dy_dt**2)
                speed_cm_s = speed_pix_s / pixels_per_cm
            speeds_cm_s[fid] = speed_cm_s
        else:
            speeds_cm_s[fid] = np.full(len(t), np.nan)

    # Masks
    pursuit_mask_any = pursuit_df[female_ids].any(axis=1).values
    copulation_mask_any = copulation_df[female_ids].any(axis=1).values

    # 1) Time to first copulation (minutes)
    if len(copulation_events) > 0:
        annotations['time_to_first_copulation_min'] = min(e['start_time_min'] for e in copulation_events)
    else:
        annotations['time_to_first_copulation_min'] = np.nan

    # 2) Time to first persistent pursuit (>1 min)
    persistent_pursuits = [p for p in pursuit_events if p['duration_min'] >= 1.0]
    if len(persistent_pursuits) > 0:
        annotations['time_to_first_persistent_pursuit_min'] = min(p['start_time_min'] for p in persistent_pursuits)
    else:
        annotations['time_to_first_persistent_pursuit_min'] = np.nan

    # 3) Speed of female being pursued (average across all pursuit frames for all females)
    pursued_speeds = []
    for fid in female_ids:
        mask = pursuit_df[fid].values
        sp = speeds_cm_s.get(fid, np.full(len(t), np.nan))
        valid = sp[mask]
        if len(valid) > 0:
            pursued_speeds.append(np.nanmean(valid))
    annotations['speed_female_being_pursued_cm_s'] = np.nan if len(pursued_speeds) == 0 else float(np.nanmean(pursued_speeds))

    # 4) Speed of male while pursuing (male speed when any pursuit active)
    male_speed = speeds_cm_s.get(male_id, np.full(len(t), np.nan))
    male_pursuit_speeds = male_speed[pursuit_mask_any]
    annotations['speed_male_while_pursuing_cm_s'] = np.nan if len(male_pursuit_speeds) == 0 else float(np.nanmean(male_pursuit_speeds))

    # 5) Speed of male when not in pursuit and not in copulation and not nan
    mask_male_idle = (~pursuit_mask_any) & (~copulation_mask_any) & (~np.isnan(male_speed))
    idle_male_speeds = male_speed[mask_male_idle]
    annotations['speed_male_idle_cm_s'] = np.nan if len(idle_male_speeds) == 0 else float(np.nanmean(idle_male_speeds))

    # 6) Speed of females when not in pursuit and not in copulation and not nan (average across females)
    female_idle_speeds = []
    for fid in female_ids:
        sp = speeds_cm_s.get(fid, np.full(len(t), np.nan))
        mask_f_idle = (~pursuit_df[fid].values) & (~copulation_df[fid].values) & (~np.isnan(sp))
        vals = sp[mask_f_idle]
        if len(vals) > 0:
            female_idle_speeds.append(np.nanmean(vals))
    annotations['speed_females_idle_cm_s'] = np.nan if len(female_idle_speeds) == 0 else float(np.nanmean(female_idle_speeds))

    # 7) Speed of the rest of the females during copulation (when any female is copulating,
    #    take speeds of females that are NOT copulating at those times)
    rest_speeds = []
    any_cop = copulation_mask_any
    if any_cop.any():
        for fid in female_ids:
            sp = speeds_cm_s.get(fid, np.full(len(t), np.nan))
            mask_rest = any_cop & (~copulation_df[fid].values) & (~np.isnan(sp))
            vals = sp[mask_rest]
            if len(vals) > 0:
                rest_speeds.append(np.nanmean(vals))
    annotations['speed_rest_females_during_copulation_cm_s'] = np.nan if len(rest_speeds) == 0 else float(np.nanmean(rest_speeds))

    print(f"  Total copulation: {total_copulation_time:.2f} min")
    print(f"  Total pursuit: {total_pursuit_time:.2f} min")
    print(f"  Number of copulations: {len(copulation_events)}")
    print(f"  Number of females copulated: {len(females_copulated)}")

    # Build behavior matrix to return (rows: female x runs, cols: time)
    # Note: we construct the same behavior_matrix used for plotting above
    behavior_matrix = behavior_matrix  # already defined in this function
    time_values = trajectories['time'].values if PLOT_TIME_UNIT == 'seconds' else trajectories['time'].values / 60.0

    return annotations, behavior_matrix, female_ids, time_values


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def generate_comparison_plots(annotations_df):
    """
    Generate comparison boxplots between datasets.

    Parameters:
    -----------
    annotations_df : DataFrame
        DataFrame with annotations for all runs
    """
    print("\nGenerating comparison plots...")

    # Metrics to compare
    metrics = [
        ('total_copulation_min', 'Total Copulation (min)'),
        ('total_pursuit_min', 'Total Pursuit (min)'),
        ('longest_copulation_min', 'Longest Copulation (min)'),
        ('longest_pursuit_min', 'Longest Pursuit (min)'),
        ('number_of_copulations', 'Number of Copulations'),
        ('number_of_females_copulated', 'Number of Females Copulated')
    ]

    # Additional metrics requested
    metrics.extend([
        ('time_to_first_copulation_min', 'Time to First Copulation (min)'),
        ('time_to_first_persistent_pursuit_min', 'Time to First Persistent Pursuit (min)'),
        ('speed_female_being_pursued_cm_s', 'Speed Female Being Pursued (cm/s)'),
        ('speed_male_while_pursuing_cm_s', 'Speed Male While Pursuing (cm/s)'),
        ('speed_male_idle_cm_s', 'Speed Male Idle (cm/s)'),
        ('speed_females_idle_cm_s', 'Speed Females Idle (cm/s)'),
        ('speed_rest_females_during_copulation_cm_s', 'Speed Other Females During Copulation (cm/s)')
    ])

    # Create a figure with subplots arranged into 3 rows
    import math
    n_metrics = len(metrics)
    nrows = 3
    ncols = math.ceil(n_metrics / nrows)

    # Per-subplot sizing
    per_w = 4
    per_h = 2
    fig_w = per_w * ncols
    fig_h = max(per_h * nrows, 15)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    datasets = annotations_df['dataset'].unique()

    for idx, (metric, title) in enumerate(metrics):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx][col_idx]

        # Prepare data for plotting (drop NaNs)
        plot_data = []
        plot_labels = []
        plot_nonan_lengths = []

        for dataset in sorted(datasets):
            dataset_values = annotations_df[annotations_df['dataset'] == dataset][metric].dropna().values
            if dataset_values.size == 0:
                # Use a single NaN so boxplot receives an entry (it will render empty)
                plot_data.append(np.array([np.nan]))
                plot_nonan_lengths.append(0)
            else:
                plot_data.append(dataset_values)
                plot_nonan_lengths.append(len(dataset_values))
            plot_labels.append(dataset)

        # Create boxplot (will handle NaN entries)
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                       widths=0.6, showfliers=False)

        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add jittered points only for non-empty datasets
        for i, (dataset_values, nval) in enumerate(zip(plot_data, plot_nonan_lengths)):
            if nval > 0:
                x = np.random.normal(i + 1, 0.04, size=nval)
                # dataset_values may include NaNs if we used placeholder; ensure slicing correct
                vals = dataset_values if len(dataset_values) == nval else dataset_values[:nval]
                ax.scatter(x, vals, alpha=0.6, s=50, color='black', zorder=3)

        # Customize plot
        ax.set_ylabel(title, fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        # Add sample sizes below plot area
        for i, dataset in enumerate(plot_labels):
            n = plot_nonan_lengths[i]
            ax.text(i + 1, ax.get_ylim()[0], f'n={n}',
                   ha='center', va='top', fontsize=8)

        # If this metric is a speed metric, fix y-range to 0-1.3
        speed_metrics = {
            'speed_female_being_pursued_cm_s',
            'speed_male_while_pursuing_cm_s',
            'speed_male_idle_cm_s',
            'speed_females_idle_cm_s',
            'speed_rest_females_during_copulation_cm_s'
        }
        if metric in speed_metrics:
            ax.set_ylim(0, 1)

    # Hide any unused axes
    total_slots = nrows * ncols
    for empty_idx in range(n_metrics, total_slots):
        r = empty_idx // ncols
        c = empty_idx % ncols
        axes[r][c].axis('off')

    # Increase spacing between subplots and leave room
    plt.subplots_adjust(hspace=0.6, wspace=0.4, top=0.92)

    plt.tight_layout()

    plt.tight_layout()

    # Save comparison plot
    comparison_path = OUTPUT_DIR / 'dataset_comparison.png'
    plt.savefig(comparison_path, dpi=DPI, bbox_inches='tight')
    print(f"Comparison plot saved: {comparison_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY DATASET")
    print("="*80)

    for dataset in sorted(annotations_df['dataset'].unique()):
        dataset_data = annotations_df[annotations_df['dataset'] == dataset]
        print(f"\n{dataset} (n={len(dataset_data)}):")

        for metric, title in metrics:
            values = dataset_data[metric].values
            print(f"  {title}:")
            print(f"    Mean ± SD: {values.mean():.2f} ± {values.std():.2f}")
            print(f"    Median (IQR): {np.median(values):.2f} ({np.percentile(values, 25):.2f}-{np.percentile(values, 75):.2f})")
            print(f"    Range: {values.min():.2f} - {values.max():.2f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to process all datasets."""
    print("="*80)
    print("BATCH ETHOGRAM GENERATION")
    print("="*80)

    total_processed = 0
    all_annotations = []

    for dataset_path in DATASET_PATHS:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            print(f"\nWARNING: Dataset directory not found: {dataset_path}")
            continue

        dataset_name = dataset_dir.name
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")

        # Load metadata
        metadata_path = dataset_dir / 'metadata.csv'
        if not metadata_path.exists():
            print(f"ERROR: metadata.csv not found in {dataset_path}")
            continue

        metadata = pd.read_csv(metadata_path)
        print(f"Loaded metadata with {len(metadata)} entries")

        # Get trajectory files
        trajectories_dir = dataset_dir / 'trajectories'
        if not trajectories_dir.exists():
            print(f"ERROR: trajectories directory not found in {dataset_path}")
            continue

        trajectory_files = list(trajectories_dir.glob('*_trajectories.csv'))
        print(f"Found {len(trajectory_files)} trajectory files")

        # Prepare container for per-run behavior matrices for this dataset
        dataset_matrices = []

        # Check for cached processing for this dataset
        cache_path = OUTPUT_DIR / f"{dataset_name}_processing.pkl"
        if cache_path.exists() and not FORCE_REPROCESS:
            try:
                print(f"\nLoading cached processing for dataset {dataset_name} from {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                dataset_matrices = cached.get('dataset_matrices', [])
                cached_annotations = cached.get('annotations', [])
                # Add cached annotations to global list
                all_annotations.extend(cached_annotations)
                print(f"Loaded {len(dataset_matrices)} runs from cache")
            except Exception as e:
                print(f"WARNING: Failed to load cache {cache_path}: {e}\nWill reprocess runs.")

        # Process each trajectory file (only if cache wasn't loaded)
        if len(dataset_matrices) == 0:
            for traj_file in sorted(trajectory_files):
                # Find corresponding metadata entry
                # Extract key parts from trajectory filename: genotype, camera, timestamp
                # Example: DGRP375_CamA_2026-01-12T15_31_08_trajectories.csv
                traj_name = traj_file.stem.replace('_trajectories', '')
                traj_parts = traj_name.split('_')

                # Extract genotype, camera, and timestamp
                if len(traj_parts) >= 3:
                    genotype = traj_parts[0]
                    camera = traj_parts[1]
                    # Timestamp is the rest (could be date + time or date + time + setup)
                    timestamp_str = '_'.join(traj_parts[2:])

                    # Find exact match in metadata
                    # Match on genotype, camera, and timestamp
                    found = False
                    for idx, row in metadata.iterrows():
                        metadata_file = Path(row['filename']).stem
                        # Check if this metadata row matches the trajectory file
                        if (metadata_file.startswith(genotype) and
                            camera in metadata_file and
                            timestamp_str in metadata_file and
                            'trajectories' in row['filename']):
                            result = process_run(traj_file, row, dataset_name)
                            if result:
                                annotations, behavior_matrix, female_ids, time_values = result
                                all_annotations.append(annotations)
                                dataset_matrices.append({'matrix': np.array(behavior_matrix), 'female_ids': female_ids, 'time_values': np.array(time_values), 'label': f"{annotations['genotype']}_{annotations['camera']}_{annotations.get('setup','')}_{annotations['date']}_{annotations['time']}"})
                                total_processed += 1
                            found = True
                            break

                    if not found:
                        print(f"\nWARNING: No exact metadata match found for {traj_file.name}")
                else:
                    print(f"\nWARNING: Unexpected filename format: {traj_file.name}")

        # After processing trajectory files for this dataset, create a figure with
        # one subplot per session (run). Arrange subplots in a small grid.
        if len(dataset_matrices) > 0:
            import math
            from matplotlib.colors import ListedColormap
            from matplotlib.patches import Patch

            n_runs = len(dataset_matrices)
            # Layout: allow dataset-specific overrides. Use 2x2 for dataset_2, otherwise up to 4 columns.
            if dataset_name == 'dataset_2':
                ncols = 2
            else:
                ncols = min(4, n_runs)
            nrows = math.ceil(n_runs / ncols)

            # Figure size: make each subplot height:width = 1:2
            per_subplot_width = 4.0
            per_subplot_height = per_subplot_width / 2.0  # height:width = 1:2
            fig_w = per_subplot_width * ncols
            fig_h = max(per_subplot_height * nrows, 4.0)
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

            # Colormap (0=no interaction,1=pursuit,2=copulation,3=unknown)
            cmap = ListedColormap(['lightgray', 'orange', 'red', 'black'])

            # For each run, plot its behavior matrix in its own subplot
            for idx, m in enumerate(dataset_matrices):
                row_idx = idx // ncols
                col_idx = idx % ncols
                ax = axes[row_idx][col_idx]

                mat = m['matrix']
                rows, cols = mat.shape
                time_vals = m['time_values']
                t0 = float(time_vals[0])
                t1 = float(time_vals[-1])

                im = ax.imshow(mat, aspect='auto', cmap=cmap,
                               extent=[t0, t1, -0.5, rows - 0.5], origin='lower', interpolation='nearest', vmin=0, vmax=3)

                # Title and y labels
                label = m.get('label', '')
                ax.set_title(label, fontsize=10)
                ax.set_yticks(range(rows))
                ax.set_yticklabels([f'F{fid}' for fid in m['female_ids']], fontsize=8)
                ax.grid(True, axis='x', alpha=0.2)

                # X label only on bottom row
                if row_idx == nrows - 1:
                    if PLOT_TIME_UNIT == 'minutes':
                        ax.set_xlabel('Time (minutes)')
                    else:
                        ax.set_xlabel('Time (seconds)')
                else:
                    ax.set_xticks([])

            # Hide any unused subplots
            total_slots = nrows * ncols
            for empty_idx in range(n_runs, total_slots):
                r = empty_idx // ncols
                c = empty_idx % ncols
                axes[r][c].axis('off')

            # Add single legend for the figure placed at the top-right outside subplots
            legend_elements = [Patch(facecolor='black', label='Unknown position'),
                               Patch(facecolor='lightgray', label='No interaction'),
                               Patch(facecolor='orange', label=f'Pursuit (< {PURSUIT_DISTANCE_CM} cm)'),
                               Patch(facecolor='red', label=f'Copulation (< {COPULATION_DISTANCE_CM} cm)')]

            # Place legend anchored to the top-right of the figure (outside subplots)

            # Place legend anchored to the top-right of the figure (outside subplots).
            # Move further out so it doesn't obscure any subplot content.
            fig.legend(handles=legend_elements,
                       loc='upper right',
                       bbox_to_anchor=(1.25, 1),
                       bbox_transform=fig.transFigure,
                       fontsize=9,
                       frameon=False)

            # Adjust layout to leave room for the legend and avoid overlap
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            out_path = OUTPUT_DIR / f"{dataset_name}_sessions_subplots.png"
            fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved per-session subplots for dataset {dataset_name} to {out_path}")

            # Save processing cache for this dataset so replotting can be done without reprocessing
            try:
                cache_content = {
                    'dataset_matrices': dataset_matrices,
                    'annotations': [a for a in all_annotations if a.get('dataset') == dataset_name]
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_content, f)
                print(f"Saved processing cache to: {cache_path}")
            except Exception as e:
                print(f"WARNING: Failed to save processing cache: {e}")

    # Save all annotations to CSV
    if len(all_annotations) > 0:
        annotations_df = pd.DataFrame(all_annotations)
        annotations_path = OUTPUT_DIR / 'all_annotations.csv'
        annotations_df.to_csv(annotations_path, index=False)
        print(f"\nAnnotations saved to: {annotations_path}")

        # Generate comparison plots
        print("\n" + "="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80)
        generate_comparison_plots(annotations_df)
    else:
        print("\nNo annotations generated - skipping comparison plots")

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
