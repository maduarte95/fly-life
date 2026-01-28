import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ============================================================================
# PARAMETERS
# ============================================================================

# Paths
DATA_DIR = Path('data/1M')
OUTPUT_DIR = Path('output/nangrams')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting parameters
PLOT_TIME_UNIT = 'minutes'
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 8

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_filename(filename):
    """Extract genotype, camera, date, time, and setup from filename."""
    # Pattern: GENOTYPE_Camera_DATE_TIME[_Setup]_trajectories.csv
    # Examples:
    #   DGRP375_CamA_2026-01-12T08_31_36_trajectories.csv
    #   DGRP375_CamA_2026-01-27T09_40_53_Setup2_trajectories.csv

    parts = filename.replace('_trajectories.csv', '').split('_')
    genotype = parts[0]
    camera = parts[1]

    # Find date-time part (contains 'T')
    datetime_str = None
    for i, part in enumerate(parts):
        if 'T' in part:
            datetime_str = parts[i]
            break

    # Check if there's a setup specification
    setup = None
    if len(parts) > 3 and 'Setup' in parts[-1]:
        setup = parts[-1]

    return genotype, camera, datetime_str, setup


def load_metadata(dataset_name):
    """Load metadata for a dataset."""
    metadata_path = DATA_DIR / dataset_name / 'metadata.csv'
    return pd.read_csv(metadata_path)


def get_male_id_from_metadata(metadata, filename):
    """Get male_id from metadata for a given trajectory filename."""
    # Try exact match first
    match = metadata[metadata['filename'] == filename]
    if not match.empty:
        return match.iloc[0]['male_id']

    # Try without Video suffix for dataset_2 files
    # Convert DGRP375_CamA_2026-01-27T09_40_53_Setup2_trajectories.csv
    # to DGRP375_CamAVideo1_2026-01-27T09_40_53_Setup2_trajectories.csv pattern
    if 'Video' not in filename:
        # For dataset_2, add Video1 for matching
        base = filename.replace('_trajectories.csv', '')
        parts = base.split('_')
        # Insert Video1 after camera
        genotype = parts[0]
        camera = parts[1]
        rest = '_'.join(parts[2:])
        new_filename = f"{genotype}_{camera}Video1_{rest}_trajectories.csv"
        match = metadata[metadata['filename'] == new_filename]
        if not match.empty:
            return match.iloc[0]['male_id']

    print(f"  Warning: Could not find male_id for {filename} in metadata")
    return None


def process_trajectory_file(traj_path, dataset_name, metadata):
    """Process a single trajectory file and return NaN statistics."""
    filename = traj_path.name
    print(f"\nProcessing: {filename}")

    # Load trajectory data
    trajectories = pd.read_csv(traj_path)

    # Get male_id from metadata
    male_id = get_male_id_from_metadata(metadata, filename)

    # Parse filename for metadata
    genotype, camera, datetime_str, setup = parse_filename(filename)

    # Get all fly IDs from columns
    fly_ids = []
    for col in trajectories.columns:
        if col.startswith('x'):
            fly_id = int(col[1:])
            fly_ids.append(fly_id)

    # Detect NaN locations for each fly
    nan_stats_list = []
    nan_states = {}

    for fly_id in fly_ids:
        x_col = f'x{fly_id}'
        y_col = f'y{fly_id}'

        # A fly has NaN location if either x or y is NaN
        has_nan = trajectories[x_col].isna() | trajectories[y_col].isna()
        nan_states[fly_id] = has_nan.values

        nan_count = has_nan.sum()
        total_frames = len(trajectories)
        nan_percentage = (nan_count / total_frames) * 100

        # Calculate time statistics
        recording_duration = trajectories['time'].iloc[-1]
        frame_rate = (len(trajectories) - 1) / recording_duration
        nan_time_seconds = nan_count / frame_rate if frame_rate > 0 else 0
        nan_time_minutes = nan_time_seconds / 60

        is_male = (fly_id == male_id) if male_id else False

        nan_stats_list.append({
            'dataset': dataset_name,
            'genotype': genotype,
            'camera': camera,
            'datetime': datetime_str,
            'setup': setup if setup else 'N/A',
            'filename': filename,
            'fly_id': fly_id,
            'is_male': is_male,
            'total_frames': total_frames,
            'nan_frames': nan_count,
            'valid_frames': total_frames - nan_count,
            'nan_percentage': nan_percentage,
            'recording_duration_sec': recording_duration,
            'nan_time_sec': nan_time_seconds,
            'nan_time_min': nan_time_minutes,
            'frame_rate': frame_rate
        })

        fly_type = "MALE" if is_male else "Female"
        print(f"  Fly {fly_id} ({fly_type}): {nan_count} NaN frames ({nan_percentage:.2f}%), {nan_time_minutes:.2f} min")

    # Create plot
    create_nan_plot(trajectories, fly_ids, nan_states, male_id,
                   genotype, camera, datetime_str, dataset_name, filename, setup)

    return nan_stats_list


def create_nan_plot(trajectories, fly_ids, nan_states, male_id,
                   genotype, camera, datetime_str, dataset_name, filename, setup):
    """Create and save a NaN visualization plot."""
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

    # Create title with genotype and dataset info
    setup_str = f" - {setup}" if setup else ""
    title = f'NaN Locations: {genotype} | {camera} | {datetime_str}{setup_str}\n[{dataset_name}]'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_yticks(list(y_positions.values()))

    # Create labels with red color for male
    y_labels = []
    for fid in fly_ids:
        if male_id and fid == male_id:
            y_labels.append(f'Fly {fid} (M)')
        else:
            y_labels.append(f'Fly {fid}')

    ax.set_yticklabels(y_labels)

    # Color the y-tick labels red for male
    for i, (fid, label) in enumerate(zip(fly_ids, ax.get_yticklabels())):
        if male_id and fid == male_id:
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

    # Save plot with informative filename
    output_filename = f"{dataset_name}_{genotype}_{camera}_{datetime_str}"
    if setup:
        output_filename += f"_{setup}"
    output_filename += "_nan_plot.png"

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def calculate_summary_stats(all_stats_df):
    """Calculate summary statistics for the CSV output."""
    # Add calculated columns
    summary_rows = []

    # Group by run (unique combination of dataset, genotype, camera, datetime, setup)
    run_groups = all_stats_df.groupby(['dataset', 'genotype', 'camera', 'datetime', 'setup', 'filename'])

    for (dataset, genotype, camera, datetime, setup, filename), group in run_groups:
        # Calculate statistics for this run
        num_flies = len(group)
        num_males = group['is_male'].sum()

        # Percentage of rows with NaN (at least one fly has NaN)
        # For this, we need to know if any fly had NaN at each time point
        # We approximate this as the maximum individual fly NaN percentage
        max_nan_pct = group['nan_percentage'].max()

        # Mean percentage of NaN time across flies
        mean_nan_pct = group['nan_percentage'].mean()

        # Mean minutes of NaN time
        mean_nan_min = group['nan_time_min'].mean()

        # Total NaN frames across all flies
        total_nan_frames = group['nan_frames'].sum()

        summary_rows.append({
            'dataset': dataset,
            'genotype': genotype,
            'camera': camera,
            'datetime': datetime,
            'setup': setup,
            'filename': filename,
            'num_flies': num_flies,
            'num_males': num_males,
            'total_nan_frames': total_nan_frames,
            'max_nan_percentage': max_nan_pct,
            'mean_nan_percentage': mean_nan_pct,
            'mean_nan_minutes': mean_nan_min,
            'recording_duration_sec': group['recording_duration_sec'].iloc[0],
            'frame_rate': group['frame_rate'].iloc[0]
        })

    return pd.DataFrame(summary_rows)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("="*80)
    print("PROCESSING ALL TRAJECTORY FILES FOR NaN ANALYSIS")
    print("="*80)

    all_stats = []

    # Process both datasets
    for dataset_name in ['dataset_1', 'dataset_2']:
        dataset_path = DATA_DIR / dataset_name
        traj_dir = dataset_path / 'trajectories'

        if not traj_dir.exists():
            print(f"\nWarning: {traj_dir} does not exist, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING {dataset_name.upper()}")
        print(f"{'='*80}")

        # Load metadata
        metadata = load_metadata(dataset_name)

        # Get all trajectory files
        traj_files = sorted(traj_dir.glob('*_trajectories.csv'))
        print(f"Found {len(traj_files)} trajectory files")

        # Process each file
        for traj_path in traj_files:
            stats = process_trajectory_file(traj_path, dataset_name, metadata)
            all_stats.extend(stats)

    # ========================================================================
    # SAVE DETAILED STATISTICS
    # ========================================================================

    print(f"\n{'='*80}")
    print("SAVING STATISTICS")
    print(f"{'='*80}")

    # Create detailed DataFrame
    all_stats_df = pd.DataFrame(all_stats)

    # Save detailed per-fly statistics
    detailed_output = OUTPUT_DIR / 'nan_statistics_per_fly.csv'
    all_stats_df.to_csv(detailed_output, index=False)
    print(f"\nDetailed per-fly statistics saved to: {detailed_output}")

    # Calculate and save summary statistics per run
    summary_df = calculate_summary_stats(all_stats_df)
    summary_output = OUTPUT_DIR / 'nan_statistics_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"Summary statistics per run saved to: {summary_output}")

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal trajectory files processed: {len(summary_df)}")
    print(f"Total flies analyzed: {len(all_stats_df)}")
    print(f"Total males: {all_stats_df['is_male'].sum()}")
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print(f"Statistics saved to: {OUTPUT_DIR}")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
