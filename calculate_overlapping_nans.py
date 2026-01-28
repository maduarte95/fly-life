import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PARAMETERS
# ============================================================================

DATA_DIR = Path('data/1M')
OUTPUT_DIR = Path('output/nangrams')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_trajectory_file_for_overlaps(traj_path, dataset_name):
    """Calculate time when at least 2 flies have missing data simultaneously."""
    filename = traj_path.name

    # Load trajectory data
    trajectories = pd.read_csv(traj_path)

    # Get all fly IDs from columns
    fly_ids = []
    for col in trajectories.columns:
        if col.startswith('x'):
            fly_id = int(col[1:])
            fly_ids.append(fly_id)

    # Create a matrix: rows = frames, columns = flies, values = has_nan (boolean)
    nan_matrix = np.zeros((len(trajectories), len(fly_ids)), dtype=bool)

    for i, fly_id in enumerate(fly_ids):
        x_col = f'x{fly_id}'
        y_col = f'y{fly_id}'
        has_nan = trajectories[x_col].isna() | trajectories[y_col].isna()
        nan_matrix[:, i] = has_nan.values

    # Count how many flies have NaN at each time point
    num_nans_per_frame = nan_matrix.sum(axis=1)

    # Count frames where at least 2 flies have NaN
    frames_with_2plus_nans = (num_nans_per_frame >= 2).sum()

    # Calculate time statistics
    recording_duration = trajectories['time'].iloc[-1]
    frame_rate = (len(trajectories) - 1) / recording_duration if recording_duration > 0 else 0
    time_with_2plus_nans_sec = frames_with_2plus_nans / frame_rate if frame_rate > 0 else 0
    time_with_2plus_nans_min = time_with_2plus_nans_sec / 60

    # Calculate percentage
    total_frames = len(trajectories)
    percentage_2plus_nans = (frames_with_2plus_nans / total_frames) * 100 if total_frames > 0 else 0

    print(f"  {filename}")
    print(f"    Frames with ≥2 flies missing: {frames_with_2plus_nans}/{total_frames} ({percentage_2plus_nans:.2f}%)")
    print(f"    Time with ≥2 flies missing: {time_with_2plus_nans_min:.2f} minutes")

    return {
        'dataset': dataset_name,
        'filename': filename,
        'total_frames': total_frames,
        'frames_with_2plus_nans': frames_with_2plus_nans,
        'percentage_2plus_nans': percentage_2plus_nans,
        'recording_duration_sec': recording_duration,
        'time_with_2plus_nans_sec': time_with_2plus_nans_sec,
        'time_with_2plus_nans_min': time_with_2plus_nans_min,
        'frame_rate': frame_rate
    }

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("="*80)
    print("CALCULATING OVERLAPPING NaN TIME (≥2 flies)")
    print("="*80)

    all_stats = []

    # Process both datasets
    for dataset_name in ['dataset_1', 'dataset_2']:
        dataset_path = DATA_DIR / dataset_name
        traj_dir = dataset_path / 'trajectories'

        if not traj_dir.exists():
            print(f"\nWarning: {traj_dir} does not exist, skipping...")
            continue

        print(f"\n{dataset_name.upper()}:")
        print("-" * 80)

        # Get all trajectory files
        traj_files = sorted(traj_dir.glob('*_trajectories.csv'))

        # Process each file
        for traj_path in traj_files:
            stats = process_trajectory_file_for_overlaps(traj_path, dataset_name)
            all_stats.append(stats)

    # Save statistics
    output_df = pd.DataFrame(all_stats)
    output_path = OUTPUT_DIR / 'nan_overlapping_stats.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Statistics saved to: {output_path}")
    print(f"{'='*80}")

    # Print summary
    print(f"\nSUMMARY:")
    for dataset_name in ['dataset_1', 'dataset_2']:
        data = output_df[output_df['dataset'] == dataset_name]['time_with_2plus_nans_min']
        print(f"\n{dataset_name.upper()}:")
        print(f"  Mean time with ≥2 flies missing: {data.mean():.2f} minutes")
        print(f"  Median time with ≥2 flies missing: {data.median():.2f} minutes")
        print(f"  Max time with ≥2 flies missing: {data.max():.2f} minutes")

if __name__ == '__main__':
    main()
