"""
Batch ethogram processing for multiple 5F1M sessions.

This script searches the `data/trajectories/0_trajectories` folder for matching
trajectory and area CSV files, processes each session (distance-based pursuit
and copulation detection plus disappearance-based copulation), aggregates
statistics across sessions and females, and produces combined ethogram and
summary plots.

Usage: run without arguments in the project root. Outputs are saved to
`output/`.
"""
import re
import math
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETERS (tweak as needed)
DATA_DIR = Path('data/trajectories/0_trajectories')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Per-camera conversion (pixels per cm)
PIXELS_PER_CM_CAMA = 208  # CamA: 1 cm = 208 pixels
PIXELS_PER_CM_CAMB = 203  # CamB: 1 cm = 203 pixels

# Behavior thresholds (same defaults as ethogram.py)
# default fallback if camera not recognized
PIXELS_PER_CM = PIXELS_PER_CM_CAMA
COPULATION_DISTANCE_CM = 0.3
COPULATION_MIN_DURATION_S = 120
COPULATION_USE_ID_DISAPPEARANCE = True
COPULATION_DISAPPEARANCE_MIN_DURATION_S = 120
PURSUIT_DISTANCE_CM = 1.0
PURSUIT_MIN_DURATION_S = 2.0

# Plotting
PLOT_TIME_UNIT = 'minutes'  # or 'seconds'


def find_sessions(data_dir: Path):
    """Find candidate sessions grouped by camera and timestamp in filenames.

    Returns a list of dicts: { 'camera': 'CamA'|'CamB', 'timestamp': str, 'trajectories': [paths...], 'areas': path_or_None }
    """
    tra_files = list(data_dir.glob('*_trajectories.csv'))
    area_files = {p.name: p for p in data_dir.glob('*_areas.csv')}

    sessions = {}

    # regex to extract camera (CamA/CamB) and timestamp chunk like 2026-01-12T08_31_36
    ts_re = re.compile(r'_(Cam[AB])Video\d+_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_')

    for p in tra_files:
        m = ts_re.search(p.name)
        if m:
            cam = m.group(1)
            ts = m.group(2)
        else:
            # fallback heuristics
            cam = 'CamA' if 'CamA' in p.name else ('CamB' if 'CamB' in p.name else None)
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', p.name)
            if not cam or not ts_match:
                continue
            ts = ts_match.group(1)

        key = (cam, ts)
        sessions.setdefault(key, {'trajectories': [], 'areas': None, 'camera': cam, 'timestamp': ts})
        sessions[key]['trajectories'].append(p)

    # match area files by camera+timestamp presence in name
    for (cam, ts), info in sessions.items():
        for name, p in area_files.items():
            if cam in name and ts in name:
                info['areas'] = p
                break

    # filter and build session list
    session_list = []
    for (cam, ts), info in sessions.items():
        if len(info['trajectories']) >= 1:
            info['trajectories'] = sorted(info['trajectories'], key=lambda x: x.name)
            session_list.append({'camera': cam, 'timestamp': ts, 'trajectories': info['trajectories'], 'areas': info['areas']})

    # sort by camera then timestamp for deterministic order
    session_list = sorted(session_list, key=lambda x: (x['camera'], x['timestamp']))
    return session_list


def detect_behaviors(trajectories_paths, areas_path, pixels_per_cm=PIXELS_PER_CM,
                     copulation_distance_cm=COPULATION_DISTANCE_CM,
                     copulation_min_duration_s=COPULATION_MIN_DURATION_S,
                     pursuit_distance_cm=PURSUIT_DISTANCE_CM,
                     pursuit_min_duration_s=PURSUIT_MIN_DURATION_S,
                     copulation_use_disappearance=COPULATION_USE_ID_DISAPPEARANCE,
                     copulation_disappearance_min_duration_s=COPULATION_DISAPPEARANCE_MIN_DURATION_S):
    """Process a single session: return dict with time array, pursuit_df, copulation_df, and summary stats."""
    # Load and concatenate trajectories
    trajs = [pd.read_csv(p) for p in trajectories_paths]
    # If there are multiple parts, offset subsequent parts' time
    if len(trajs) > 1:
        video1_duration = trajs[0]['time'].iloc[-1]
        for i in range(1, len(trajs)):
            trajs[i]['time'] = trajs[i]['time'] + video1_duration

    trajectories = pd.concat(trajs, ignore_index=True)

    # basic frame rate estimation
    recording_time = trajectories['time'].iloc[-1]
    frame_rate = len(trajectories) / recording_time if recording_time > 0 else 30.0

    # identify male from areas file (smallest mean)
    areas = pd.read_csv(areas_path) if areas_path is not None else None
    if areas is not None:
        male_index_areas = areas['mean'].idxmin()
        male_id_trajectories = male_index_areas + 1
    else:
        # fallback: assume ID 1 is male
        male_id_trajectories = 1

    male_x = trajectories[f'x{male_id_trajectories}'].values
    male_y = trajectories[f'y{male_id_trajectories}'].values

    all_fly_ids = range(1, 7)
    female_ids = [fid for fid in all_fly_ids if fid != male_id_trajectories]

    # compute distance to each female
    distances_cm = {}
    for fid in female_ids:
        fx = trajectories[f'x{fid}'].values
        fy = trajectories[f'y{fid}'].values
        dist_px = np.sqrt((male_x - fx) ** 2 + (male_y - fy) ** 2)
        distances_cm[fid] = dist_px / pixels_per_cm

    distances_df = pd.DataFrame(distances_cm)
    distances_df['time'] = trajectories['time'].values

    # thresholds in frames
    pursuit_frames_threshold = int(pursuit_min_duration_s * frame_rate)
    copulation_frames_threshold = int(copulation_min_duration_s * frame_rate)

    pursuit_states = {}
    copulation_states = {}

    # detect pursuit and copulation by contiguous-run method
    for fid in female_ids:
        within_pursuit = (distances_df[fid] < pursuit_distance_cm).astype(int)
        within_cop = (distances_df[fid] < copulation_distance_cm).astype(int)

        # reusable run-length detection
        def runs_to_bool_arr(within_arr, frames_threshold):
            out = np.zeros(len(within_arr), dtype=bool)
            in_run = False
            start_idx = 0
            for i in range(len(within_arr)):
                if within_arr.iloc[i] == 1 and not in_run:
                    in_run = True
                    start_idx = i
                elif within_arr.iloc[i] == 0 and in_run:
                    duration = i - start_idx
                    if duration >= frames_threshold:
                        out[start_idx:i] = True
                    in_run = False
            if in_run:
                duration = len(within_arr) - start_idx
                if duration >= frames_threshold:
                    out[start_idx:] = True
            return out

        pursuit_states[fid] = runs_to_bool_arr(within_pursuit, pursuit_frames_threshold)
        copulation_states[fid] = runs_to_bool_arr(within_cop, copulation_frames_threshold)

    pursuit_df = pd.DataFrame(pursuit_states)
    pursuit_df['time'] = trajectories['time'].values
    copulation_df = pd.DataFrame(copulation_states)
    copulation_df['time'] = trajectories['time'].values

    # disappearance-based copulation: male missing
    if copulation_use_disappearance:
        disappearance_frames_threshold = int(copulation_disappearance_min_duration_s * frame_rate)
        male_missing = np.isnan(male_x)
        in_missing = False
        start_idx = 0
        disappearance_events = []
        for i in range(len(male_missing)):
            if male_missing[i] and not in_missing:
                in_missing = True
                start_idx = i
            elif not male_missing[i] and in_missing:
                duration = i - start_idx
                if duration >= disappearance_frames_threshold:
                    disappearance_events.append((start_idx, i))
                in_missing = False
        if in_missing:
            duration = len(male_missing) - start_idx
            if duration >= disappearance_frames_threshold:
                disappearance_events.append((start_idx, len(male_missing)))

        # for each event, find closest female before disappearance
        lookback_frames = int(5 * frame_rate)
        for (start, end) in disappearance_events:
            lookback_start = max(0, start - lookback_frames)
            avg_distances = {}
            for fid in female_ids:
                d_before = distances_df[fid].iloc[lookback_start:start]
                valid = d_before.dropna()
                avg_distances[fid] = valid.mean() if len(valid) > 0 else np.inf
            closest = min(avg_distances, key=avg_distances.get)
            copulation_df.loc[start:end-1, closest] = True

    # compute summary stats per session (aggregated across females and per female)
    stats = {'frame_rate': frame_rate, 'n_frames': len(trajectories), 'duration_s': recording_time}
    per_female_stats = {}
    for fid in female_ids:
        pf = {}
        pf['total_copulation_frames'] = int(copulation_df[fid].sum())
        pf['total_copulation_s'] = pf['total_copulation_frames'] / frame_rate
        pf['total_pursuit_frames'] = int(pursuit_df[fid].sum())
        pf['total_pursuit_s'] = pf['total_pursuit_frames'] / frame_rate

        # number of copulation events by counting rising edges of copulation boolean
        cop_bool = copulation_df[fid].values.astype(bool)
        if cop_bool.sum() > 0:
            edges = np.diff(np.concatenate([[0], cop_bool.astype(int)]))
            pf['n_copulation_events'] = int((edges == 1).sum())
            first_idx = np.where(cop_bool)[0][0]
            pf['time_first_copulation_s'] = float(trajectories['time'].iloc[first_idx])
            # longest copulation run
            runs = _compute_runs(cop_bool)
            pf['longest_copulation_s'] = max((r[1] for r in runs), default=0) / frame_rate
        else:
            pf['n_copulation_events'] = 0
            pf['time_first_copulation_s'] = np.nan
            pf['longest_copulation_s'] = 0.0

        # pursuit runs
        pur_bool = pursuit_df[fid].values.astype(bool)
        if pur_bool.sum() > 0:
            runs = _compute_runs(pur_bool)
            pf['n_pursuit_events'] = int(sum(1 for r in runs))
            pf['longest_pursuit_s'] = max((r[1] for r in runs), default=0) / frame_rate
            # first persistent (>5min) pursuit
            persistent_frames = int(5 * 60 * frame_rate)
            persistent_run = next((r for r in runs if r[1] >= persistent_frames), None)
            if persistent_run is not None:
                pf['time_first_persistent_pursuit_s'] = float(trajectories['time'].iloc[persistent_run[0]])
            else:
                pf['time_first_persistent_pursuit_s'] = np.nan
        else:
            pf['n_pursuit_events'] = 0
            pf['longest_pursuit_s'] = 0.0
            pf['time_first_persistent_pursuit_s'] = np.nan

        per_female_stats[fid] = pf

    return {
        'trajectories': trajectories,
        'frame_rate': frame_rate,
        'female_ids': female_ids,
        'pursuit_df': pursuit_df,
        'copulation_df': copulation_df,
        'per_female_stats': per_female_stats,
        'stats': stats,
    }


def _compute_runs(bool_arr):
    """Return list of (start_idx, length) for contiguous True runs in bool_arr."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(bool_arr):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            runs.append((start, i - start))
            in_run = False
    if in_run:
        runs.append((start, len(bool_arr) - start))
    return runs


def aggregate_sessions(session_results):
    """Create a table of summary statistics per session (aggregated across females)."""
    rows = []
    for sess in session_results:
        ts = sess['timestamp']
        cam = sess.get('camera', 'CamA')
        session_key = f"{cam}_{ts}"
        per_female = sess['result']['per_female_stats']
        agg = {'timestamp': ts, 'camera': cam, 'session_key': session_key}
        # totals across females
        agg['total_copulation_s'] = sum(pf['total_copulation_s'] for pf in per_female.values())
        agg['total_pursuit_s'] = sum(pf['total_pursuit_s'] for pf in per_female.values())
        agg['n_copulation_events'] = sum(pf['n_copulation_events'] for pf in per_female.values())
        agg['n_females_copulated'] = sum(1 for pf in per_female.values() if pf['n_copulation_events'] > 0)
        agg['longest_copulation_s'] = max((pf['longest_copulation_s'] for pf in per_female.values()), default=0)
        agg['longest_pursuit_s'] = max((pf['longest_pursuit_s'] for pf in per_female.values()), default=0)
        # time to first copulation across females
        times_first = [pf['time_first_copulation_s'] for pf in per_female.values() if not pd.isna(pf['time_first_copulation_s'])]
        agg['time_first_copulation_s'] = min(times_first) if times_first else np.nan
        # time to first persistent pursuit (>5min) across females
        times_persistent = [pf['time_first_persistent_pursuit_s'] for pf in per_female.values() if not pd.isna(pf['time_first_persistent_pursuit_s'])]
        agg['time_first_persistent_pursuit_s'] = min(times_persistent) if times_persistent else np.nan
        # longest of pursuit or copulation
        agg['longest_any_s'] = max(agg['longest_copulation_s'], agg['longest_pursuit_s'])
        rows.append(agg)
    return pd.DataFrame(rows)


def plot_summary_dotplots(agg_df, session_results):
    """Plot requested metrics as aligned dot-plots with consistent session colors.

    Each session gets one color; colors are used across all plots.
    Rows represent sessions (same order as session_results), columns are metrics.
    """
    # Determine session order and color mapping
    sessions = [f"{s['camera']}_{s['timestamp']}" for s in session_results]
    n = len(sessions)
    palette = sns.color_palette('tab20', n_colors=n)
    color_map = {sess: palette[i] for i, sess in enumerate(sessions)}

    # Metrics to plot: (column_name, display_label, is_time_seconds)
    metrics = [
        ('total_copulation_s', 'Total copulation (min)', True),
        ('total_pursuit_s', 'Total pursuit (min)', True),
        ('longest_copulation_s', 'Longest copulation (min)', True),
        ('longest_pursuit_s', 'Longest pursuit (min)', True),
        ('longest_any_s', 'Longest any behavior (min)', True),
        ('n_copulation_events', 'Number of copulation events', False),
        ('n_females_copulated', 'Number of females copulated', False),
        ('time_first_copulation_s', 'Time to first copulation (min)', True),
        ('time_first_persistent_pursuit_s', 'Time to first persistent pursuit (min)', True),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, max(6, 0.4 * n)), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    # Build a lookup from session_key to agg row
    agg_df_indexed = agg_df.set_index('session_key')

    # Prepare legend handles for sessions labeled Session 1..N
    session_keys = sessions
    session_labels = [f"Session {i+1}" for i in range(n)]
    legend_handles = []
    for i, sk in enumerate(session_keys):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[sk], markersize=8, markeredgecolor='k', label=session_labels[i]))

    for col_idx, (col, label, is_time) in enumerate(metrics):
        ax = axes[col_idx]
        vals = []
        colors = []
        for i, s in enumerate(session_results):
            key = f"{s['camera']}_{s['timestamp']}"
            if key in agg_df_indexed.index:
                val = agg_df_indexed.loc[key, col]
            else:
                val = np.nan
            if pd.isna(val):
                vals.append(np.nan)
            else:
                vals.append(val / 60.0 if is_time else val)
            colors.append(color_map[key])

        # Convert to numpy and mask NaNs
        vals_arr = np.array(vals, dtype=float)
        mask = ~np.isnan(vals_arr)
        vals_masked = vals_arr[mask]
        colors_masked = np.array(colors)[mask]

        # Violin or box for the distribution (draw beneath points)
        if len(vals_masked) > 0:
            # sns.violinplot(y=vals_masked, ax=ax, inner=None, color='lightgray', alpha=0.5, zorder=1)
            sns.boxplot(y=vals_masked, ax=ax, width=0.1, showcaps=True,
                        boxprops={'facecolor': 'white', 'zorder': 1}, showfliers=False,
                        whiskerprops={'zorder': 1}, capprops={'zorder': 1}, medianprops={'zorder': 2})

        # Overlay scatter for each session (jitter x slightly) with higher zorder
        jitter = np.random.normal(loc=0.0, scale=0.04, size=len(vals_masked))
        xs = np.zeros(len(vals_masked)) + jitter
        ax.scatter(xs, vals_masked, c=colors_masked, s=160, edgecolor='k', zorder=10)

        ax.set_ylabel(label)
        ax.set_xlabel('')
        # tidy up x-axis (no ticks)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add a legend with session labels on the last axis
    axes[-1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Sessions', fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / 'summary_dotplots_sessions.png'
    fig.savefig(out, dpi=250, bbox_inches='tight')
    print(f'Saved summary dot-plots to {out}')


def plot_sessions_ethograms(session_results, cols=4, figsize_per= (6, 3)):
    n = len(session_results)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0]*cols, figsize_per[1]*rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis('off')

    for i, sess in enumerate(session_results):
        ax = axes[i]
        res = sess['result']
        trajectories = res['trajectories']
        time_vals = trajectories['time'].values
        if PLOT_TIME_UNIT == 'minutes':
            t = time_vals / 60
            xlabel = 'Time (minutes)'
        else:
            t = time_vals
            xlabel = 'Time (seconds)'

        female_ids = res['female_ids']
        y_positions = {fid: j for j, fid in enumerate(female_ids)}
        colors = {'no_interaction': 'lightgray', 'pursuit': 'orange', 'copulation': 'red'}

        pursuit_df = res['pursuit_df']
        copulation_df = res['copulation_df']

        for fid in female_ids:
            y = y_positions[fid]
            behavior = np.full(len(trajectories), 0)
            behavior[pursuit_df[fid].values.astype(bool)] = 1
            behavior[copulation_df[fid].values.astype(bool)] = 2

            # compact spans
            i0 = 0
            while i0 < len(behavior):
                val = behavior[i0]
                j = i0
                while j + 1 < len(behavior) and behavior[j+1] == val:
                    j += 1
                start_t = t[i0]
                end_t = t[j]
                if val == 0:
                    color = colors['no_interaction']
                elif val == 1:
                    color = colors['pursuit']
                else:
                    color = colors['copulation']
                ax.fill_between([start_t, end_t], y - 0.4, y + 0.4, color=color, alpha=0.9, linewidth=0)
                i0 = j + 1

        ax.set_title(sess['timestamp'])
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([f'F{fid}' for fid in female_ids])
        ax.set_xlabel(xlabel)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ethograms_all_sessions.png'
    fig.savefig(out, dpi=250, bbox_inches='tight')
    print(f'Saved combined ethograms to {out}')


def plot_summary_violin(agg_df):
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # total copulation
    sns.violinplot(y=agg_df['total_copulation_s'], ax=axes[0], inner=None, color='lightcoral')
    sns.stripplot(y=agg_df['total_copulation_s'], ax=axes[0], color='black')
    axes[0].set_ylabel('Total copulation time (s)')
    axes[0].set_title('Total copulation time per session')

    sns.violinplot(y=agg_df['total_pursuit_s'], ax=axes[1], inner=None, color='orange')
    sns.stripplot(y=agg_df['total_pursuit_s'], ax=axes[1], color='black')
    axes[1].set_ylabel('Total pursuit time (s)')
    axes[1].set_title('Total pursuit time per session')

    plt.tight_layout()
    out = OUTPUT_DIR / 'summary_violin.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved summary violin plots to {out}')


def main():
    cache_file = OUTPUT_DIR / 'session_results.pkl'

    # If cache exists, load and skip raw processing to save time
    if cache_file.exists():
        print(f'Loading cached session results from {cache_file} (delete this file to reprocess raw CSVs)')
        with open(cache_file, 'rb') as f:
            session_results = pickle.load(f)
    else:
        sessions = find_sessions(DATA_DIR)
        print(f'Found {len(sessions)} candidate sessions')
        session_results = []
        for s in sessions:
            # choose trajectory parts and areas
            trajs = s['trajectories']
            areas = s['areas']
            cam = s.get('camera', 'CamA')
            print(f"Processing session {s['timestamp']} ({cam}): {len(trajs)} trajectory parts, areas: {areas}")
            # pick pixels per cm based on camera
            if cam == 'CamA':
                pixels_per_cm = PIXELS_PER_CM_CAMA
            elif cam == 'CamB':
                pixels_per_cm = PIXELS_PER_CM_CAMB
            else:
                pixels_per_cm = PIXELS_PER_CM
            try:
                result = detect_behaviors(trajs, areas, pixels_per_cm=pixels_per_cm)
            except Exception as e:
                print(f"  Skipped session {s['timestamp']} due to error: {e}")
                continue
            session_results.append({'camera': cam, 'timestamp': s['timestamp'], 'result': result})

        # Save cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(session_results, f)
            print(f'Saved cached session results to {cache_file}')
        except Exception as e:
            print(f'Warning: could not save cache to {cache_file}: {e}')

    if not session_results:
        print('No sessions processed, exiting')
        return

    # aggregate
    agg_df = aggregate_sessions(session_results)
    agg_out = OUTPUT_DIR / 'aggregated_session_stats.csv'
    agg_df.to_csv(agg_out, index=False)
    print(f'Saved aggregated stats to {agg_out}')

    # plots
    plot_sessions_ethograms(session_results, cols=4)
    plot_summary_dotplots(agg_df, session_results)


if __name__ == '__main__':
    main()
# Ethogram for Observing Animal Behavior
