"""
Batch ethogram processing for 2 males + 5 females sessions.

This script mirrors the single-male batch processor but supports sessions
where the `areas` file contains 7 rows (2 males + 5 females) and the
trajectories have two male columns (x/y for each male). It finds sessions by
camera+timestamp, processes distances from each male to each female, detects
pursuit and copulation (including disappearance-based detection per male),
aggregates stats, and writes plots and a cache file so you don't reprocess
raw CSVs on every change.

Usage: run from the project root. Outputs go to `output_2M/`.
"""
import re
import math
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETERS
DATA_DIR = Path('data/trajectories/0_trajectories')
OUTPUT_DIR = Path('output_2M')
OUTPUT_DIR.mkdir(exist_ok=True)

# Per-camera conversion
PIXELS_PER_CM_CAMA = 208
PIXELS_PER_CM_CAMB = 203

# Behavior thresholds
COPULATION_DISTANCE_CM = 0.3
COPULATION_MIN_DURATION_S = 120
COPULATION_USE_ID_DISAPPEARANCE = True
COPULATION_DISAPPEARANCE_MIN_DURATION_S = 120
PURSUIT_DISTANCE_CM = 1.0
PURSUIT_MIN_DURATION_S = 2.0

# Plotting
PLOT_TIME_UNIT = 'minutes'


def find_sessions(data_dir: Path):
    tra_files = list(data_dir.glob('*_trajectories.csv'))
    area_files = {p.name: p for p in data_dir.glob('*_areas.csv')}

    sessions = {}
    ts_re = re.compile(r'_(Cam[AB])Video\d+_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_')

    for p in tra_files:
        m = ts_re.search(p.name)
        if m:
            cam = m.group(1)
            ts = m.group(2)
        else:
            cam = 'CamA' if 'CamA' in p.name else ('CamB' if 'CamB' in p.name else None)
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', p.name)
            if not cam or not ts_match:
                continue
            ts = ts_match.group(1)

        key = (cam, ts)
        sessions.setdefault(key, {'trajectories': [], 'areas': None, 'camera': cam, 'timestamp': ts})
        sessions[key]['trajectories'].append(p)

    for (cam, ts), info in sessions.items():
        for name, p in area_files.items():
            if cam in name and ts in name:
                info['areas'] = p
                break

    session_list = []
    for (cam, ts), info in sessions.items():
        if len(info['trajectories']) >= 1:
            info['trajectories'] = sorted(info['trajectories'], key=lambda x: x.name)
            session_list.append({'camera': cam, 'timestamp': ts, 'trajectories': info['trajectories'], 'areas': info['areas']})

    session_list = sorted(session_list, key=lambda x: (x['camera'], x['timestamp']))
    return session_list


def _compute_runs(bool_arr):
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


def detect_behaviors_2m(trajectories_paths, areas_path, pixels_per_cm=PIXELS_PER_CM_CAMA,
                       copulation_distance_cm=COPULATION_DISTANCE_CM,
                       copulation_min_duration_s=COPULATION_MIN_DURATION_S,
                       pursuit_distance_cm=PURSUIT_DISTANCE_CM,
                       pursuit_min_duration_s=PURSUIT_MIN_DURATION_S,
                       copulation_use_disappearance=COPULATION_USE_ID_DISAPPEARANCE,
                       copulation_disappearance_min_duration_s=COPULATION_DISAPPEARANCE_MIN_DURATION_S):
    # load and concatenate
    trajs = [pd.read_csv(p) for p in trajectories_paths]
    if len(trajs) > 1:
        video1_duration = trajs[0]['time'].iloc[-1]
        for i in range(1, len(trajs)):
            trajs[i]['time'] = trajs[i]['time'] + video1_duration
    trajectories = pd.concat(trajs, ignore_index=True)

    recording_time = trajectories['time'].iloc[-1]
    frame_rate = len(trajectories) / recording_time if recording_time > 0 else 30.0

    areas = pd.read_csv(areas_path) if areas_path is not None else None
    if areas is not None and len(areas) >= 7:
        # pick two smallest mean areas as males
        male_indices = areas['mean'].nsmallest(2).index.tolist()
        male_ids = [idx + 1 for idx in male_indices]
    else:
        # fallback: assume IDs 1 and 2 are males
        male_ids = [1, 2]

    all_ids = list(range(1, 8))  # 1..7 (2 males + 5 females)
    female_ids = [fid for fid in all_ids if fid not in male_ids]

    # extract male coordinates
    male_coords = {}
    for m in male_ids:
        male_coords[m] = (trajectories[f'x{m}'].values, trajectories[f'y{m}'].values)

    # distances per male->female
    distances = {m: {} for m in male_ids}
    for m in male_ids:
        mx, my = male_coords[m]
        for f in female_ids:
            fx = trajectories[f'x{f}'].values
            fy = trajectories[f'y{f}'].values
            dist_px = np.sqrt((mx - fx) ** 2 + (my - fy) ** 2)
            distances[m][f] = dist_px / pixels_per_cm

    # detection thresholds in frames
    pursuit_frames_threshold = int(pursuit_min_duration_s * frame_rate)
    copulation_frames_threshold = int(copulation_min_duration_s * frame_rate)

    # Store boolean arrays per male per female
    pursuit = {m: {} for m in male_ids}
    copulation = {m: {} for m in male_ids}

    for m in male_ids:
        for f in female_ids:
            within_pursuit = (pd.Series(distances[m][f]) < pursuit_distance_cm).astype(int)
            within_cop = (pd.Series(distances[m][f]) < copulation_distance_cm).astype(int)

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

            pursuit[m][f] = runs_to_bool_arr(within_pursuit, pursuit_frames_threshold)
            copulation[m][f] = runs_to_bool_arr(within_cop, copulation_frames_threshold)

    # disappearance-based detection per male
    if copulation_use_disappearance:
        disappearance_frames_threshold = int(copulation_disappearance_min_duration_s * frame_rate)
        for m in male_ids:
            mx, _ = male_coords[m]
            male_missing = np.isnan(mx)
            in_missing = False
            start_idx = 0
            events = []
            for i in range(len(male_missing)):
                if male_missing[i] and not in_missing:
                    in_missing = True
                    start_idx = i
                elif not male_missing[i] and in_missing:
                    duration = i - start_idx
                    if duration >= disappearance_frames_threshold:
                        events.append((start_idx, i))
                    in_missing = False
            if in_missing:
                duration = len(male_missing) - start_idx
                if duration >= disappearance_frames_threshold:
                    events.append((start_idx, len(male_missing)))

            lookback_frames = int(5 * frame_rate)
            for (start, end) in events:
                lookback_start = max(0, start - lookback_frames)
                avg_dist = {}
                for f in female_ids:
                    d_before = pd.Series(distances[m][f]).iloc[lookback_start:start].dropna()
                    avg_dist[f] = d_before.mean() if len(d_before) > 0 else np.inf
                closest = min(avg_dist, key=avg_dist.get)
                copulation[m][closest][start:end] = True

    # Build DataFrames similar to 1M: for convenience, create per-male DataFrames
    pursuit_dfs = {m: pd.DataFrame(pursuit[m]) for m in male_ids}
    copulation_dfs = {m: pd.DataFrame(copulation[m]) for m in male_ids}
    for m in male_ids:
        pursuit_dfs[m]['time'] = trajectories['time'].values
        copulation_dfs[m]['time'] = trajectories['time'].values

    # Aggregated session-level stats (combine both males)
    per_female_stats = {}
    for f in female_ids:
        pf = {}
        # total copulation/pursuit across both males
        cop_frames = sum(int(copulation_dfs[m][f].sum()) for m in male_ids)
        pur_frames = sum(int(pursuit_dfs[m][f].sum()) for m in male_ids)
        pf['total_copulation_frames'] = cop_frames
        pf['total_copulation_s'] = cop_frames / frame_rate
        pf['total_pursuit_frames'] = pur_frames
        pf['total_pursuit_s'] = pur_frames / frame_rate

        # number of copulation events (count edges across males)
        n_events = 0
        first_times = []
        longest_cop = 0
        longest_pur = 0
        for m in male_ids:
            cop_bool = copulation_dfs[m][f].values.astype(bool)
            if cop_bool.sum() > 0:
                edges = np.diff(np.concatenate([[0], cop_bool.astype(int)]))
                n_events += int((edges == 1).sum())
                first_times.append(float(trajectories['time'].iloc[np.where(cop_bool)[0][0]]))
                runs = _compute_runs(cop_bool)
                if runs:
                    longest_cop = max(longest_cop, max(r[1] for r in runs) / frame_rate)

            pur_bool = pursuit_dfs[m][f].values.astype(bool)
            if pur_bool.sum() > 0:
                runs = _compute_runs(pur_bool)
                if runs:
                    longest_pur = max(longest_pur, max(r[1] for r in runs) / frame_rate)
        pf['n_copulation_events'] = n_events
        pf['time_first_copulation_s'] = min(first_times) if first_times else np.nan
        pf['longest_copulation_s'] = longest_cop
        pf['longest_pursuit_s'] = longest_pur

        # first persistent pursuit (>5 min) across males
        persistent_frames = int(5 * 60 * frame_rate)
        persistent_times = []
        for m in male_ids:
            pur_bool = pursuit_dfs[m][f].values.astype(bool)
            runs = _compute_runs(pur_bool)
            pr = next((r for r in runs if r[1] >= persistent_frames), None)
            if pr is not None:
                persistent_times.append(float(trajectories['time'].iloc[pr[0]]))
        pf['time_first_persistent_pursuit_s'] = min(persistent_times) if persistent_times else np.nan

        per_female_stats[f] = pf

    # Session-level aggregation
    stats = {'frame_rate': frame_rate, 'n_frames': len(trajectories), 'duration_s': recording_time}
    session_agg = {
        'total_copulation_s': sum(pf['total_copulation_s'] for pf in per_female_stats.values()),
        'total_pursuit_s': sum(pf['total_pursuit_s'] for pf in per_female_stats.values()),
        'n_copulation_events': sum(pf['n_copulation_events'] for pf in per_female_stats.values()),
        'n_females_copulated': sum(1 for pf in per_female_stats.values() if pf['n_copulation_events'] > 0),
        'longest_copulation_s': max((pf['longest_copulation_s'] for pf in per_female_stats.values()), default=0),
        'longest_pursuit_s': max((pf['longest_pursuit_s'] for pf in per_female_stats.values()), default=0),
        'time_first_copulation_s': min((pf['time_first_copulation_s'] for pf in per_female_stats.values() if not pd.isna(pf['time_first_copulation_s'])), default=np.nan),
        'time_first_persistent_pursuit_s': min((pf['time_first_persistent_pursuit_s'] for pf in per_female_stats.values() if not pd.isna(pf['time_first_persistent_pursuit_s'])), default=np.nan),
    }
    session_agg['longest_any_s'] = max(session_agg['longest_copulation_s'], session_agg['longest_pursuit_s'])

    return {
        'trajectories': trajectories,
        'frame_rate': frame_rate,
        'male_ids': male_ids,
        'female_ids': female_ids,
        'pursuit_dfs': pursuit_dfs,
        'copulation_dfs': copulation_dfs,
        'per_female_stats': per_female_stats,
        'session_agg': session_agg,
        'stats': stats,
    }


def aggregate_sessions_2m(session_results):
    rows = []
    for sess in session_results:
        cam = sess.get('camera', 'CamA')
        ts = sess['timestamp']
        key = f"{cam}_{ts}"
        agg = {'session_key': key, 'camera': cam, 'timestamp': ts}
        res = sess['result']
        pa = res['session_agg']
        agg.update(pa)
        rows.append(agg)
    return pd.DataFrame(rows)


def plot_sessions_ethograms_2m(session_results, cols=4, figsize_per=(6, 3)):
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
        male_ids = res['male_ids']
        y_positions = {fid: j for j, fid in enumerate(female_ids)}

        colors = {
            'no_interaction': 'lightgray',
            'pursuit_m1': '#FFA500',
            'copulation_m1': '#FF4500',
            'pursuit_m2': '#87CEFA',
            'copulation_m2': '#1E90FF'
        }

        # Build behavior per female: 0 none, 1 pursuit m1, 2 copulation m1, 3 pursuit m2, 4 copulation m2
        for fid in female_ids:
            y = y_positions[fid]
            behavior = np.zeros(len(trajectories), dtype=int)
            # male1
            m1 = male_ids[0]
            m2 = male_ids[1]
            if fid in res['pursuit_dfs'][m1].columns:
                behavior[res['pursuit_dfs'][m1][fid].values.astype(bool)] = 1
            if fid in res['copulation_dfs'][m1].columns:
                behavior[res['copulation_dfs'][m1][fid].values.astype(bool)] = 2
            # male2
            if fid in res['pursuit_dfs'][m2].columns:
                # only set if not already copulation by m1
                mask = res['pursuit_dfs'][m2][fid].values.astype(bool) & (behavior == 0)
                behavior[mask] = 3
            if fid in res['copulation_dfs'][m2].columns:
                mask = res['copulation_dfs'][m2][fid].values.astype(bool) & (behavior < 3)
                # If m1 copulation present, keep m1 copulation; otherwise mark m2
                behavior[mask] = 4

            # Plot spans
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
                    color = colors['pursuit_m1']
                elif val == 2:
                    color = colors['copulation_m1']
                elif val == 3:
                    color = colors['pursuit_m2']
                else:
                    color = colors['copulation_m2']

                ax.fill_between([start_t, end_t], y - 0.4, y + 0.4, color=color, alpha=0.9, linewidth=0)
                i0 = j + 1

        ax.set_title(f"{sess['camera']}_{sess['timestamp']}")
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([f'F{fid}' for fid in female_ids])
        ax.set_xlabel(xlabel)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ethograms_all_sessions_2M.png'
    fig.savefig(out, dpi=250, bbox_inches='tight')
    print(f'Saved combined ethograms (2M) to {out}')


def plot_summary_dotplots_2m(agg_df, session_results):
    # similar to 1M but uses agg_df row order
    sessions = [f"{s['camera']}_{s['timestamp']}" for s in session_results]
    n = len(sessions)
    palette = sns.color_palette('tab20', n_colors=n)
    color_map = {sess: palette[i] for i, sess in enumerate(sessions)}

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

    agg_df_indexed = agg_df.set_index('session_key')
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
            vals.append(val / 60.0 if (is_time and not pd.isna(val)) else val)
            colors.append(color_map[key])

        vals_arr = np.array(vals, dtype=float)
        mask = ~np.isnan(vals_arr)
        vals_masked = vals_arr[mask]
        colors_masked = np.array(colors)[mask]

        if len(vals_masked) > 0:
            sns.violinplot(y=vals_masked, ax=ax, inner=None, color='lightgray', alpha=0.5, zorder=1)
            sns.boxplot(y=vals_masked, ax=ax, width=0.1, showcaps=True,
                        boxprops={'facecolor': 'white', 'zorder': 1}, showfliers=False,
                        whiskerprops={'zorder': 1}, capprops={'zorder': 1}, medianprops={'zorder': 2})

        jitter = np.random.normal(loc=0.0, scale=0.04, size=len(vals_masked))
        xs = np.zeros(len(vals_masked)) + jitter
        ax.scatter(xs, vals_masked, c=colors_masked, s=80, edgecolor='k', zorder=10)
        ax.set_ylabel(label)
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Sessions', fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / 'summary_dotplots_sessions_2M.png'
    fig.savefig(out, dpi=250, bbox_inches='tight')
    print(f'Saved summary dot-plots (2M) to {out}')


def main():
    cache_file = OUTPUT_DIR / 'session_results_2M.pkl'
    if cache_file.exists():
        print(f'Loading cached session results from {cache_file} (delete to reprocess)')
        with open(cache_file, 'rb') as f:
            session_results = pickle.load(f)
    else:
        sessions = find_sessions(DATA_DIR)
        print(f'Found {len(sessions)} candidate sessions')
        session_results = []
        for s in sessions:
            trajs = s['trajectories']
            areas = s['areas']
            cam = s.get('camera', 'CamA')
            print(f"Processing session {s['timestamp']} ({cam}): {len(trajs)} parts, areas: {areas}")
            pixels_per_cm = PIXELS_PER_CM_CAMA if cam == 'CamA' else PIXELS_PER_CM_CAMB
            try:
                result = detect_behaviors_2m(trajs, areas, pixels_per_cm=pixels_per_cm)
            except Exception as e:
                print(f"  Skipped session {s['timestamp']} due to error: {e}")
                continue
            session_results.append({'camera': cam, 'timestamp': s['timestamp'], 'result': result})

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(session_results, f)
            print(f'Saved cached session results to {cache_file}')
        except Exception as e:
            print(f'Warning: could not save cache: {e}')

    if not session_results:
        print('No sessions processed, exiting')
        return

    agg_df = aggregate_sessions_2m(session_results)
    agg_out = OUTPUT_DIR / 'aggregated_session_stats_2M.csv'
    agg_df.to_csv(agg_out, index=False)
    print(f'Saved aggregated stats to {agg_out}')

    plot_sessions_ethograms_2m(session_results, cols=4)
    plot_summary_dotplots_2m(agg_df, session_results)


if __name__ == '__main__':
    main()
