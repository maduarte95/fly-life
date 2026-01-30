"""
Plot trajectories and heatmaps for the 5F1M sessions described in the repository.

Produces for each session:
- Individual arena trajectory plots (6 panels, one per fly) and a combined plot.
- Heatmaps (time spent) by 8 angular sectors (45deg each): 6 individual + combined.
- Heatmaps (time spent) by concentric rings: 6 individual + combined.

Assumptions:
- Trajectories are given in pixels; arena center is estimated as the mean of all
  non-NaN positions across flies. Radius is estimated from the 99th percentile
  distance from that center and padded by 5%.
- Time per frame is taken from the 'time' column in the trajectories CSVs; total
  plotting period is 0-120 minutes (7200 seconds).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.collections import LineCollection
from glob import glob
import re


TRAJECTORIES_DIR = Path('data/1M/dataset_2')

def discover_sessions():
    """Discover all sessions by finding unique timestamp+setup combinations."""
    if not TRAJECTORIES_DIR.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {TRAJECTORIES_DIR}")
    
    # Find all trajectory files
    traj_files = sorted(TRAJECTORIES_DIR.glob('*_trajectories.csv'))
    
    # Extract session identifiers (everything before _Video1 or _Video2)
    sessions = {}
    for f in traj_files:
        # Pattern: SPECIES_CamXVideoY_TIMESTAMP_SETUP_trajectories.csv
        # Extract everything up to and including the timestamp+setup
        match = re.match(r'(.*?)_(CamA|CamB)Video([12])_(.*?)_(Setup\d+)_trajectories', f.name)
        if match:
            species = match.group(1)
            cam = match.group(2)
            video = match.group(3)
            timestamp = match.group(4)
            setup = match.group(5)
            
            # Session key = species + timestamp + setup (ignores camera)
            session_key = f"{species}_{timestamp}_{setup}"
            
            if session_key not in sessions:
                sessions[session_key] = {'Video1': None, 'Video2': None, 'areas': None}
            
            # Store the file path
            sessions[session_key][f'Video{video}'] = f
    
    # Find corresponding areas files for each session
    areas_files = sorted(TRAJECTORIES_DIR.glob('*_areas.csv'))
    for session_key in sessions:
        species_match = re.search(r'DGRP\d+', session_key)
        if species_match:
            species = species_match.group()
            for areas_file in areas_files:
                if species in areas_file.name and session_key.split(species)[1] in areas_file.name:
                    sessions[session_key]['areas'] = areas_file
                    break
    
    # Validate all sessions have required files
    valid_sessions = {}
    for session_key, files in sessions.items():
        if files['Video1'] and files['Video2'] and files['areas']:
            valid_sessions[session_key] = files
    
    return valid_sessions

SESSIONS = discover_sessions()
print(f"Found {len(SESSIONS)} sessions:")
for i, session_key in enumerate(sorted(SESSIONS.keys()), 1):
    print(f"  {i}. {session_key}")

OUTPUT_BASE_DIR = Path('output/plots_arena')
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

PLOT_TIME_MAX = 120 * 60  # 120 minutes in seconds
# fixed vmax for sector color mapping (seconds) for individual plots
SECTOR_COLOR_VMAX_INDIV = 2000.0
# if None, combined sector vmax will be computed from data; otherwise override
SECTOR_COLOR_VMAX_COMBINED = None

# radial color vmax overrides (None -> compute from data)
RADIAL_COLOR_VMAX_INDIV = None
RADIAL_COLOR_VMAX_COMBINED = None


def get_session_output_dir(session_key):
    """Get the output directory for a specific session."""
    session_dir = OUTPUT_BASE_DIR / session_key
    session_dir.mkdir(exist_ok=True)
    return session_dir


def load_and_concat(traj1, traj2):
    t1 = pd.read_csv(traj1)
    t2 = pd.read_csv(traj2)
    # offset second video times to continue from first
    offset = t1['time'].iloc[-1]
    t2 = t2.copy()
    t2['time'] = t2['time'] + offset
    traj = pd.concat([t1, t2], ignore_index=True)
    return traj


def identify_male_ids(areas_path):
    areas = pd.read_csv(areas_path)
    # smallest mean area -> male 1; for 2M experiments this may change
    male_index = areas['mean'].idxmin()
    male_id = male_index + 1
    return male_id


def estimate_arena_geometry(df, fly_ids):
    # collect all valid positions
    xs = []
    ys = []
    for fid in fly_ids:
        xcol = f'x{fid}'
        ycol = f'y{fid}'
        if xcol in df.columns and ycol in df.columns:
            xs.append(df[xcol].values)
            ys.append(df[ycol].values)
    if not xs:
        raise RuntimeError('No position columns found')
    xs = np.concatenate([a[~np.isnan(a)] for a in xs])
    ys = np.concatenate([a[~np.isnan(a)] for a in ys])
    # compute center as midpoint of min/max extents (simpler and robust)
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    # radius as half the larger span (x or y), with a small padding
    span_x = (xmax - xmin) * 0.5
    span_y = (ymax - ymin) * 0.5
    radius = max(span_x, span_y) * 1.05
    return (cx, cy, radius)


def time_weights(times):
    # compute time spent at each frame as difference to next frame
    dt = np.diff(times)
    # last frame assume same as median dt
    if len(dt) == 0:
        return np.array([0.0])
    median_dt = np.median(dt)
    dt = np.append(dt, median_dt)
    return dt

def hide_spines(ax):
    """Hide the rectangular axes spines and ticks for a plotting Axes."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_individual_arenas(traj, fly_ids, center, radius, output_dir):
    # one subplot per fly + combined
    n = len(fly_ids)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    times = traj['time'].values
    mask_time = times <= PLOT_TIME_MAX
    # detect pursuit states using ethogram-like thresholds
    PIXELS_PER_CM = 208.0
    PURSUIT_DISTANCE_CM = 1.0
    PURSUIT_MIN_DURATION = 5.0
    # copulation thresholds (distance in cm, min duration in seconds)
    COPULATION_DISTANCE_CM = 0.3
    COPULATION_MIN_DURATION = 30.0
    # estimate frame rate
    total_time = times[-1] - times[0] if len(times) > 1 else 1.0
    frame_rate = len(times) / total_time if total_time > 0 else 1.0
    min_frames = int(PURSUIT_MIN_DURATION * frame_rate)
    min_frames_cop = int(COPULATION_MIN_DURATION * frame_rate)

    areas_path = SESSIONS[current_session]['areas']
    male_id_detected = identify_male_ids(areas_path)
    pursuit_states = {}
    # compute male coords
    mx = traj[f'x{male_id_detected}'].values
    my = traj[f'y{male_id_detected}'].values
    for fid in fly_ids:
        if fid == male_id_detected:
            continue
        fx = traj[f'x{fid}'].values
        fy = traj[f'y{fid}'].values
        dist_px = np.sqrt((mx - fx)**2 + (my - fy)**2)
        dist_cm = dist_px / PIXELS_PER_CM
        within = (dist_cm < PURSUIT_DISTANCE_CM).astype(int)
        pursuit = np.zeros(len(within), dtype=bool)
        in_p = False
        start_idx = 0
        for i in range(len(within)):
            if within[i] == 1 and not in_p:
                in_p = True
                start_idx = i
            elif within[i] == 0 and in_p:
                dur = i - start_idx
                if dur >= min_frames:
                    pursuit[start_idx:i] = True
                in_p = False
        if in_p:
            dur = len(within) - start_idx
            if dur >= min_frames:
                pursuit[start_idx:] = True
        pursuit_states[fid] = pursuit

    # detect copulation events (distance smaller than COPULATION_DISTANCE_CM for sufficient duration)
    copulation_states = {}
    for fid in fly_ids:
        if fid == male_id_detected:
            continue
        fx = traj[f'x{fid}'].values
        fy = traj[f'y{fid}'].values
        dist_px = np.sqrt((mx - fx)**2 + (my - fy)**2)
        dist_cm = dist_px / PIXELS_PER_CM
        within_cop = (dist_cm < COPULATION_DISTANCE_CM).astype(int)
        cop = np.zeros(len(within_cop), dtype=bool)
        in_c = False
        start_idx = 0
        for i in range(len(within_cop)):
            if within_cop[i] == 1 and not in_c:
                in_c = True
                start_idx = i
            elif within_cop[i] == 0 and in_c:
                dur = i - start_idx
                if dur >= min_frames_cop:
                    cop[start_idx:i] = True
                in_c = False
        if in_c:
            dur = len(within_cop) - start_idx
            if dur >= min_frames_cop:
                cop[start_idx:] = True
        copulation_states[fid] = cop

    # male copulation when any female is copulating
    male_copulation = np.zeros(len(times), dtype=bool)
    for arr in copulation_states.values():
        male_copulation = male_copulation | arr

    # male pursuit if any female is being pursued
    male_pursuit = np.zeros(len(times), dtype=bool)
    for arr in pursuit_states.values():
        male_pursuit = male_pursuit | arr

    for i, fid in enumerate(fly_ids):
        ax = axes[i]
        x = traj[f'x{fid}'].values
        y = traj[f'y{fid}'].values
        # gender label
        gender = 'Male' if fid == male_id_detected else 'Female'
        ax.set_title(f'Fly {fid} — {gender}')
        # draw arena
        circle = plt.Circle((center[0], center[1]), radius, edgecolor='k', facecolor='none')
        ax.add_patch(circle)
        # basic trajectory
        ax.plot(x[mask_time], y[mask_time], linewidth=0.7, color='gray')
        # overlay pursuit segments in red
        if fid == male_id_detected:
            pm = male_pursuit
        else:
            pm = pursuit_states.get(fid, np.zeros(len(times), dtype=bool))
        pm_mask = pm & mask_time
        if pm_mask.sum() > 1:
            idxs = np.where(pm_mask)[0]
            runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0]+1)
            for run in runs:
                if len(run) < 2:
                    continue
                seg_x = x[run]
                seg_y = y[run]
                pts = np.vstack([seg_x, seg_y]).T
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                # use RGB(242,169,59) for pursuit highlighting (hex #F2A93B)
                lc2 = LineCollection(segs, colors='#F2A93B', linewidths=2.0)
                ax.add_collection(lc2)
        # overlay copulation locations as red dots (or lines)
        if fid == male_id_detected:
            cop_mask = male_copulation & mask_time
        else:
            cop_mask = copulation_states.get(fid, np.zeros(len(times), dtype=bool)) & mask_time
        if cop_mask.sum() > 0:
            ax.scatter(x[cop_mask], y[cop_mask], s=18, c='red', edgecolors='k', linewidth=0.3, zorder=6)
        ax.set_aspect('equal')
        ax.set_xlim(center[0]-radius*1.1, center[0]+radius*1.1)
        ax.set_ylim(center[1]-radius*1.1, center[1]+radius*1.1)
        # remove x/y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        hide_spines(ax)

    # hide unused subplots
    for j in range(n, nrows*ncols):
        axes[j].axis('off')

    out = output_dir / 'individual_arenas.png'
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')

    # combined plot
    fig, ax = plt.subplots(figsize=(6,6))
    circle = plt.Circle((center[0], center[1]), radius, edgecolor='k', facecolor='none')
    ax.add_patch(circle)
    cmap = plt.get_cmap('tab10')
    for i, fid in enumerate(fly_ids):
        x = traj[f'x{fid}'].values
        y = traj[f'y{fid}'].values
        ax.plot(x[mask_time], y[mask_time], linewidth=0.8, label=f'Fly {fid}', color=cmap(i % 10))
    ax.set_aspect('equal')
    ax.set_xlim(center[0]-radius*1.1, center[0]+radius*1.1)
    ax.set_ylim(center[1]-radius*1.1, center[1]+radius*1.1)
    ax.legend(loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    hide_spines(ax)
    out = output_dir / 'combined_trajectories.png'
    plt.tight_layout()
    
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def sector_heatmaps(traj, fly_ids, center, radius, output_dir):
    # compute angles and distance per frame and time weights
    times = traj['time'].values
    dt = time_weights(times)
    mask_time = times <= PLOT_TIME_MAX

    angles_all = {}
    dists_all = {}
    for fid in fly_ids:
        x = traj[f'x{fid}'].values - center[0]
        y = traj[f'y{fid}'].values - center[1]
        angles = np.degrees(np.arctan2(y, x)) % 360
        dists = np.sqrt(x**2 + y**2)
        angles_all[fid] = angles
        dists_all[fid] = dists

    # 18 sectors (20 deg each)
    sectors = np.arange(0, 360, 20)
    nsec = len(sectors)

    # compute counts for all flies first so we can normalize colors across individuals
    counts_per_fly = {}
    combined_counts = np.zeros(nsec)
    for fid in fly_ids:
        ang = angles_all[fid]
        dists = dists_all[fid]
        mask = (dists <= radius) & mask_time
        ang_masked = ang[mask]
        dt_masked = dt[mask]
        counts = np.zeros(nsec)
        for k in range(nsec):
            lo = sectors[k]
            hi = sectors[(k+1)%nsec]
            if lo < hi:
                sel = (ang_masked >= lo) & (ang_masked < hi)
            else:
                sel = (ang_masked >= lo) | (ang_masked < hi)
            counts[k] = dt_masked[sel].sum()
        counts_per_fly[fid] = counts
        combined_counts += counts

    # determine global max for normalization (include combined)
    global_max = 0.0
    for fid, c in counts_per_fly.items():
        if c.sum() > global_max:
            global_max = max(global_max, c.max())
    if combined_counts.sum() > global_max:
        global_max = max(global_max, combined_counts.max())

    # individual plots: draw filled sector wedges colored by shared colormap (white->red)
    from matplotlib.patches import Wedge
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = plt.cm.Reds

    n = len(fly_ids)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()

    # setup normalization for sector plots (individual vs combined)
    indiv_vmax = SECTOR_COLOR_VMAX_INDIV
    norm = Normalize(vmin=0.0, vmax=max(1.0, indiv_vmax))

    for i, fid in enumerate(fly_ids):
        ax = axes[i]
        counts = counts_per_fly[fid]
        if counts.sum() > 0:
            for k in range(nsec):
                lo = sectors[k]
                hi = sectors[(k+1)%nsec]
                theta1 = lo
                theta2 = hi
                color = cmap(norm(counts[k]))
                wedge = Wedge(center, radius, theta1, theta2, facecolor=color, edgecolor='k', linewidth=0.4)
                ax.add_patch(wedge)
            # arena boundary
            outer = plt.Circle(center, radius, edgecolor='k', facecolor='none')
            ax.add_patch(outer)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_aspect('equal')
        ax.set_xlim(center[0]-radius*1.05, center[0]+radius*1.05)
        ax.set_ylim(center[1]-radius*1.05, center[1]+radius*1.05)
        ax.set_title(f'Fly {fid} sector time (s)')
        hide_spines(ax)

    for j in range(n, nrows*ncols):
        axes[j].axis('off')

    # add shared colorbar for the figure on the rightmost side
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # reserve space on the right and add a dedicated colorbar axis
    fig.subplots_adjust(right=0.75)
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cax, orientation='vertical', label='Time (s)')

    out = output_dir / 'sector_time_individual.png'
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')

    # combined (sum across flies) as filled sectors
    fig, ax = plt.subplots(figsize=(6,6))
    if combined_counts.sum() > 0:
        # determine combined vmax (allow override)
        comb_vmax = SECTOR_COLOR_VMAX_COMBINED if SECTOR_COLOR_VMAX_COMBINED is not None else max(1.0, combined_counts.max())
        combined_norm = Normalize(vmin=0.0, vmax=comb_vmax)
        for k in range(nsec):
            lo = sectors[k]
            hi = sectors[(k+1)%nsec]
            theta1 = lo
            theta2 = hi
            color = cmap(combined_norm(combined_counts[k]))
            wedge = Wedge(center, radius, theta1, theta2, facecolor=color, edgecolor='k', linewidth=0.6)
            ax.add_patch(wedge)
        outer = plt.Circle(center, radius, edgecolor='k', facecolor='none', linewidth=0.8)
        ax.add_patch(outer)
        sm = ScalarMappable(norm=combined_norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='Time (s)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_aspect('equal')
    ax.set_xlim(center[0]-radius*1.05, center[0]+radius*1.05)
    ax.set_ylim(center[1]-radius*1.05, center[1]+radius*1.05)
    ax.set_title('Combined sector time (s)')
    hide_spines(ax)
    out = output_dir / 'sector_time_combined.png'
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def radial_heatmaps(traj, fly_ids, center, radius, output_dir, n_rings=5):
    times = traj['time'].values
    dt = time_weights(times)
    mask_time = times <= PLOT_TIME_MAX

    # define ring edges from 0..radius
    edges = np.linspace(0, radius, n_rings+1)

    # individual
    n = len(fly_ids)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()

    for i, fid in enumerate(fly_ids):
        ax = axes[i]
        x = traj[f'x{fid}'].values - center[0]
        y = traj[f'y{fid}'].values - center[1]
        dists = np.sqrt(x**2 + y**2)
        mask = mask_time
        counts = np.zeros(n_rings)
        for r in range(n_rings):
            sel = (dists >= edges[r]) & (dists < edges[r+1]) & mask
            counts[r] = dt[sel].sum()
        # draw concentric rings (annuli) colored by time spent
        if counts.sum() > 0:
            from matplotlib.patches import Wedge
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            cmap = plt.cm.Reds
            # normalize from 0..max so 0 -> white, max -> deep red
            # use global normalization across flies for radial as well (use maximum across flies)
            # we'll compute a simple per-figure vmax below if not already set
            try:
                global_radial_vmax
            except NameError:
                # compute global radial vmax across all flies unless override provided
                if RADIAL_COLOR_VMAX_INDIV is not None:
                    global_radial_vmax = max(1.0, RADIAL_COLOR_VMAX_INDIV)
                else:
                    all_counts = []
                    for fid2 in fly_ids:
                        x2 = traj[f'x{fid2}'].values - center[0]
                        y2 = traj[f'x{fid2}'].values - center[1]
                        dists2 = np.sqrt(x2**2 + y2**2)
                        cnts2 = np.zeros(n_rings)
                        for r in range(n_rings):
                            sel2 = (dists2 >= edges[r]) & (dists2 < edges[r+1]) & mask_time
                            cnts2[r] = dt[sel2].sum()
                        all_counts.append(cnts2)
                    all_counts = np.concatenate(all_counts)
                    global_radial_vmax = max(1.0, all_counts.max())
            norm = Normalize(vmin=0.0, vmax=global_radial_vmax)
            colors = cmap(norm(counts))
            # draw from outermost to innermost so inner rings are on top
            for r in reversed(range(n_rings)):
                r_outer = edges[r+1]
                r_inner = edges[r]
                width = r_outer - r_inner
                color = colors[r]
                wedge = Wedge(center, r_outer, 0, 360, width=width, facecolor=color, edgecolor='k', linewidth=0.4)
                ax.add_patch(wedge)
            # draw arena boundary
            outer_circle = plt.Circle(center, edges[-1], edgecolor='k', facecolor='none', linewidth=0.6)
            ax.add_patch(outer_circle)
            # colorbar will be added once after plotting all subplots
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_aspect('equal')
        ax.set_xlim(center[0]-radius*1.05, center[0]+radius*1.05)
        ax.set_ylim(center[1]-radius*1.05, center[1]+radius*1.05)
        ax.set_title(f'Fly {fid} radial time (s)')
        hide_spines(ax)

    for j in range(n, nrows*ncols):
        axes[j].axis('off')
        
    # add a single colorbar on the right for the radial individual figure
    from matplotlib.cm import ScalarMappable
    # use individual radial normalization for the colorbar
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=global_radial_vmax), cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=0.75)
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cax, orientation='vertical', label='Time (s)')
    out = output_dir / 'radial_time_individual.png'
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')

    # combined across flies
    combined_counts = np.zeros(n_rings)
    for fid in fly_ids:
        x = traj[f'x{fid}'].values - center[0]
        y = traj[f'y{fid}'].values - center[1]
        dists = np.sqrt(x**2 + y**2)
        mask = mask_time
        for r in range(n_rings):
            sel = (dists >= edges[r]) & (dists < edges[r+1]) & mask
            combined_counts[r] += dt[sel].sum()

    # combined radial plot as concentric rings
    fig, ax = plt.subplots(figsize=(6,6))
    if combined_counts.sum() > 0:
        from matplotlib.patches import Wedge
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        cmap = plt.cm.Reds
        # combined radial normalization; allow override
        comb_radial_vmax = RADIAL_COLOR_VMAX_COMBINED if RADIAL_COLOR_VMAX_COMBINED is not None else max(1.0, combined_counts.max())
        norm_comb = Normalize(vmin=0.0, vmax=comb_radial_vmax)
        colors = cmap(norm_comb(combined_counts))
        for r in reversed(range(n_rings)):
            r_outer = edges[r+1]
            r_inner = edges[r]
            width = r_outer - r_inner
            color = colors[r]
            wedge = Wedge(center, r_outer, 0, 360, width=width, facecolor=color, edgecolor='k', linewidth=0.6)
            ax.add_patch(wedge)
        outer_circle = plt.Circle(center, edges[-1], edgecolor='k', facecolor='none', linewidth=0.8)
        ax.add_patch(outer_circle)
        sm = ScalarMappable(norm=norm_comb, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='Time (s)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_aspect('equal')
    ax.set_xlim(center[0]-radius*1.05, center[0]+radius*1.05)
    ax.set_ylim(center[1]-radius*1.05, center[1]+radius*1.05)
    ax.set_title('Combined radial time (s)')
    hide_spines(ax)
    out = output_dir / 'radial_time_combined.png'
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    print(f'Saved {out}')


def main():
    global current_session
    
    for session_key in sorted(SESSIONS.keys()):
        current_session = session_key
        print(f"\n{'='*60}")
        print(f"Processing session: {session_key}")
        print(f"{'='*60}")
        
        # Get file paths for this session
        traj1_path = SESSIONS[session_key]['Video1']
        traj2_path = SESSIONS[session_key]['Video2']
        areas_path = SESSIONS[session_key]['areas']
        
        # Load and concatenate trajectories
        traj = load_and_concat(traj1_path, traj2_path)
        areas = pd.read_csv(areas_path)
        
        # identify male (smallest mean)
        male_index = areas['mean'].idxmin()
        male_id = male_index + 1
        all_ids = list(range(1, 7))
        female_ids = [fid for fid in all_ids if fid != male_id]
        fly_ids = all_ids

        center_x, center_y, radius = estimate_arena_geometry(traj, fly_ids)
        center = (center_x, center_y)
        print(f'Estimated arena center: ({center_x:.1f}, {center_y:.1f}), radius: {radius:.1f}')

        # Get output directory for this session
        output_dir = get_session_output_dir(session_key)
        
        # Generate plots
        plot_individual_arenas(traj, fly_ids, center, radius, output_dir)
        sector_heatmaps(traj, fly_ids, center, radius, output_dir)
        radial_heatmaps(traj, fly_ids, center, radius, output_dir, n_rings=8)
        
        # Close all figures to free memory
        plt.close('all')
        
    print(f"\n{'='*60}")
    print("All sessions processed successfully!")
    print(f"Output saved to: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
