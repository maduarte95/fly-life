import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the per-fly statistics
df = pd.read_csv('output/nangrams/nan_statistics_per_fly.csv')

# Load the overlapping NaN statistics
overlap_df = pd.read_csv('output/nangrams/nan_overlapping_stats.csv')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# SUBPLOT 1: Mean NaN Time per Fly
# ============================================================================

box_data1 = [
    df[df['dataset'] == 'dataset_1']['nan_time_min'],
    df[df['dataset'] == 'dataset_2']['nan_time_min']
]

bp1 = ax1.boxplot(box_data1,
                  labels=['Dataset 1', 'Dataset 2'],
                  patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8))

# Color the boxes
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)

# Customize plot
ax1.set_ylabel('NaN Time per Fly (minutes)', fontsize=12)
ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_title('Missing Data per Individual Fly', fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)

# Add sample size information
n1 = len(df[df['dataset'] == 'dataset_1'])
n2 = len(df[df['dataset'] == 'dataset_2'])
ax1.text(1, ax1.get_ylim()[1] * 0.95, f'n={n1} flies', ha='center', fontsize=10)
ax1.text(2, ax1.get_ylim()[1] * 0.95, f'n={n2} flies', ha='center', fontsize=10)

# Add legend for mean marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
           markeredgecolor='red', markersize=8, label='Mean')
]
ax1.legend(handles=legend_elements, loc='upper left')

# ============================================================================
# SUBPLOT 2: Time with ≥2 Flies Missing Simultaneously
# ============================================================================

box_data2 = [
    overlap_df[overlap_df['dataset'] == 'dataset_1']['time_with_2plus_nans_min'],
    overlap_df[overlap_df['dataset'] == 'dataset_2']['time_with_2plus_nans_min']
]

bp2 = ax2.boxplot(box_data2,
                  labels=['Dataset 1', 'Dataset 2'],
                  patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8))

# Color the boxes
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

# Customize plot
ax2.set_ylabel('Time with ≥2 Flies Missing (minutes)', fontsize=12)
ax2.set_xlabel('Dataset', fontsize=12)
ax2.set_title('Time with ≥2 Flies Missing Simultaneously', fontsize=13, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

# Add sample size information
n3 = len(overlap_df[overlap_df['dataset'] == 'dataset_1'])
n4 = len(overlap_df[overlap_df['dataset'] == 'dataset_2'])
ax2.text(1, ax2.get_ylim()[1] * 0.95, f'n={n3} runs', ha='center', fontsize=10)
ax2.text(2, ax2.get_ylim()[1] * 0.95, f'n={n4} runs', ha='center', fontsize=10)

# Add legend for mean marker
ax2.legend(handles=legend_elements, loc='upper left')

# Add overall title
fig.suptitle('Comparison of Missing Data (NaN) Between Datasets',
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('output/nangrams/nan_comparison_boxplot.png', dpi=300, bbox_inches='tight')
print("Box plot saved to: output/nangrams/nan_comparison_boxplot.png")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n" + "-"*80)
print("METRIC 1: Mean NaN Time per Individual Fly")
print("-"*80)

for dataset_name in ['dataset_1', 'dataset_2']:
    data = df[df['dataset'] == dataset_name]['nan_time_min']
    print(f"\n{dataset_name.upper()}:")
    print(f"  Sample size: {len(data)} flies")
    print(f"  Mean NaN time: {data.mean():.2f} minutes")
    print(f"  Median NaN time: {data.median():.2f} minutes")
    print(f"  Std deviation: {data.std():.2f} minutes")
    print(f"  Min: {data.min():.2f} minutes")
    print(f"  Max: {data.max():.2f} minutes")
    print(f"  25th percentile: {data.quantile(0.25):.2f} minutes")
    print(f"  75th percentile: {data.quantile(0.75):.2f} minutes")

print("\n" + "-"*80)
print("METRIC 2: Time with ≥2 Flies Missing Simultaneously (per run)")
print("-"*80)

for dataset_name in ['dataset_1', 'dataset_2']:
    data = overlap_df[overlap_df['dataset'] == dataset_name]['time_with_2plus_nans_min']
    print(f"\n{dataset_name.upper()}:")
    print(f"  Sample size: {len(data)} runs")
    print(f"  Mean time with ≥2 flies missing: {data.mean():.2f} minutes")
    print(f"  Median time with ≥2 flies missing: {data.median():.2f} minutes")
    print(f"  Std deviation: {data.std():.2f} minutes")
    print(f"  Min: {data.min():.2f} minutes")
    print(f"  Max: {data.max():.2f} minutes")
    print(f"  25th percentile: {data.quantile(0.25):.2f} minutes")
    print(f"  75th percentile: {data.quantile(0.75):.2f} minutes")

print("\n" + "="*80)
plt.show()
