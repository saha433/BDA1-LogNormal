"""
plot_boxplots_ln.py  —  Box plots for BDA Assignment-1 Q5 (Log-Normal)

Reads the three CSV files produced by q5_lognormal and draws:
  1. A single box for the full 60-minute dataset
  2. Six boxes for the 10-minute blocks
  3. Sixty boxes for the per-minute intervals

Run AFTER q5_lognormal has produced the CSV files:
  python plot_boxplots_ln.py
"""

import csv, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── helper: read CSV into list of dicts ──────────────────────────────────
def read_csv(path):
    try:
        with open(path, newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run q5_lognormal first.")
        sys.exit(1)

def row_to_box(row):
    """Return dict of box-plot stats from a CSV row."""
    return dict(
        med   = float(row['median']),
        q1    = float(row['p25']),
        q3    = float(row['p75']),
        whislo= float(row['min']),
        whishi= float(row['max']),
        fliers= [],
        label = row.get('interval', row.get('minute', row.get('block','?')))
    )

# ── load data ────────────────────────────────────────────────────────────
global_rows  = read_csv('global_stats_ln.csv')
minute_rows  = read_csv('minute_stats_ln.csv')
ten_min_rows = read_csv('ten_min_stats_ln.csv')

# Overall row is first in global_stats_ln.csv
overall_row = next(r for r in global_rows if r['interval'] == 'Overall')
block_rows  = [r for r in global_rows if r['interval'] != 'Overall']

overall_box  = row_to_box(overall_row)
block_boxes  = [row_to_box(r) for r in block_rows]
minute_boxes = [row_to_box(r) for r in minute_rows]

# ── figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.suptitle("BDA Assignment-1 Q5 — Log-Normal Stream Statistics\n"
             r"Log-Normal($\mu$=0, $\sigma$=0.5) clipped to [0,1]",
             fontsize=14, fontweight='bold')

COLOUR = '#e67e22'   # orange — distinctive for log-normal

# ── Panel 1: 60-minute overall ───────────────────────────────────────────
ax = axes[0]
ax.bxp([overall_box], positions=[1], widths=0.4,
       boxprops=dict(color=COLOUR), medianprops=dict(color='darkred', lw=2),
       whiskerprops=dict(color=COLOUR), capprops=dict(color=COLOUR))
ax.set_title("60-Minute Dataset (Overall)", fontsize=12)
ax.set_xticks([1]); ax.set_xticklabels(['Overall'])
ax.set_ylabel("Value (clipped to [0,1])")
ax.set_xlim(0, 2)
# annotate mean
mean_val = float(overall_row['mean'])
ax.axhline(mean_val, color='steelblue', ls='--', lw=1.2, label=f'Mean={mean_val:.4f}')
ax.legend(fontsize=9)

# ── Panel 2: 10-minute blocks ────────────────────────────────────────────
ax = axes[1]
positions = list(range(1, len(block_boxes)+1))
ax.bxp(block_boxes, positions=positions, widths=0.4,
       boxprops=dict(color=COLOUR), medianprops=dict(color='darkred', lw=2),
       whiskerprops=dict(color=COLOUR), capprops=dict(color=COLOUR))
ax.set_title("10-Minute Blocks", fontsize=12)
ax.set_xticks(positions)
ax.set_xticklabels([f"Block {i}" for i in range(1, len(block_boxes)+1)], fontsize=8)
ax.set_ylabel("Value")
ax.axhline(mean_val, color='steelblue', ls='--', lw=1.0, label=f'Overall Mean={mean_val:.4f}')
ax.legend(fontsize=9)

# ── Panel 3: per-minute ──────────────────────────────────────────────────
ax = axes[2]
positions = list(range(1, len(minute_boxes)+1))
ax.bxp(minute_boxes, positions=positions, widths=0.6,
       boxprops=dict(color=COLOUR, lw=0.8),
       medianprops=dict(color='darkred', lw=1.2),
       whiskerprops=dict(color=COLOUR, lw=0.6),
       capprops=dict(color=COLOUR, lw=0.6))
ax.set_title("Per-Minute Intervals (60 boxes)", fontsize=12)
ax.set_xticks(positions[::5])
ax.set_xticklabels([f"Min {i}" for i in positions[::5]], fontsize=7, rotation=45)
ax.set_ylabel("Value")
ax.axhline(mean_val, color='steelblue', ls='--', lw=1.0, label=f'Overall Mean={mean_val:.4f}')

# Compute IQR fences and shade outlier zones
q1  = float(overall_row['p25'])
q3  = float(overall_row['p75'])
iqr = q3 - q1
lo  = q1 - 1.5*iqr
hi  = q3 + 1.5*iqr
ax.axhspan(0,   max(lo,0), color='salmon',     alpha=0.15, label=f'Lower fence < {lo:.4f}')
ax.axhspan(hi,  1.0,       color='lightcoral', alpha=0.15, label=f'Upper fence > {hi:.4f}')
ax.legend(fontsize=8)

plt.tight_layout(rect=[0,0,1,0.95])
out = 'lognormal_boxplots.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()
