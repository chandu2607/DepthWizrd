"""
Generates the architecture pipeline diagram for DepthWizard (runs/final_architecture/pipeline_diagram.png).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path("runs/final_architecture")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor('#0D1117')

def draw_box(ax, x, y, w, h, title, subtitle, color, text_color='#FFFFFF', badge=None):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.2",
                                  facecolor=color, edgecolor='#30363D', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, title, color=text_color, fontsize=11, fontweight='bold',
            ha='center', va='center', fontfamily='sans-serif')
    ax.text(x + w/2, y + h*0.32, subtitle, color='#8B949E', fontsize=8.5,
            ha='center', va='center', fontfamily='sans-serif')
    if badge:
        b_rect = patches.FancyBboxPatch((x + w - 1.6, y + h - 0.35), 1.5, 0.3,
                                        boxstyle="round,pad=0.04,rounding_size=0.1",
                                        facecolor='#238636', edgecolor='none')
        ax.add_patch(b_rect)
        ax.text(x + w - 0.85, y + h - 0.2, badge, color='#FFFFFF', fontsize=7,
                fontweight='bold', ha='center', va='center')

def draw_arrow(ax, x1, y1, x2, y2, label=None, color='#58A6FF'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4",
                                color=color, lw=2.0))
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, color='#C9D1D9',
                fontsize=8, ha='center', va='bottom', fontweight='semibold')

# Title
ax.text(8.0, 8.4, "DepthWizard — Single-View Elevation Reconstruction & 3D Flythrough",
        color='#FF7B72', fontsize=18, fontweight='bold', ha='center', fontfamily='sans-serif')
ax.text(8.0, 8.0, "End-to-End System Architecture (SIH Problem Statement 26175)",
        color='#8B949E', fontsize=11, ha='center', fontfamily='sans-serif')

# Tier 1: Input Ingestion
draw_box(ax, 0.8, 5.8, 3.2, 1.4, "Input Engine", "PNG / JPG (rDSM) & GeoTIFF (DSM)\nCRS, GSD, Affine Metadata Parser", "#161B22", "#E6EDF3", "Unified")

# Tier 2: Monocular Depth Backbone
draw_box(ax, 4.8, 5.8, 3.2, 1.4, "Depth Backbone", "Depth Anything V2 (ViT-Small)\nNormalised Depth d ∈ [0, 1] + Cache", "#161B22", "#E6EDF3", "Pretrained")

# Tier 3: Modular Calibration Engine
draw_box(ax, 8.8, 5.8, 3.4, 1.4, "Calibration Engine", "DEM Anchoring / Ground Ref / GCP\nPhase 29 PeakRecoveryMLP Prior", "#161B22", "#E6EDF3", "Modular")

# Tier 4: DSM Assembly
draw_box(ax, 12.8, 5.8, 2.5, 1.4, "Surface Assembly", "DSM = DTM + nDSM\nMetric / Relative Scale", "#161B22", "#E6EDF3", "Scientifc")

# Tier 5: 3D Reconstruction & Interactive Viewer
draw_box(ax, 0.8, 2.0, 4.5, 2.4, "Interactive 3D WebGL Flythrough", "Three.js WebGL in Streamlit\n• Orbit, Pan, Zoom, Rotate\n• WASD First-Person Navigation\n• Automated Cinematic Flythrough\n• RGB / Elevation / Height / Slope", "#1C2128", "#58A6FF", "60 FPS")

# Tier 6: Structural Height & Slope Analytics
draw_box(ax, 6.0, 2.0, 4.5, 2.4, "Height & Slope Analytics", "• Building Massing Inspector (Z_roof - Z_ground)\n• Interactive Point Height Tool\n• Terrain vs Facade Gradient Slope (deg, %)\n• Aspect & Area Calculations", "#1C2128", "#3FB950", "Analysis")

# Tier 7: Validation Dashboard
draw_box(ax, 11.2, 2.0, 4.1, 2.4, "Validation Dashboard", "• Ground Truth Reference Comparison\n• MAE, RMSE, Pearson R, Spearman Rho\n• Height-Binned Residuals (<10m to >=40m)\n• 2D Absolute Error Map", "#1C2128", "#D29922", "Audit")

# Arrows
draw_arrow(ax, 4.0, 6.5, 4.8, 6.5, "RGB Matrix")
draw_arrow(ax, 8.0, 6.5, 8.8, 6.5, "Relative d_norm")
draw_arrow(ax, 12.2, 6.5, 12.8, 6.5, "Refined ΔH")

draw_arrow(ax, 14.0, 5.8, 14.0, 5.0, "")
draw_arrow(ax, 14.0, 5.0, 3.0, 5.0, "Verified DSM / rDSM Arrays")
draw_arrow(ax, 3.0, 5.0, 3.0, 4.4, "")
draw_arrow(ax, 8.2, 5.0, 8.2, 4.4, "")
draw_arrow(ax, 13.2, 5.0, 13.2, 4.4, "")

fig.tight_layout()
fig.savefig(OUT_DIR / "pipeline_diagram.png", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print("Pipeline diagram saved to runs/final_architecture/pipeline_diagram.png")
