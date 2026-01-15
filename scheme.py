"""
Create a 2D schematic diagram of a SARRP-style cone-beam CT gantry geometry.

Goal:
- A clear side-view (sagittal-like) schematic showing:
  1. The rodent (animal) at the center
  2. The rotation axis (center of rotation, COR)
  3. The X-ray source position at one gantry angle
  4. The detector plane at that angle
  5. Key distance labels: SOD, SDD, and detector dimensions
  6. Angles: gantry rotation angle, any gantry tilt
  7. Piercing point (where central ray hits detector center)

Requirements:

Use matplotlib to draw:
- A circle or ellipse representing the rodent cross-section (schematic)
- A point or small marker for COR (center of rotation)
- A point labeled "Source" at distance SOD from COR
- A vertical line representing the detector plane
- A line from Source through COR to the detector (central ray)
- Arrows and distance labels for:
  - SOD (source to isocenter/COR distance)
  - SDD (source to detector distance)
  - Detector pixel size and dimension
  - Magnification factor (SDD/SOD)
- A curved arrow showing the gantry rotation angle (e.g., 0°, 90°, 180°)
- Optional: show the piercing point (PP) on the detector and its offset from center

Title: "SARRP Cone-Beam CT Geometry (Side View at One Gantry Angle)"

Annotations:
- Label each key component (source, detector, rodent, COR, rotation axis)
- Show the detector in its local u-v frame (u horizontal, v vertical)
- Optionally include a note about:
  - "Rodent frame: fixed"
  - "Gantry + detector: rotate around COR"
  - "Central ray defines magnification"

Output:
- Save figure as "geometry_schematic.png"
- Print diagram to console for quick inspection

Example geometry values to use in the schematic (for concreteness):
- SOD = 353.266 mm
- SDD = 634.0 mm (≈ 1.75× magnification)
- COR at origin (0, 0, 0) in rodent frame
- Detector size: 204.8 × 204.8 mm
- Rodent radius: ~15 mm (schematic, not to scale)

Please generate the full function `plot_geometry_schematic(geom: Geometry) -> None`
that takes a Geometry object and produces the diagram above.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from geometry import Geometry, load_geometry_from_yaml


def plot_geometry_schematic(geom: Geometry, angle_idx: int = 0, save_path: str = "geometry_schematic.png") -> None:
    """
    Plot a 2D schematic diagram of the SARRP cone-beam CT geometry.
    
    Parameters
    ----------
    geom : Geometry
        Geometry object containing all CT system parameters.
    angle_idx : int, optional
        Index of the projection angle to visualize (default: 0).
    save_path : str, optional
        Path to save the output figure (default: "geometry_schematic.png").
    """
    # Get the gantry angle for this view
    angle = geom.angles_rad[angle_idx]
    angle_deg = np.rad2deg(angle)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    # Define COR at origin
    cor_x, cor_y = 0, 0
    
    # Calculate source position (rotates around COR)
    # At angle=0, source is at (-SOD, 0) in standard setup
    # As gantry rotates by angle θ, source rotates counterclockwise
    source_x = -geom.SOD * np.cos(angle)
    source_y = -geom.SOD * np.sin(angle)
    
    # Calculate detector center position (opposite side of COR from source)
    # At angle=0, detector is at (+IDD, 0)
    det_center_x = geom.IDD * np.cos(angle)
    det_center_y = geom.IDD * np.sin(angle)
    
    # Detector orientation: perpendicular to the source-detector line
    # The detector normal points from detector towards source
    det_normal_angle = angle + np.pi  # pointing back towards source
    
    # Detector dimensions
    det_width = geom.det_size[0]  # u direction (horizontal in detector frame)
    det_height = geom.det_size[1]  # v direction (vertical in detector frame)
    
    # Calculate detector corners (rectangle perpendicular to central ray)
    # u-axis is perpendicular to central ray (in-plane rotation)
    # v-axis would be out of the 2D plane, so we show detector as a line
    u_direction_angle = angle + np.pi/2
    u_direction = np.array([np.cos(u_direction_angle), np.sin(u_direction_angle)])
    
    # Detector endpoints (showing width in 2D side view)
    det_half_width = det_width / 2
    det_p1 = np.array([det_center_x, det_center_y]) - u_direction * det_half_width
    det_p2 = np.array([det_center_x, det_center_y]) + u_direction * det_half_width
    
    # --- Draw components ---
    
    # 1. Draw rodent (circle at COR)
    rodent_radius = 15  # mm, schematic size
    rodent = Circle((cor_x, cor_y), rodent_radius, color='lightblue', 
                    alpha=0.6, edgecolor='blue', linewidth=2, label='Rodent')
    ax.add_patch(rodent)
    
    # 2. Draw COR (center of rotation)
    ax.plot(cor_x, cor_y, 'k+', markersize=15, markeredgewidth=2, label='COR (Isocenter)')
    ax.plot(cor_x, cor_y, 'ko', markersize=8, fillstyle='none', markeredgewidth=1.5)
    
    # 3. Draw X-ray source
    ax.plot(source_x, source_y, 'r*', markersize=20, label='X-ray Source')
    
    # 4. Draw detector plane (as a thick line)
    ax.plot([det_p1[0], det_p2[0]], [det_p1[1], det_p2[1]], 
            'g-', linewidth=6, label='Detector', solid_capstyle='round')
    
    # 5. Draw detector center point
    ax.plot(det_center_x, det_center_y, 'go', markersize=8)
    
    # 6. Draw central ray (from source through COR to detector)
    ax.plot([source_x, det_center_x], [source_y, det_center_y], 
            'r--', linewidth=1.5, alpha=0.7, label='Central Ray')
    
    # 7. Draw cone beam edges (schematic - show a few rays)
    # Rays to detector edges
    ax.plot([source_x, det_p1[0]], [source_y, det_p1[1]], 
            'r:', linewidth=1, alpha=0.4)
    ax.plot([source_x, det_p2[0]], [source_y, det_p2[1]], 
            'r:', linewidth=1, alpha=0.4)
    
    # --- Distance annotations ---
    
    # SOD: Source to COR distance
    mid_x_sod = (source_x + cor_x) / 2
    mid_y_sod = (source_y + cor_y) / 2
    offset_perp = np.array([-np.sin(angle), np.cos(angle)]) * 30
    
    ax.annotate('', xy=(cor_x, cor_y), xytext=(source_x, source_y),
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=2, shrinkA=0, shrinkB=0))
    ax.text(mid_x_sod + offset_perp[0], mid_y_sod + offset_perp[1], 
            f'SOD = {geom.SOD:.1f} mm', fontsize=11, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # SDD: Source to detector distance
    mid_x_sdd = (source_x + det_center_x) / 2
    mid_y_sdd = (source_y + det_center_y) / 2
    offset_perp_sdd = -offset_perp
    
    ax.annotate('', xy=(det_center_x, det_center_y), xytext=(source_x, source_y),
                arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2, shrinkA=0, shrinkB=0))
    ax.text(mid_x_sdd + offset_perp_sdd[0], mid_y_sdd + offset_perp_sdd[1], 
            f'SDD = {geom.SDD:.1f} mm', fontsize=11, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # IDD: Isocenter to detector distance (optional)
    mid_x_idd = (cor_x + det_center_x) / 2
    mid_y_idd = (cor_y + det_center_y) / 2
    
    # Detector width annotation
    det_mid_u = (det_p1 + det_p2) / 2
    offset_det = u_direction * (det_half_width * 1.15)
    ax.text(det_mid_u[0] + offset_det[0], det_mid_u[1] + offset_det[1],
            f'Det Width: {det_width:.1f} mm\n({geom.det_nu} pixels)',
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7))
    
    # Magnification factor
    magnification = geom.SDD / geom.SOD
    ax.text(0.02, 0.98, f'Magnification: {magnification:.3f}×',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.7))
    
    # Pixel size info
    ax.text(0.02, 0.90, 
            f'Pixel size: {geom.det_pixel_size[0]:.3f} × {geom.det_pixel_size[1]:.3f} mm',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.7))
    
    # --- Rotation angle indicator ---
    # Draw an arc showing the gantry rotation
    if angle_deg != 0:
        arc_radius = geom.SOD * 0.4
        arc = Arc((cor_x, cor_y), 2*arc_radius, 2*arc_radius, 
                  angle=0, theta1=180, theta2=180 - angle_deg,
                  color='purple', linewidth=2, linestyle='--')
        ax.add_patch(arc)
        
        # Angle label
        arc_label_angle = np.deg2rad(180 - angle_deg/2)
        arc_label_x = cor_x + arc_radius * 1.2 * np.cos(arc_label_angle)
        arc_label_y = cor_y + arc_radius * 1.2 * np.sin(arc_label_angle)
        ax.text(arc_label_x, arc_label_y, f'θ = {angle_deg:.1f}°',
                fontsize=10, color='purple', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='purple', alpha=0.8))
    
    # --- Reference frame annotations ---
    # Add text box with geometry notes
    notes = (
        "Geometry Notes:\n"
        "• Rodent frame: fixed at origin\n"
        "• Gantry rotates around COR\n"
        "• Source & detector move together\n"
        f"• Projection {angle_idx + 1}/{geom.n_projections}"
    )
    ax.text(0.98, 0.98, notes,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                     edgecolor='gray', alpha=0.9))
    
    # --- Axis setup ---
    # Set axis limits with some padding
    all_x = [source_x, cor_x, det_center_x, det_p1[0], det_p2[0]]
    all_y = [source_y, cor_y, det_center_y, det_p1[1], det_p2[1]]
    
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    padding = max(x_range, y_range) * 0.2
    
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    
    # Labels and formatting
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'SARRP Cone-Beam CT Geometry\n(2D Side View at Gantry Angle {angle_deg:.1f}°)',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Geometry schematic saved to: {save_path}")
    
    # Show in console
    plt.show()


if __name__ == "__main__":
    # Load geometry from config file
    config_path = Path(__file__).parent / "conf.yaml"
    
    if config_path.exists():
        geom = load_geometry_from_yaml(config_path)
        
        # Plot at different angles
        plot_geometry_schematic(geom, angle_idx=0, save_path="geometry_schematic_0deg.png")
        
        # Optionally plot at other angles
        if geom.n_projections > 1:
            plot_geometry_schematic(geom, angle_idx=geom.n_projections//4, 
                                   save_path="geometry_schematic_90deg.png")
    else:
        print(f"Error: Configuration file not found at {config_path}")
        print("Creating schematic with default parameters...")
        
        # Create a default geometry for demonstration
        from dataclasses import replace
        geom_default = Geometry(
            SOD=353.266,
            SDD=634.0,
            IDD=634.0 - 353.266,
            source_pos=(-353.266, 0, 0),
            detector_ref=(634.0 - 353.266, 0, 0),
            COR=(0, 0, 0),
            CTilt=(1, 0, 0, 0),
            det_nu=512,
            det_nv=512,
            det_size=(204.8, 204.8),
            rotation_axis=(0, 0, 1),
            angle_start_deg=0,
            angle_end_deg=360,
            n_projections=360,
        )
        plot_geometry_schematic(geom_default, angle_idx=0, 
                               save_path="geometry_schematic_default.png")
