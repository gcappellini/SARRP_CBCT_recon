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
