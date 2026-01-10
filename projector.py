"""
Implement a pure function `world_to_detector` for SARRP-style cone-beam CT.

Maps 3D points in world coordinates to 2D detector coordinates for a given projection angle.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import Geometry


def quaternion_to_rotation_matrix(q: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Parameters
    ----------
    q : tuple[float, float, float, float]
        Quaternion as (x, y, z, w).
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    rotation = Rotation.from_quat(q)
    return rotation.as_matrix()


def world_to_detector(
    point_world: np.ndarray,
    angle_idx: int,
    geom: Geometry,
) -> Tuple[float, float]:
    """
    Map a 3D point in rodent/world coordinates to detector (u, v) for one projection.
    
    Parameters
    ----------
    point_world : np.ndarray
        3D point in world coordinates [x, y, z] in mm.
    angle_idx : int
        Index into the angle array (0 <= angle_idx < n_projections).
    geom : Geometry
        Geometry object with all SARRP parameters.
    
    Returns
    -------
    tuple[float, float]
        Detector pixel coordinates (u, v). May be non-integer (sub-pixel precision).
    """
    
    # 1. Translate to COR (center of rotation)
    p = np.array(point_world, dtype=np.float64) - np.array(geom.COR, dtype=np.float64)
    
    # 2. Rotate into the current gantry frame
    # Get the current angle (rotation around the rotation axis)
    theta = geom.angles_rad[angle_idx]
    
    # Create rotation matrix around the rotation axis (e.g., Y-axis for [0, 1, 0])
    axis = np.array(geom.rotation_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  # Normalize
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    R_gantry = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    # Apply gantry rotation
    p_rotated = R_gantry @ p
    
    # 3. Apply CTilt (if needed)
    # CTilt is given as a quaternion, apply it to account for any tilt
    R_ctilt = quaternion_to_rotation_matrix(geom.CTilt)
    p_rotated = R_ctilt @ p_rotated
    
    # 4. Define source and detector plane in the view frame
    # Source is at (0, 0, -SOD) in the view frame
    source = np.array([0.0, 0.0, -geom.SOD], dtype=np.float64)
    
    # Detector plane is at z = IDD (distance from isocenter to detector)
    detector_z = geom.IDD
    
    # 5. Intersect the ray with the detector plane
    # Ray: source + t * (p_rotated - source)
    direction = p_rotated - source
    
    # Solve for t: source[2] + t * direction[2] = detector_z
    if abs(direction[2]) < 1e-10:
        # Ray is parallel to detector plane
        raise ValueError("Ray is parallel to detector plane; cannot project.")
    
    t = (detector_z - source[2]) / direction[2]
    
    # Compute intersection point on detector plane
    p_det_view = source + t * direction
    
    # Extract x, y coordinates (u, v in detector plane)
    x_det = p_det_view[0]
    y_det = p_det_view[1]
    
    # 6. Convert detector-plane coordinates to pixel indices
    # Piercing point (principal point) in pixels
    # Assume detector reference point is at the center for now; compute offset if needed
    # For simplicity, piercing point is at the center
    PP = np.array([geom.det_nu / 2.0, geom.det_nv / 2.0], dtype=np.float64)
    
    u = PP[0] + x_det / geom.det_pixel_size[0]
    v = PP[1] + y_det / geom.det_pixel_size[1]
    
    return (float(u), float(v))
