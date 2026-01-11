from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from geometry import Geometry
from projector import world_to_detector


def bilinear_interpolate(
    projection: np.ndarray,
    u: float,
    v: float,
) -> float:
    """
    Bilinearly interpolate a value from a 2D projection at (u, v).
    
    Parameters
    ----------
    projection : np.ndarray
        2D array of detector values, shape (nu, nv).
    u : float
        Detector u coordinate (column).
    v : float
        Detector v coordinate (row).
    
    Returns
    -------
    float
        Interpolated value. Returns 0.0 if (u, v) is out of bounds.
    """
    nu, nv = projection.shape
    
    # Check bounds with a small tolerance for floating-point errors
    if u < -0.5 or u >= nu - 0.5 or v < -0.5 or v >= nv - 0.5:
        return 0.0
    
    # Get integer and fractional parts
    u_int = int(np.floor(u))
    v_int = int(np.floor(v))
    u_frac = u - u_int
    v_frac = v - v_int
    
    # Clamp integer indices to valid range
    u_int = np.clip(u_int, 0, nu - 2)
    v_int = np.clip(v_int, 0, nv - 2)
    
    # Bilinear interpolation
    val00 = projection[u_int, v_int]
    val10 = projection[u_int + 1, v_int]
    val01 = projection[u_int, v_int + 1]
    val11 = projection[u_int + 1, v_int + 1]
    
    val0 = val00 * (1 - u_frac) + val10 * u_frac
    val1 = val01 * (1 - u_frac) + val11 * u_frac
    
    return val0 * (1 - v_frac) + val1 * v_frac


def backproject(
    projections: np.ndarray,
    geom: Geometry,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    volume_origin: Tuple[float, float, float],
    logger=None,
) -> np.ndarray:
    """
    Reconstruct a 3D volume in the rodent frame using voxel-driven backprojection.
    
    Parameters
    ----------
    projections : np.ndarray
        Filtered projections, shape (n_projections, nu, nv).
        Must already be log-transformed and filtered along the u-axis (FDK filtered).
    geom : Geometry
        Geometry object with all SARRP parameters.
    volume_shape : tuple[int, int, int]
        Shape of the output volume (nx, ny, nz).
    voxel_spacing : tuple[float, float, float]
        Spacing between voxels in mm (sx, sy, sz).
    volume_origin : tuple[float, float, float]
        Coordinates (in mm) of the first voxel corner in world space.
    logger : logging.Logger, optional
        Logger for progress reporting. If None, no logging.
    
    Returns
    -------
    np.ndarray
        Reconstructed volume, shape `volume_shape`, dtype float32.
    """
    
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 1. Initialize output volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    accumulator = np.zeros(volume_shape, dtype=np.float32)
    
    nx, ny, nz = volume_shape
    sx, sy, sz = voxel_spacing
    ox, oy, oz = volume_origin
    
    # Get detector dimensions
    n_projections, nu, nv = projections.shape
    
    logger.info(f"  Starting backprojection loop with {nx}×{ny}×{nz} voxels and {n_projections} projections")
    
    # 2. Loop over voxels in the rodent frame
    total_voxels = nx * ny * nz
    voxel_count = 0
    
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                voxel_count += 1
                
                # Log progress every 10% or every 100k voxels (whichever is less frequent)
                if voxel_count % max(total_voxels // 10, 100000) == 0:
                    progress_pct = 100.0 * voxel_count / total_voxels
                    logger.info(f"    Progress: {voxel_count:,}/{total_voxels:,} voxels ({progress_pct:.1f}%)")
                
                # Compute voxel center in world coordinates
                x = ox + (ix + 0.5) * sx
                y = oy + (iy + 0.5) * sy
                z = oz + (iz + 0.5) * sz
                point_world = np.array([x, y, z], dtype=np.float64)
                
                # 3. Loop over all projection angles
                for angle_idx in range(n_projections):
                    # Project voxel center to detector
                    try:
                        u, v = world_to_detector(point_world, angle_idx, geom)
                    except ValueError:
                        # Ray parallel to detector plane; skip
                        continue
                    
                    # Check if (u, v) is within detector bounds
                    if 0 <= u < nu and 0 <= v < nv:
                        # Sample projection via bilinear interpolation
                        value = bilinear_interpolate(projections[angle_idx], u, v)
                        volume[ix, iy, iz] += value
                        accumulator[ix, iy, iz] += 1.0
    
    logger.info(f"    Progress: {total_voxels:,}/{total_voxels:,} voxels (100.0%)")
    
    # 4. Normalization: divide by number of contributing projections
    # Avoid division by zero
    mask = accumulator > 0
    volume[mask] /= accumulator[mask]
    
    logger.info(f"  Backprojection normalization complete")
    
    # 5. Return reconstructed volume
    return volume
