"""
GPU-accelerated versions of the projector and backprojector for cone-beam CT.

Uses CuPy for vectorized GPU operations on NVIDIA GPUs.
Falls back to CPU (NumPy) if CUDA is not available.
"""

from typing import Tuple, Union
import logging

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np


def quaternion_to_rotation_matrix_gpu(q):
    """
    Convert quaternion to rotation matrix on GPU.
    
    Parameters
    ----------
    q : GPUArray[4,] or ndarray[4,]
        Quaternion as (x, y, z, w).
    
    Returns
    -------
    GPUArray[3, 3] or ndarray[3, 3]
        Rotation matrix.
    """
    xp = cp.get_array_module(q)
    
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    R = xp.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])
    
    return R


def rodrigues_rotation_gpu(points, axis, theta, xp):
    """
    Apply Rodrigues' rotation formula to points on GPU.
    
    Parameters
    ----------
    points : GPUArray[N, 3] or ndarray[N, 3]
        Points to rotate.
    axis : GPUArray[3,] or ndarray[3,]
        Rotation axis (normalized).
    theta : float
        Rotation angle in radians.
    xp : module
        NumPy or CuPy module.
    
    Returns
    -------
    GPUArray[N, 3] or ndarray[N, 3]
        Rotated points.
    """
    cos_theta = xp.cos(theta)
    sin_theta = xp.sin(theta)
    
    # Rodrigues: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    # k is the normalized axis
    axis = axis / xp.linalg.norm(axis)
    
    # Cross product: k × v
    cross = xp.cross(xp.tile(axis, (points.shape[0], 1)), points, axisb=1)
    
    # Dot product: k · v
    dot = xp.sum(axis[xp.newaxis, :] * points, axis=1, keepdims=True)
    
    # Apply Rodrigues formula
    rotated = (
        points * cos_theta +
        cross * sin_theta +
        axis[xp.newaxis, :] * dot * (1 - cos_theta)
    )
    
    return rotated


def world_to_detector_batch(
    points_world,
    angle_idx: int,
    geom,
) -> Tuple:
    """
    Batch GPU-accelerated mapping of N 3D world points to detector (u, v) at one angle.
    
    Parameters
    ----------
    points_world : GPUArray[N, 3] or ndarray[N, 3]
        N 3D points in world coordinates [x, y, z] in mm.
    angle_idx : int
        Index into the angle array.
    geom : Geometry
        Geometry object with all SARRP parameters.
    
    Returns
    -------
    tuple[GPUArray[N,], GPUArray[N,]] or tuple[ndarray[N,], ndarray[N,]]
        Detector pixel coordinates (u, v) for each input point.
    """
    xp = cp.get_array_module(points_world)
    
    # 1. Translate to COR
    COR = xp.array(geom.COR, dtype=xp.float32)
    p = points_world - COR[xp.newaxis, :]
    
    # 2. Rotate into gantry frame
    theta = float(geom.angles_rad[angle_idx])
    axis = xp.array(geom.rotation_axis, dtype=xp.float32)
    p_rotated = rodrigues_rotation_gpu(p, axis, theta, xp)
    
    # 3. Apply CTilt (quaternion rotation)
    CTilt = xp.array(geom.CTilt, dtype=xp.float32)
    R_ctilt = quaternion_to_rotation_matrix_gpu(CTilt)
    p_rotated = (R_ctilt @ p_rotated.T).T
    
    # 4. Ray-plane intersection
    source = xp.array([0.0, 0.0, -geom.SOD], dtype=xp.float32)
    detector_z = xp.float32(geom.IDD)
    
    direction = p_rotated - source[xp.newaxis, :]
    
    # Solve: source[2] + t * direction[2] = detector_z
    t = (detector_z - source[2]) / direction[:, 2]
    
    # Intersection point on detector plane
    p_det_view = source[xp.newaxis, :] + t[:, xp.newaxis] * direction
    
    # 5. Convert to pixel coordinates
    PP = xp.array([geom.det_nu / 2.0, geom.det_nv / 2.0], dtype=xp.float32)
    pixel_size = xp.array(geom.det_pixel_size, dtype=xp.float32)
    
    u = PP[0] + p_det_view[:, 0] / pixel_size[0]
    v = PP[1] + p_det_view[:, 1] / pixel_size[1]
    
    return u, v


def bilinear_interpolate_gpu(image_gpu, u_gpu, v_gpu):
    """
    Vectorized bilinear interpolation on GPU.
    
    Parameters
    ----------
    image_gpu : GPUArray[nu, nv] or ndarray[nu, nv]
        2D detector image.
    u_gpu : GPUArray[N,] or ndarray[N,]
        Horizontal pixel coordinates.
    v_gpu : GPUArray[N,] or ndarray[N,]
        Vertical pixel coordinates.
    
    Returns
    -------
    GPUArray[N,] or ndarray[N,]
        Interpolated values at (u, v) points.
    """
    xp = cp.get_array_module(image_gpu)
    
    nu, nv = image_gpu.shape
    
    # Clamp coordinates to valid range
    u_clamped = xp.clip(u_gpu, 0, nu - 1.001)
    v_clamped = xp.clip(v_gpu, 0, nv - 1.001)
    
    # Integer and fractional parts
    u_int = xp.floor(u_clamped).astype(xp.int32)
    v_int = xp.floor(v_clamped).astype(xp.int32)
    u_frac = u_clamped - u_int
    v_frac = v_clamped - v_int
    
    # Clamp integer indices
    u_int = xp.clip(u_int, 0, nu - 2)
    v_int = xp.clip(v_int, 0, nv - 2)
    
    # Bilinear interpolation
    val00 = image_gpu[u_int, v_int]
    val10 = image_gpu[u_int + 1, v_int]
    val01 = image_gpu[u_int, v_int + 1]
    val11 = image_gpu[u_int + 1, v_int + 1]
    
    val0 = val00 * (1 - u_frac) + val10 * u_frac
    val1 = val01 * (1 - u_frac) + val11 * u_frac
    
    return val0 * (1 - v_frac) + val1 * v_frac


def backproject_gpu(
    projections,
    geom,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    volume_origin: Tuple[float, float, float],
    use_gpu: bool = True,
    logger=None,
) -> np.ndarray:
    """
    GPU-accelerated voxel-driven backprojection.
    
    Parameters
    ----------
    projections : ndarray[n_proj, nu, nv]
        Projection stack (CPU or GPU array).
    geom : Geometry
        Geometry object.
    volume_shape : tuple[int, int, int]
        Output volume shape (nx, ny, nz).
    voxel_spacing : tuple[float, float, float]
        Voxel spacing in mm.
    volume_origin : tuple[float, float, float]
        Volume origin in mm.
    use_gpu : bool, optional
        If True, use GPU. If False, use CPU. Default True.
    logger : logging.Logger, optional
        Logger for progress reporting. If None, no logging.
    
    Returns
    -------
    ndarray[nx, ny, nz]
        Reconstructed volume on CPU.
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Select array module
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np
    
    # Move projections to GPU if needed
    if use_gpu and CUPY_AVAILABLE:
        projections = cp.asarray(projections, dtype=cp.float32)
    else:
        projections = np.asarray(projections, dtype=np.float32)
    
    nx, ny, nz = volume_shape
    sx, sy, sz = voxel_spacing
    ox, oy, oz = volume_origin
    n_projections, nu, nv = projections.shape
    
    logger.info(f"  Starting GPU backprojection with {nx}×{ny}×{nz} voxels and {n_projections} projections")
    
    # Allocate volume and accumulator on GPU
    volume = xp.zeros(volume_shape, dtype=xp.float32)
    accumulator = xp.zeros(volume_shape, dtype=xp.float32)
    
    # Create 3D grid of voxel centers
    x = xp.arange(nx, dtype=xp.float32) * sx + ox + sx / 2
    y = xp.arange(ny, dtype=xp.float32) * sy + oy + sy / 2
    z = xp.arange(nz, dtype=xp.float32) * sz + oz + sz / 2
    
    xx, yy, zz = xp.meshgrid(x, y, z, indexing='ij')
    
    # Reshape to (N, 3) where N = nx*ny*nz
    points_world = xp.stack(
        [xx.ravel(), yy.ravel(), zz.ravel()],
        axis=1
    ).astype(xp.float32)
    
    # Process each angle
    for angle_idx in range(n_projections):
        # Log progress every 10% or at least every 5 projections
        if (angle_idx + 1) % max(max(n_projections // 10, 1), 5) == 0 or angle_idx == 0:
            progress_pct = 100.0 * (angle_idx + 1) / n_projections
            logger.info(f"    Progress: {angle_idx + 1}/{n_projections} projections ({progress_pct:.1f}%)")
        
        # Project all voxel centers to detector
        u, v = world_to_detector_batch(points_world, angle_idx, geom)
        
        # Check bounds
        in_bounds = (u >= 0) & (u < nu) & (v >= 0) & (v < nv)
        
        # Sample projection at valid (u, v)
        sample_values = bilinear_interpolate_gpu(projections[angle_idx], u, v)
        sample_values = sample_values * in_bounds.astype(xp.float32)
        
        # Accumulate into volume
        volume_flat = volume.ravel()
        accumulator_flat = accumulator.ravel()
        
        volume_flat += sample_values
        accumulator_flat += in_bounds.astype(xp.float32)
        
        volume = volume_flat.reshape(volume_shape)
        accumulator = accumulator_flat.reshape(volume_shape)
    
    logger.info(f"    Progress: {n_projections}/{n_projections} projections (100.0%)")
    logger.info(f"  Normalizing volume...")
    
    # Normalize
    mask = accumulator > 0
    volume[mask] /= accumulator[mask]
    
    logger.info(f"  GPU backprojection normalization complete")
    
    # Move back to CPU
    if use_gpu and CUPY_AVAILABLE:
        volume = cp.asnumpy(volume)
    
    return volume.astype(np.float32)
