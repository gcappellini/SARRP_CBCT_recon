"""
Ramp filter preprocessing for cone-beam CT projections.
Apply 1D FFT-based ramp filtering along the detector u-axis before backprojection.
"""

from typing import Callable, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


def _compute_window(n: int, window_type: str, xp):
    """
    Compute window function for ramp filter in frequency domain.

    Parameters
    ----------
    n : int
        Number of samples (detector width).
    window_type : str
        Window type: 'ram-lak', 'shepp-logan', 'hann', 'hamming', 'cosine'.
    xp : module
        NumPy or CuPy module.

    Returns
    -------
    window : array[n,]
        Window values.
    """
    window_type = window_type.lower()
    if window_type in ['ram-lak', 'none']:
        return xp.ones(n, dtype=xp.float32)

    # Normalized frequency array [0, 1] at Nyquist
    freqs = xp.fft.fftfreq(n)
    omega = 2.0 * xp.abs(freqs)

    if window_type == 'shepp-logan':
        # sinc(ω/2) = sin(πω/2) / (πω/2)
        window = xp.sinc(omega / 2.0)
    elif window_type == 'cosine':
        window = xp.cos(omega * xp.pi / 2.0)
    elif window_type == 'hamming':
        window = 0.54 + 0.46 * xp.cos(omega * xp.pi)
    elif window_type == 'hann':
        window = (1.0 + xp.cos(omega * xp.pi)) / 2.0
    else:
        raise ValueError(f"Unknown window: {window_type}")

    return window.astype(xp.float32)


def apply_ramp_filter(
    projections: np.ndarray,
    du: float,
    use_gpu: Optional[bool] = None,
    window: str = 'shepp-logan',
    gaussian_sigma: float = 0.0
) -> np.ndarray:
    """
    Apply 1D ramp filter along detector u-axis to all projections.

    Parameters
    ----------
    projections : array, shape (n_proj, nv, nu)
        Input projections (log-transformed if needed).
        Detector u-axis is axis -1.
    du : float
        Detector pixel size along u in mm.
    use_gpu : bool or None
        If True, use CuPy FFT. If None, use CuPy if available.
    window : str
        Window type: 'ram-lak', 'shepp-logan', 'hann', 'hamming', 'cosine'.
    gaussian_sigma : float
        Gaussian pre-smoothing sigma in pixels. If > 0, applies Gaussian blur
        to each projection before ramp filtering. Values 0.5-1.0 recommended
        for edge-preserving noise reduction. Default: 0.0 (no smoothing).

    Returns
    -------
    filtered : array, shape (n_proj, nv, nu)
        Ramp-filtered projections.
    """
    if projections.ndim != 3:
        raise ValueError("projections must have shape (n_proj, nv, nu)")

    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE

    n_proj, nv, nu = projections.shape

    if n_proj == 0:
        return projections.copy()

    # --- Frequency-domain ramp filter ---
    freqs = np.fft.fftfreq(nu, d=du)
    ramp = np.abs(freqs)
    w = _compute_window(nu, window, np)
    ramp_windowed = (ramp * w * 2.0 * du).astype(np.float32)

    # --- CPU path (NumPy FFT) ---
    if not use_gpu or not CUPY_AVAILABLE:
        proj_np = np.asarray(projections, dtype=np.float32)
        
        # Optional Gaussian pre-smoothing for noise reduction
        if gaussian_sigma > 0:
            for i in range(n_proj):
                proj_np[i] = gaussian_filter(proj_np[i], sigma=gaussian_sigma)
        
        # FFT along u-axis (axis=-1)
        F = np.fft.fft(proj_np, axis=-1)
        # Multiply by ramp (broadcast over n_proj, nv dimensions)
        F_filtered = F * ramp_windowed[np.newaxis, np.newaxis, :]
        filtered = np.fft.ifft(F_filtered, axis=-1).real
        return filtered.astype(np.float32)

    # --- GPU path (CuPy FFT) ---
    ramp_gpu = cp.asarray(ramp_windowed, dtype=cp.float32)
    
    try:
        proj_gpu = cp.asarray(projections, dtype=cp.float32)
        
        # Optional Gaussian pre-smoothing for noise reduction
        if gaussian_sigma > 0:
            try:
                from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu
                for i in range(n_proj):
                    proj_gpu[i] = gaussian_filter_gpu(proj_gpu[i], sigma=gaussian_sigma)
            except ImportError:
                # Fallback to CPU smoothing if cupyx not available
                proj_cpu = cp.asnumpy(proj_gpu)
                for i in range(n_proj):
                    proj_cpu[i] = gaussian_filter(proj_cpu[i], sigma=gaussian_sigma)
                proj_gpu = cp.asarray(proj_cpu, dtype=cp.float32)
        
        F = cp.fft.fft(proj_gpu, axis=-1)
        F_filtered = F * ramp_gpu[cp.newaxis, cp.newaxis, :]
        filtered = cp.fft.ifft(F_filtered, axis=-1).real
        return cp.asnumpy(filtered.astype(cp.float32))
    except cp.cuda.memory.OutOfMemoryError:
        # Fallback to CPU
        proj_np = np.asarray(projections, dtype=np.float32)
        
        # Optional Gaussian pre-smoothing for noise reduction
        if gaussian_sigma > 0:
            for i in range(n_proj):
                proj_np[i] = gaussian_filter(proj_np[i], sigma=gaussian_sigma)
        
        F = np.fft.fft(proj_np, axis=-1)
        F_filtered = F * ramp_windowed[np.newaxis, np.newaxis, :]
        filtered = np.fft.ifft(F_filtered, axis=-1).real
        return filtered.astype(np.float32)


def make_cpu_projection_provider(
    projections: np.ndarray,
    du: float,
    window: str = 'shepp-logan',
    gaussian_sigma: float = 0.0
) -> Callable:
    """
    Create a per-angle CPU projection provider.

    Returns a callable `get_projection(angle_idx)` that applies the ramp filter
    to a single projection on demand.

    Parameters
    ----------
    projections : array, shape (n_proj, nv, nu)
        Raw projection stack.
    du : float
        Detector pixel size along u in mm.
    window : str
        Window type.
    gaussian_sigma : float
        Gaussian pre-smoothing sigma in pixels (0.0 = no smoothing).

    Returns
    -------
    get_projection : callable
        Function(angle_idx: int) -> array (nv, nu) filtered projection.
    """
    if projections.ndim != 3:
        raise ValueError("projections must have shape (n_proj, nv, nu)")

    n_proj, nv, nu = projections.shape

    # Precompute frequency filter
    freqs = np.fft.fftfreq(nu, d=du)
    ramp = np.abs(freqs)
    w = _compute_window(nu, window, np)
    ramp_windowed = (ramp * w * 2.0 * du).astype(np.float32)

    def get_projection(angle_idx: int) -> np.ndarray:
        if angle_idx < 0 or angle_idx >= n_proj:
            raise IndexError(f"angle_idx {angle_idx} out of range [0, {n_proj})")
        
        proj = np.asarray(projections[angle_idx], dtype=np.float32)  # shape: (nv, nu)
        
        # Optional Gaussian pre-smoothing for noise reduction
        if gaussian_sigma > 0:
            proj = gaussian_filter(proj, sigma=gaussian_sigma)
        
        # FFT along u-axis (axis=-1)
        F = np.fft.fft(proj, axis=-1)
        F_filtered = F * ramp_windowed[np.newaxis, :]
        filtered = np.fft.ifft(F_filtered, axis=-1).real
        return filtered.astype(np.float32)

    return get_projection


def ramp_filter_and_backproject(
    projections: np.ndarray,
    geom,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    volume_origin: Tuple[float, float, float],
    window: str = 'shepp-logan',
    gaussian_sigma: float = 0.0
) -> np.ndarray:
    """
    Apply ramp filter to projections and backproject.

    Parameters
    ----------
    projections : array, shape (n_proj, nv, nu)
        Input projections.
    geom : Geometry
        Geometry object with det_pixel_size attribute.
    volume_shape : tuple
        (nx, ny, nz)
    voxel_spacing : tuple
        (dx, dy, dz)
    volume_origin : tuple
        (ox, oy, oz)
    window : str
        Window type for ramp filter.
    gaussian_sigma : float
        Gaussian pre-smoothing sigma in pixels (0.0 = no smoothing).

    Returns
    -------
    volume : array, shape volume_shape
        Reconstructed volume.
    """
    from backproject import backproject

    du = float(geom.det_pixel_size[0])
    proj_filt = apply_ramp_filter(projections, du, use_gpu=False, window=window, gaussian_sigma=gaussian_sigma)
    return backproject(proj_filt, geom, volume_shape, voxel_spacing, volume_origin)
