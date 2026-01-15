"""
Add a 1D ramp-filter preprocessing step for cone-beam CT projections
before calling the existing backproject() function.

Context:
- `projections` is a NumPy array with shape (n_projections, nv, nu)
  where:
    n_projections = number of views
    nv = detector rows (v direction)
    nu = detector columns (u direction, filter axis)
- The current pipeline:
    1) load projections (optionally log-transform, flat-field, etc.)
    2) call backproject(projections, geom, volume_shape, voxel_spacing, origin)

Goal:
- Insert a function `apply_ramp_filter(projections, du) -> np.ndarray`
  that returns filtered projections suitable for FDK-style reconstruction.

Requirements for apply_ramp_filter:

1. Signature and purpose
   - Implement:

        def apply_ramp_filter(projections: np.ndarray, du: float) -> np.ndarray:
            \"\"\"Apply a 1D ramp filter along the detector u-axis to all projections.

            Parameters
            ----------
            projections : array, shape (n_proj, nv, nu)
                Input projections (already log-transformed if needed).
            du : float
                Detector pixel size along u in mm.

            Returns
            -------
            filtered : array, shape (n_proj, nv, nu)
                Ramp-filtered projections.
            \"\"\"

2. Filter design (Ram–Lak, basic version)
   - Work in the frequency domain along the last axis (u):
       * For each row along v and each projection:
           - Take FFT along axis -1.
           - Multiply by the ramp filter in frequency domain.
           - Inverse FFT and keep the real part.
   - Build the ramp filter once and reuse it:
       * nu = projections.shape[-1]
       * frequency sampling: use np.fft.fftfreq(nu, d=du)
       * ramp magnitude: |f|  (absolute value of spatial frequency)
       * Optionally include a Hamming or Hann window to reduce ringing.

   - Example outline inside the function:
       * nu = projections.shape[-1]
       * freqs = np.fft.fftfreq(nu, d=du)
       * ramp = np.abs(freqs)
       * Optionally: ramp *= np.hanning(nu)  # windowed ramp
       * Then:
           - F = np.fft.fft(projections, axis=-1)
           - F_filtered = F * ramp  # broadcast along projection and v dimensions
           - filtered = np.fft.ifft(F_filtered, axis=-1).real

3. Performance and clarity
   - Vectorize over (n_proj, nv); avoid Python loops over projections/rows.
   - Keep code readable, with comments referencing:
       - “FFT along u”
       - “multiply by ramp |f|”
       - “IFFT back to spatial domain”

4. Integration with existing pipeline
   - Add a helper function `ramp_filter_and_backproject(...)` that:
       * calls `apply_ramp_filter` on the input projections
       * passes the filtered projections into the existing `backproject` function
       * returns the reconstructed volume.

   - Example:

        def ramp_filter_and_backproject(
            projections: np.ndarray,
            geom: Geometry,
            volume_shape: tuple[int, int, int],
            voxel_spacing: tuple[float, float, float],
            volume_origin: tuple[float, float, float],
        ) -> np.ndarray:
            du = geom.det_pixel_size[0]
            proj_filt = apply_ramp_filter(projections, du)
            return backproject(proj_filt, geom, volume_shape, voxel_spacing, volume_origin)

Please implement `apply_ramp_filter` and `ramp_filter_and_backproject` exactly as described.
"""

from typing import Tuple, Optional

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

from backproject import backproject


def apply_ramp_filter(projections: np.ndarray, du: float, use_gpu: Optional[bool] = None) -> np.ndarray:
    """Apply a 1D ramp filter along the detector u-axis to all projections.

    This function will use CuPy FFT when `use_gpu=True` and CuPy is available.
    If `use_gpu` is None the function will use CuPy when available.

    Parameters
    ----------
    projections : array, shape (n_proj, nu, nv)
        Input projections (already log-transformed if needed). The function
        assumes the detector u-axis is the first spatial axis after the
        projection index (shape: n_projections, nu, nv).
    du : float
        Detector pixel size along u in mm.
    use_gpu : bool or None
        If True, attempt to use CuPy for FFTs. If False, force NumPy.
        If None (default), use CuPy when available.

    Returns
    -------
    filtered : array, shape (n_proj, nu, nv)
        Ramp-filtered projections. Returned type is NumPy ndarray when
        `use_gpu` is False or CuPy is unavailable; otherwise returns a
        CuPy array if `use_gpu` is True.
    """

    if projections.ndim != 3:
        raise ValueError("projections must be a 3D array with shape (n_proj, nu, nv)")

    # Decide whether to use GPU
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE

    # Axis 1 is detector u-axis (nu)
    n_proj, nu, nv = projections.shape

    if use_gpu and CUPY_AVAILABLE:
        xp = cp

        # Prepare GPU-side filter arrays once
        freqs = xp.fft.fftfreq(nu, d=du)
        ramp = xp.abs(freqs)
        window = xp.hamming(nu)
        ramp_windowed = ramp * window

        # Estimate available GPU memory and process in batches to avoid OOM
        try:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        except Exception:
            free_bytes = None

        # Conservative estimate: complex64 (8 bytes) per element and extra overhead
        bytes_per_elem = 8
        overhead_factor = 3  # account for temporaries

        per_proj_bytes = int(nu * nv * bytes_per_elem * overhead_factor)

        if free_bytes is None or per_proj_bytes == 0:
            batch_size = 1
        else:
            batch_size = max(1, min(n_proj, int(free_bytes / per_proj_bytes)))

        # Ensure at least one projection per batch
        batch_size = max(1, batch_size)

        out_chunks = []
        try:
            for i in range(0, n_proj, batch_size):
                j = min(n_proj, i + batch_size)
                proj_gpu_batch = cp.asarray(projections[i:j], dtype=cp.float32)

                # FFT along u (axis=1)
                F = xp.fft.fft(proj_gpu_batch, axis=1)
                F_filtered = F * ramp_windowed.reshape(1, nu, 1)
                filtered_gpu = xp.fft.ifft(F_filtered, axis=1).real

                out_chunks.append(filtered_gpu)

            if len(out_chunks) == 1:
                return out_chunks[0]
            else:
                return xp.concatenate(out_chunks, axis=0)
        except cp.cuda.memory.OutOfMemoryError:
            # Fallback: perform filtering on CPU to avoid repeated OOMs
            # Convert to NumPy and reuse CPU implementation below
            projections = np.asarray(projections, dtype=np.float32)
            use_gpu = False
            # fall through to CPU branch
    else:
        xp = np
        proj_np = np.asarray(projections, dtype=np.float32)

        freqs = np.fft.fftfreq(nu, d=du)
        ramp = np.abs(freqs)
        window = np.hamming(nu)
        ramp_windowed = ramp * window

        F = np.fft.fft(proj_np, axis=1)
        F_filtered = F * ramp_windowed.reshape(1, nu, 1)
        filtered = np.fft.ifft(F_filtered, axis=1).real

        return filtered


def ramp_filter_and_backproject(
    projections: np.ndarray,
    geom,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    volume_origin: Tuple[float, float, float],
    use_gpu: Optional[bool] = None,
    logger=None,
) -> np.ndarray:
    """Apply ramp filter to projections and call `backproject`.

    Parameters mirror those expected by `backproject`; `geom` must provide
    detector pixel size as `det_pixel_size` (tuple-like) where element 0
    is the u-axis pixel size in mm.
    """

    # Extract detector u pixel size from geometry
    try:
        du = float(geom.det_pixel_size[0])
    except Exception as e:
        raise ValueError("Could not read detector pixel size from geom.det_pixel_size") from e

    proj_filt = apply_ramp_filter(projections, du, use_gpu=use_gpu)

    # If GPU filtered data (CuPy) but backproject expects NumPy (use_gpu False),
    # convert to NumPy. If filtered is CuPy and backproject_gpu will consume it
    # on GPU it's fine to pass the CuPy array along.
    if CUPY_AVAILABLE and use_gpu:
        # Return a CuPy array to allow GPU backprojection to reuse it.
        return backproject(proj_filt, geom, volume_shape, voxel_spacing, volume_origin, logger=logger)
    else:
        return backproject(proj_filt, geom, volume_shape, voxel_spacing, volume_origin, logger=logger)

