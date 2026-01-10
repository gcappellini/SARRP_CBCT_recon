"""
Main script to run full SARRP CBCT reconstruction on GPU.

Loads real projection data and reconstructs a 3D volume using GPU-accelerated backprojection.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml

from geometry import load_geometry_from_yaml
from gpu_ops import backproject_gpu, CUPY_AVAILABLE


def load_projections_mhd(mhd_path: str) -> np.ndarray:
    """
    Load projections from MHD/RAW file using SimpleITK.
    
    Parameters
    ----------
    mhd_path : str
        Path to the .mhd file.
    
    Returns
    -------
    np.ndarray
        Projections with shape (n_projections, nu, nv).
    """
    print(f"Loading projections from: {mhd_path}")
    img = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(img)
    
    # MHD format is typically (z, y, x) -> (n_proj, nv, nu)
    # We need (n_proj, nu, nv)
    print(f"Raw shape from MHD: {arr.shape}")
    
    # Transpose if needed based on typical MHD ordering
    if arr.ndim == 3:
        # Assume order is (n_proj, nv, nu), need (n_proj, nu, nv)
        arr = np.transpose(arr, (0, 2, 1))
    
    print(f"Projections shape after reordering: {arr.shape}")
    print(f"Projection data type: {arr.dtype}")
    print(f"Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")
    
    return arr


def save_volume_mhd(volume: np.ndarray, output_path: str, spacing: tuple, origin: tuple):
    """
    Save volume as MHD/RAW using SimpleITK.
    
    Parameters
    ----------
    volume : np.ndarray
        3D volume array.
    output_path : str
        Output path for .mhd file.
    spacing : tuple
        Voxel spacing (sx, sy, sz).
    origin : tuple
        Volume origin (ox, oy, oz).
    """
    img = sitk.GetImageFromArray(volume.transpose(2, 1, 0))  # ITK uses (z, y, x)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, output_path)
    print(f"Saved volume to: {output_path}")


def save_slices(volume: np.ndarray, output_dir: Path, prefix: str = "recon"):
    """
    Save central slices as PNG images.
    
    Parameters
    ----------
    volume : np.ndarray
        3D volume.
    output_dir : Path
        Output directory.
    prefix : str
        Prefix for output files.
    """
    nx, ny, nz = volume.shape
    
    # Normalize for visualization
    vmin, vmax = np.percentile(volume, [1, 99])
    volume_norm = np.clip((volume - vmin) / (vmax - vmin), 0, 1)
    
    # Axial slice (xy plane at center z)
    slice_axial = volume_norm[:, :, nz // 2]
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_axial.T, origin="lower", cmap="gray")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Axial slice (z = {nz // 2})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(output_dir / f"{prefix}_axial.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_axial.png")
    
    # Coronal slice (xz plane at center y)
    slice_coronal = volume_norm[:, ny // 2, :]
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_coronal.T, origin="lower", cmap="gray")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Coronal slice (y = {ny // 2})")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig(output_dir / f"{prefix}_coronal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_coronal.png")
    
    # Sagittal slice (yz plane at center x)
    slice_sagittal = volume_norm[nx // 2, :, :]
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_sagittal.T, origin="lower", cmap="gray")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Sagittal slice (x = {nx // 2})")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.savefig(output_dir / f"{prefix}_sagittal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_sagittal.png")


def main():
    """Main reconstruction pipeline."""
    
    parser = argparse.ArgumentParser(
        description="SARRP CBCT GPU reconstruction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--projections",
        type=str,
        default="projections.mhd",
        help="Path to projections MHD file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("SARRP CBCT GPU Reconstruction")
    print("=" * 80)
    
    # Check GPU availability
    use_gpu = not args.no_gpu and CUPY_AVAILABLE
    if use_gpu:
        print("✓ GPU acceleration enabled (CuPy)")
    else:
        print("⚠ Using CPU (GPU not available or disabled)")
    
    print()
    
    # 1. Load geometry and configuration
    print("Step 1: Loading geometry...")
    geom = load_geometry_from_yaml(args.config)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract volume parameters
    voxel_spacing = tuple(map(float, config["voxel spacing (mm)"].split()))
    volume_dims = tuple(map(int, config["volume dimensions (voxels)"].split()))
    
    # Calculate volume origin to center it around isocenter
    volume_origin = (
        -volume_dims[0] * voxel_spacing[0] / 2,
        -volume_dims[1] * voxel_spacing[1] / 2,
        -volume_dims[2] * voxel_spacing[2] / 2,
    )
    
    print(f"  Geometry: SOD={geom.SOD:.2f} mm, SDD={geom.SDD:.2f} mm, IDD={geom.IDD:.2f} mm")
    print(f"  Detector: {geom.det_nu} × {geom.det_nv} pixels")
    print(f"  Projections: {geom.n_projections} angles from {geom.angle_start_deg}° to {geom.angle_end_deg}°")
    print(f"  Volume: {volume_dims} voxels, spacing={voxel_spacing} mm")
    print(f"  Volume origin: {volume_origin} mm")
    print()
    
    # 2. Load projections
    print("Step 2: Loading projections...")
    projections = load_projections_mhd(args.projections)
    print()
    
    # Verify dimensions match
    n_proj, nu, nv = projections.shape
    if n_proj != geom.n_projections:
        print(f"⚠ Warning: Number of projections in data ({n_proj}) != config ({geom.n_projections})")
    if nu != geom.det_nu or nv != geom.det_nv:
        print(f"⚠ Warning: Detector size mismatch: data=({nu}, {nv}), config=({geom.det_nu}, {geom.det_nv})")
    
    # 3. Run reconstruction
    print("Step 3: Running backprojection...")
    print(f"  Processing {n_proj} projections...")
    print(f"  Reconstructing volume of shape {volume_dims}...")
    
    t_start = time.time()
    
    volume = backproject_gpu(
        projections=projections,
        geom=geom,
        volume_shape=volume_dims,
        voxel_spacing=voxel_spacing,
        volume_origin=volume_origin,
        use_gpu=use_gpu,
    )
    
    t_elapsed = time.time() - t_start
    
    print(f"  ✓ Reconstruction complete in {t_elapsed:.2f} seconds")
    print()
    
    # 4. Print diagnostics
    print("Step 4: Volume statistics")
    print(f"  Shape: {volume.shape}")
    print(f"  Data type: {volume.dtype}")
    print(f"  Min:  {volume.min():.6f}")
    print(f"  Max:  {volume.max():.6f}")
    print(f"  Mean: {volume.mean():.6f}")
    print(f"  Std:  {volume.std():.6f}")
    
    # Find maximum voxel
    max_idx = np.unravel_index(np.argmax(volume), volume.shape)
    max_coords = (
        volume_origin[0] + (max_idx[0] + 0.5) * voxel_spacing[0],
        volume_origin[1] + (max_idx[1] + 0.5) * voxel_spacing[1],
        volume_origin[2] + (max_idx[2] + 0.5) * voxel_spacing[2],
    )
    print(f"  Max value location:")
    print(f"    Voxel index: {max_idx}")
    print(f"    World coords: ({max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f}) mm")
    print()
    
    # 5. Save outputs
    print("Step 5: Saving results...")
    
    # Save as NumPy
    np.save(output_dir / "recon_volume.npy", volume)
    print(f"  Saved: recon_volume.npy")
    
    # Save as MHD
    save_volume_mhd(
        volume,
        str(output_dir / "recon_volume.mhd"),
        spacing=voxel_spacing,
        origin=volume_origin,
    )
    
    # Save slices
    save_slices(volume, output_dir, prefix="recon")
    print()
    
    print("=" * 80)
    print(f"✓ Reconstruction complete! Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
