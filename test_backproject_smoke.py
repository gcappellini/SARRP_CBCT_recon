"""
Lightweight smoke test for the cone-beam backprojector using a delta projection pattern.

This test creates synthetic projections with a single non-zero pixel and backprojects
them into a 3D volume to verify the geometry is working correctly.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from matplotlib import pyplot as plt

from backproject import backproject
from geometry import Geometry
from datetime import datetime


def write_test_config(path: str) -> None:
    """
    Write a minimal YAML configuration for testing.
    
    Parameters
    ----------
    path : str
        Path to the output YAML file.
    """
    config = {
        "source position relative to isocenter (mm)": "0 0 -353.266",
        "detector reference point (mm)": "102.4 102.4 263.609",
        "detector pixel counts": "256 256",
        "physical detector size (mm)": "204.8 204.8",
        "angular start (degrees)": 0,
        "angular end (degrees)": 360,
        "number of projections": 60,
        "CTilt": "0 0 -0.708352 0.705859",
        "COR offset (mm)": "-1.032 -2.285 0",
        "rotation axis direction": "0 1 0",
        "Source-to-Detector Distance SDD (mm)": 616.875,
        "Source-to-Isocenter Distance SID (mm)": 353.266,
        "Isocenter-to-Detector Distance IDD (mm)": 263.609,
        "volume": {
            "shape": [64, 64, 64],
            "voxel_spacing": [0.4, 0.4, 0.4],
            "origin": [-12.8, -12.8, -12.8],
        },
    }
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_test_geometry(conf_path: str) -> Geometry:
    """
    Load geometry parameters from a test YAML configuration file.
    
    Parameters
    ----------
    conf_path : str
        Path to the config_test.yaml file.
    
    Returns
    -------
    Geometry
        Geometry instance with all parameters loaded.
    """
    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Parse source position
    source_pos = tuple(
        map(float, config["source position relative to isocenter (mm)"].split())
    )
    
    # Parse detector reference point
    detector_ref = tuple(
        map(float, config["detector reference point (mm)"].split())
    )
    
    # Parse detector pixel counts
    det_nu, det_nv = map(int, config["detector pixel counts"].split())
    
    # Parse physical detector size
    det_size = tuple(
        map(float, config["physical detector size (mm)"].split())
    )
    
    # Parse COR offset
    COR = tuple(map(float, config["COR offset (mm)"].split()))
    
    # Parse CTilt (quaternion)
    CTilt = tuple(map(float, config["CTilt"].split()))
    
    # Parse rotation axis direction
    rotation_axis = tuple(map(float, config["rotation axis direction"].split()))
    
    # Parse angular parameters
    angle_start_deg = float(config["angular start (degrees)"])
    angle_end_deg = float(config["angular end (degrees)"])
    n_projections = int(config["number of projections"])
    
    # Parse distances
    SDD = float(config["Source-to-Detector Distance SDD (mm)"])
    SOD = float(config["Source-to-Isocenter Distance SID (mm)"])
    IDD = float(config["Isocenter-to-Detector Distance IDD (mm)"])
    
    return Geometry(
        SOD=SOD,
        SDD=SDD,
        IDD=IDD,
        source_pos=source_pos,
        detector_ref=detector_ref,
        COR=COR,
        CTilt=CTilt,
        det_nu=det_nu,
        det_nv=det_nv,
        det_size=det_size,
        rotation_axis=rotation_axis,
        angle_start_deg=angle_start_deg,
        angle_end_deg=angle_end_deg,
        n_projections=n_projections,
    )


def make_delta_projections(geom: Geometry, u0: int, v0: int) -> np.ndarray:
    """
    Create synthetic projection array with a single non-zero pixel.
    
    Parameters
    ----------
    geom : Geometry
        Geometry object with projection parameters.
    u0 : int
        Horizontal detector pixel coordinate for the delta.
    v0 : int
        Vertical detector pixel coordinate for the delta.
    
    Returns
    -------
    np.ndarray
        Projection stack of shape (n_projections, nu, nv) with a single
        pixel set to 1.0 in projection 0.
    """
    n_proj = geom.n_projections
    projections = np.zeros((n_proj, geom.det_nu, geom.det_nv), dtype=np.float32)
    
    # Set a single pixel in the first projection
    proj_idx = 0
    projections[proj_idx, u0, v0] = 1.0
    
    return projections


def run_backproject_smoke_test():
    """
    Run a lightweight smoke test of the backprojector on a delta pattern.
    
    Creates a synthetic projection with a single non-zero pixel, backprojects it,
    and saves the results for visual inspection.
    """
    # Step 1: Create test configuration
    config_path = "config_test.yaml"
    print(f"Creating test configuration: {config_path}")
    write_test_config(config_path)
    
    # Step 2: Load geometry
    print("Loading test geometry...")
    geom = load_test_geometry(config_path)
    
    # Step 3: Load volume parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    volume_shape = tuple(config["volume"]["shape"])
    voxel_spacing = tuple(config["volume"]["voxel_spacing"])
    volume_origin = tuple(config["volume"]["origin"])
    
    print(f"Volume shape: {volume_shape}")
    print(f"Voxel spacing: {voxel_spacing} mm")
    print(f"Volume origin: {volume_origin} mm")
    
    # Step 4: Create synthetic delta projections
    u0 = geom.det_nu // 2
    v0 = geom.det_nv // 2
    print(f"\nCreating delta projections at detector pixel ({u0}, {v0})...")
    projections = make_delta_projections(geom, u0, v0)
    print(f"Projections shape: {projections.shape}")
    
    # Step 5: Run backprojection
    print("\nRunning backprojection...")
    volume = backproject(
        projections,
        geom,
        volume_shape,
        voxel_spacing,
        volume_origin,
    )
    print(f"Reconstructed volume shape: {volume.shape}")
    
    # Step 6: Output results
    print("\n=== Volume Statistics ===")
    print(f"Min:  {volume.min():.6f}")
    print(f"Max:  {volume.max():.6f}")
    print(f"Mean: {volume.mean():.6f}")
    print(f"Std:  {volume.std():.6f}")
    
    # Find voxel with maximum value
    max_idx = np.unravel_index(np.argmax(volume), volume.shape)
    max_coords = (
        volume_origin[0] + (max_idx[0] + 0.5) * voxel_spacing[0],
        volume_origin[1] + (max_idx[1] + 0.5) * voxel_spacing[1],
        volume_origin[2] + (max_idx[2] + 0.5) * voxel_spacing[2],
    )
    print(f"\nMax value location:")
    print(f"  Voxel index: {max_idx}")
    print(f"  World coords: ({max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f}) mm")
    
    # Save full volume
    output_path = "smoke_test/smoke_volume.npy"
    np.save(output_path, volume)
    print(f"\nSaved volume to: {output_path}")
    
    # Extract and save slices through the maximum voxel
    nx, ny, nz = volume.shape
    max_idx = np.unravel_index(np.argmax(volume), volume.shape)
    
    # Normalize for visualization
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume_norm = (volume - vmin) / (vmax - vmin)
    else:
        volume_norm = volume
    
    # Axial slice (xy plane) at max voxel z-coordinate
    iz_max = max_idx[2]
    slice_axial = volume_norm[:, :, iz_max]
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_axial.T, origin="lower", cmap="gray")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Axial slice at max voxel (z = {iz_max}, world z = {volume_origin[2] + (iz_max + 0.5) * voxel_spacing[2]:.2f} mm)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("smoke_test/smoke_axial.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: smoke_axial.png")
    
    # Coronal slice (xz plane) at max voxel y-coordinate
    iy_max = max_idx[1]
    slice_coronal = volume_norm[:, iy_max, :]
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_coronal.T, origin="lower", cmap="gray")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Coronal slice at max voxel (y = {iy_max}, world y = {volume_origin[1] + (iy_max + 0.5) * voxel_spacing[1]:.2f} mm)")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig("smoke_test/smoke_coronal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: smoke_coronal.png")
    
    # Sagittal slice (yz plane) at max voxel x-coordinate
    ix_max = max_idx[0]
    slice_sagittal = volume_norm[ix_max, :, :]
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_sagittal.T, origin="lower", cmap="gray")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Sagittal slice at max voxel (x = {ix_max}, world x = {volume_origin[0] + (ix_max + 0.5) * voxel_spacing[0]:.2f} mm)")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.savefig("smoke_test/smoke_sagittal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: smoke_sagittal.png")
    
    print("\n=== Smoke test complete! ===")
    print("Check the PNG files to verify the backprojection geometry.")


if __name__ == "__main__":
    start_time = datetime.now()
    run_backproject_smoke_test()
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed}")