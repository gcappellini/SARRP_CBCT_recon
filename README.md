# SARRP CBCT Reconstruction

GPU-accelerated cone-beam CT reconstruction for Small Animal Radiation Research Platform (SARRP).

## Features

- **Geometry handling**: Full SARRP geometry specification from config files
- **CPU implementation**: Reference implementation with NumPy
- **GPU acceleration**: CuPy-based GPU implementation for ~50-100x speedup
- **Voxel-driven backprojection**: Efficient reconstruction algorithm
- **Bilinear interpolation**: Sub-pixel accuracy for detector sampling

## Requirements

```bash
pip install numpy scipy pyyaml matplotlib SimpleITK cupy-cuda12x  # for CUDA 12.x
# OR
pip install numpy scipy pyyaml matplotlib SimpleITK cupy-cuda11x  # for CUDA 11.x
```

## Data Files

**Note**: The projection data files (`projections.raw`, `projections.mhd`) are not included in this repository due to their large size (5.6 GB total, exceeds GitHub LFS 2GB limit). 

To use this code:
1. Obtain the SARRP projection data from your institution or collaborators
2. Place `projections.mhd` and `projections.raw` in the repository root
3. Ensure the `conf.yaml` matches your dataset geometry

## Usage

### Quick Test (Smoke Test)

Run a quick validation test with synthetic data:

```bash
python test_backproject_smoke.py
```

This creates a small test volume and verifies the geometry is correct.

### Full Reconstruction

Run GPU-accelerated reconstruction on the full dataset:

```bash
python main.py
```

Options:
- `--config CONFIG`: Path to configuration YAML (default: `conf.yaml`)
- `--projections PROJ`: Path to projections MHD file (default: `projections.mhd`)
- `--output-dir DIR`: Output directory (default: `output`)
- `--no-gpu`: Force CPU mode (useful for debugging)

Example:
```bash
python main.py --config conf.yaml --output-dir results/
```

## Project Structure

```
├── geometry.py              # Geometry dataclass and YAML loader
├── projector.py             # CPU forward projector
├── backproject.py           # CPU backprojector
├── gpu_ops.py               # GPU-accelerated implementations
├── main.py                  # Main reconstruction pipeline
├── test_backproject_smoke.py # Validation test
├── conf.yaml                # Geometry configuration
└── calibration_received.cal # Original calibration file
```

## Geometry Configuration

The `conf.yaml` file contains all geometry parameters:

- **Source geometry**: SOD (Source-to-Object Distance), SDD (Source-to-Detector Distance)
- **Detector**: pixel counts, physical size, pixel spacing
- **Rotation**: angles, COR (Center of Rotation) offset, axis direction
- **Volume**: dimensions, voxel spacing

## Output

The reconstruction produces:

- `recon_volume.npy`: NumPy array of reconstructed volume
- `recon_volume.mhd/raw`: SimpleITK format for visualization
- `recon_axial.png`, `recon_coronal.png`, `recon_sagittal.png`: Central slice visualizations

## Performance

On a modern GPU (e.g., RTX 4090), full reconstruction of 1440 projections (1024×1024) into a 271×438×438 volume takes approximately 5 mins.

## Contact

Guglielmo Cappellini

