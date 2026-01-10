"""
Define a Python dataclass `Geometry` for SARRP-style cone-beam CT.

Loads all geometry parameters from the conf.yaml file and provides derived quantities.
All distances are in millimetres, all angles in radians.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


@dataclass
class Geometry:
    """Cone-beam CT geometry parameters for SARRP system."""
    
    # Source and detector distances
    SOD: float  # Source-to-Isocenter Distance [mm]
    SDD: float  # Source-to-Detector Distance [mm]
    IDD: float  # Isocenter-to-Detector Distance [mm]
    
    # Source position relative to isocenter
    source_pos: Tuple[float, float, float]  # [mm]
    
    # Detector reference point
    detector_ref: Tuple[float, float, float]  # [mm]
    
    # Center of rotation offset
    COR: Tuple[float, float, float]  # [mm] in rodent frame
    
    # CTilt orientation (quaternion)
    CTilt: Tuple[float, float, float, float]
    
    # Detector parameters
    det_nu: int  # number of pixels (horizontal)
    det_nv: int  # number of pixels (vertical)
    det_size: Tuple[float, float]  # physical size [mm]
    det_pixel_size: Tuple[float, float] = field(init=False)  # derived
    
    # Rotation axis direction
    rotation_axis: Tuple[float, float, float]
    
    # Projection angles
    angle_start_deg: float
    angle_end_deg: float
    n_projections: int
    angles_rad: np.ndarray = field(init=False)  # angles in radians
    
    def __post_init__(self) -> None:
        """Compute derived quantities after initialization."""
        # Detector pixel size
        self.det_pixel_size = (
            self.det_size[0] / self.det_nu,
            self.det_size[1] / self.det_nv,
        )
        
        # Generate angle array in radians
        angles_deg = np.linspace(
            self.angle_start_deg,
            self.angle_end_deg,
            self.n_projections,
            endpoint=False,
        )
        self.angles_rad = np.deg2rad(angles_deg)


def load_geometry_from_yaml(path: str | Path) -> Geometry:
    """
    Load geometry parameters from a YAML configuration file.
    
    Parameters
    ----------
    path : str or Path
        Path to the conf.yaml file.
    
    Returns
    -------
    Geometry
        Geometry instance with all parameters loaded and derived quantities computed.
    """
    path = Path(path)
    
    with open(path, "r") as f:
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


geom = load_geometry_from_yaml(Path(__file__).parent / "conf.yaml")
print(geom.det_pixel_size)
print(geom.angles_rad[:5])