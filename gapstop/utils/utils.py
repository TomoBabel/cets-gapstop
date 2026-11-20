import os
import shutil
import struct
from pathlib import Path
from typing import Optional, Tuple

import mrcfile
import numpy as np
import yaml

from cets_data_model.models.models import (
    CoordinateSystem,
    Axis,
    AxisType,
    AxisUnit,
    Scale,
    PointMatrixSet3D,
    ProbabilityMap3D,
)
from gapstop.constants import (
    EM_DATA_TYPE_FLOAT32,
    EM_DATA_TYPES,
    EM_HEADER_SIZE,
    EM_MACHINE_BYTE_ORDER,
    EM_MACHINE_PC,
    N_MOTL_COLUMNS,
)


# The yaml files are named after the annotation id, which already contains the tomogram id
def write_coordinates_yaml(
    coordinates: PointMatrixSet3D, output_directory: Path
) -> None:
    _write_obj_yaml(coordinates, output_directory)


def write_scores_yaml(scores: ProbabilityMap3D, output_directory: Path) -> None:
    _write_obj_yaml(scores, output_directory)


def _write_obj_yaml(
    cets_obj: PointMatrixSet3D | ProbabilityMap3D, output_directory: Path
) -> None:
    yaml_file = output_directory / f"{cets_obj.id}_gapstop_to_cets.yaml"
    metadata_dict = cets_obj.model_dump(mode="json")
    with open(yaml_file, "w") as f:
        yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
    print(f"yaml file successfully written! -> {yaml_file}")


def read_em_motl(em_file: os.PathLike) -> np.ndarray:
    """Reads a motive list stored in EM format (e.g. written by cryoCAT with
    output_type="emmotl").

    The EM file holds a 2D image of 20 x N values (x varying fastest), i.e. one row of the 20
    motive list columns per particle.

    :returns: the motive list as an (N, 20) float64 array, in the order of the file (cryoCAT
    sorts the particles by descending score).
    """
    with open(em_file, "rb") as f:
        header = f.read(EM_HEADER_SIZE)
        if len(header) < EM_HEADER_SIZE:
            raise ValueError(f"{em_file} is too short to be an EM file")
        byte_order = EM_MACHINE_BYTE_ORDER.get(header[0])
        data_type = EM_DATA_TYPES.get(header[3])
        if byte_order is None or data_type is None:
            raise ValueError(
                f"{em_file} is not a supported EM file (machine code {header[0]}, data type "
                f"code {header[3]})"
            )
        nx, ny, nz = np.frombuffer(header[4:16], dtype=f"{byte_order}i4")
        data = np.fromfile(f, dtype=f"{byte_order}{data_type}", count=nx * ny * nz)
    if data.size != nx * ny * nz:
        raise ValueError(
            f"{em_file} contains {data.size} values, but its header declares "
            f"{nx} x {ny} x {nz}"
        )
    if nx != N_MOTL_COLUMNS or nz != 1:
        raise ValueError(
            f"{em_file} is not a motive list: expected {N_MOTL_COLUMNS} x N x 1 values, "
            f"found {nx} x {ny} x {nz}"
        )
    return data.reshape(ny, nx).astype(np.float64)


def write_em_motl(motl: np.ndarray, em_file: os.PathLike) -> Path:
    """Writes a motive list, an (N, 20) array, in EM format (little-endian float32, as
    cryoCAT does with output_type="emmotl"). Being float32, the values keep about 7
    significant digits."""
    motl = np.asarray(motl)
    if motl.ndim != 2 or motl.shape[1] != N_MOTL_COLUMNS:
        raise ValueError(
            f"A motive list must have N x {N_MOTL_COLUMNS} values, found {motl.shape}"
        )
    header = bytearray(EM_HEADER_SIZE)
    header[0], header[3] = EM_MACHINE_PC, EM_DATA_TYPE_FLOAT32
    header[4:16] = struct.pack("<3i", N_MOTL_COLUMNS, motl.shape[0], 1)
    em_path = Path(em_file)
    em_path.write_bytes(bytes(header) + motl.astype("<f4").tobytes())
    return em_path


def euler_zxz_to_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Rotation matrix of the motive list Euler ZXZ angles (degrees):
    R = Rz(psi) @ Rx(theta) @ Rz(phi).

    It rotates the template into the orientation of the particle in the tomogram. It is the
    matrix scipion-em-gapstop builds with euler_matrix(phi, theta, psi, "szxz") before
    inverting it: since scipion-em-gapstop 3.3.1, Scipion stores the inverse (particle ->
    reference). It is also the one cets-dynamo builds for the equivalent Dynamo angles
    (cryoCAT converts Dynamo tables with phi = -narot, psi = -tdrot, theta = -tilt), and has
    the same direction as the poses of cets-pytom_tm and cets-scipion, so the orientations of
    all the converters are consistent.
    """
    return _rot_z(psi) @ _rot_x(theta) @ _rot_z(phi)


def matrix_to_euler_zxz(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of ``euler_zxz_to_matrix``: the motive list Euler ZXZ angles (phi, theta, psi),
    in degrees, of a rotation matrix, with phi in [0, 360), theta in [0, 180] and psi in
    (-180, 180].

    When theta is 0 or 180 only psi + phi (or psi - phi) is defined, so psi is set to 0: the
    angles may differ from the ones the matrix was built from, but not the rotation.
    """
    theta = np.arccos(np.clip(matrix[2, 2], -1.0, 1.0))
    if abs(np.sin(theta)) < 1e-8:
        psi = 0.0
        phi = np.arctan2(matrix[1, 0], matrix[0, 0])
        if matrix[2, 2] < 0:
            phi = -phi
    else:
        psi = np.arctan2(matrix[0, 2], -matrix[1, 2])
        phi = np.arctan2(matrix[2, 0], matrix[2, 1])
    return (
        float(np.rad2deg(phi)) % 360.0,
        float(np.rad2deg(theta)),
        float(np.rad2deg(psi)),
    )


def _rot_z(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _rot_x(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def gen_coordinate_systems(
    name: str, ndim: int = 2
) -> Tuple[CoordinateSystem, CoordinateSystem]:
    """Builds the (array, physical) coordinate-system pair for an ``ndim`` image/frame.

    Names follow the CETS proposal convention ``{name}_array`` / ``{name}_physical``. The array
    system is pixel/array coords (unit ``pixel``); the physical system is in Å. ``ndim`` is 2 for
    2-D images (x, y) and 3 for volumes (x, y, z).
    """
    axes = ("x", "y", "z")[:ndim]
    array_cs = CoordinateSystem(
        name=f"{name}_array",
        axes=[
            Axis(name=a, axis_type=AxisType.array, axis_unit=AxisUnit.pixel)
            for a in axes
        ],
    )
    physical_cs = CoordinateSystem(
        name=f"{name}_physical",
        axes=[
            Axis(name=a, axis_type=AxisType.space, axis_unit=AxisUnit.angstrom)
            for a in axes
        ],
    )
    return array_cs, physical_cs


def gen_array_to_physical(
    pixel_size: float, array_cs_name: str, physical_cs_name: str, ndim: int = 2
) -> Scale:
    """The single canonical ``array_to_physical`` transformation for an image: a Scale mapping
    pixel/array coordinates to physical (Å) coordinates by the (isotropic) pixel/voxel size.
    The spec requires exactly one such transformation per image."""
    return Scale(
        scale=[pixel_size] * ndim,
        name="array_to_physical",
        input=array_cs_name,
        output=physical_cs_name,
    )


def validate_file(filename: os.PathLike, expected_ext: str) -> Path:
    if filename is None:
        raise ValueError("The introduced file cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{filename} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {filename}: {p}")

    ext = p.suffix
    if ext != expected_ext:
        raise ValueError(f"Invalid file extension '{ext}'. Expected: {expected_ext}")
    return p


def get_voxel_size(
    cets_obj: PointMatrixSet3D | ProbabilityMap3D,
) -> Optional[float]:
    """Voxel size (Å) of a CETS annotation: the scale of its ``array_to_physical``
    transformation. None if it has none."""
    for transformation in cets_obj.coordinate_transformations or []:
        if isinstance(transformation, Scale) and transformation.scale:
            return float(transformation.scale[0])
    return None


def export_mrc(
    mrc_file: os.PathLike,
    out_file: os.PathLike,
    voxel_size: Optional[float] = None,
    copy: bool = False,
) -> Path:
    """Makes an mrc file available with another name: as a symbolic link, unless a copy is
    requested or needed.

    :param voxel_size: voxel size (Å) the file must have in its header. If it is not the one
    of ``mrc_file``, the file is copied and its header updated, instead of linked.
    """
    src = validate_file(mrc_file, expected_ext=Path(mrc_file).suffix)
    dst = Path(out_file).expanduser()
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src:
            return dst
        raise FileExistsError(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.open(src, header_only=True, permissive=True) as mrc:
        header_voxel_size = float(mrc.voxel_size.x)
    fix_header = voxel_size is not None and not np.isclose(
        header_voxel_size, voxel_size, rtol=1e-4
    )
    if not (copy or fix_header):
        dst.symlink_to(src)
        return dst
    shutil.copyfile(src, dst)
    if fix_header:
        # Memory-mapped, so only the header is rewritten: a file opened with
        # header_only=True in "r+" mode is truncated to its header when closed.
        with mrcfile.mmap(dst, mode="r+", permissive=True) as mrc:
            mrc.voxel_size = voxel_size
    return dst
