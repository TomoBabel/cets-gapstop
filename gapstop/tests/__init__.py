import struct
from pathlib import Path

import mrcfile
import numpy as np

from gapstop.constants import MOTL_COLUMNS, N_MOTL_COLUMNS

# Test data: the files of a real GAPSTOP(TM) template matching of TS_03 (see
# tests_cli_fixtures.py).
DATA_DIR = Path(__file__).parent / "data"
# Its motive list (cryoCAT): 1042 particles of tomogram 1
REAL_MOTL = DATA_DIR / "particleList.em"
REAL_MOTL_N_PARTICLES = 1042
# (x, y, z, phi, psi, theta) of its first 5 particles; positions are 1-based
REAL_MOTL_FIRST_ROWS = [
    (210.0, 144.0, 50.0, 285.0, 55.73, 87.11),
    (218.0, 220.0, 40.0, 285.0, 23.27, 90.0),
    (158.0, 81.0, 37.0, 45.0, -21.59, 100.48),
    (114.0, 5.0, 33.0, 165.0, 135.98, 105.83),
    (50.0, 64.0, 10.0, 75.0, 75.79, 91.74),
]
# Voxel grid (x, y, z) and sampling rate of the tomogram / scores map of TS_03
REAL_GRID = (240, 232, 70)
REAL_PIXEL_SIZE = 21.6

TOMO_ID = "TS_01"
PIXEL_SIZE = 5.4


def motl_row(**values: float) -> list[float]:
    """A motive list row with the given column values (0 elsewhere)."""
    row = [0.0] * N_MOTL_COLUMNS
    for column, value in values.items():
        row[MOTL_COLUMNS.index(column)] = value
    return row


def write_em_motl(em_file: Path, rows: list[list[float]]) -> Path:
    """Writes a motive list in EM format (little-endian float32), as cryoCAT does."""
    header = bytearray(512)
    header[0], header[3] = 6, 5  # PC (little-endian), float32
    header[4:16] = struct.pack("<3i", N_MOTL_COLUMNS, len(rows), 1)
    data = np.asarray(rows, dtype="<f4").reshape(len(rows), N_MOTL_COLUMNS)
    em_file.write_bytes(bytes(header) + data.tobytes())
    return em_file


def write_mrc(
    mrc_file: Path, dims_xyz: tuple[int, int, int], voxel_size: float
) -> Path:
    x, y, z = dims_xyz
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(np.zeros((z, y, x), dtype=np.int8))
        mrc.voxel_size = voxel_size
    return mrc_file
