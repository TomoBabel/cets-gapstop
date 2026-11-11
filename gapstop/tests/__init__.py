import struct
from pathlib import Path

import mrcfile
import numpy as np

from gapstop.constants import MOTL_COLUMNS, N_MOTL_COLUMNS

# Excerpt of a real GAPSTOP(TM) motive list (cryoCAT 0.2.0): the original header with the number
# of particles set to 5, and the first 5 particles of TestGapStopTM run 000388, TS_03.
TEST_DATA_DIR = Path(__file__).parent / "test_data"
REAL_MOTL_EXCERPT = TEST_DATA_DIR / "TS_03_particleList_excerpt.em"
# (x, y, z, phi, psi, theta) of those particles; positions are 1-based
REAL_MOTL_EXCERPT_ROWS = [
    (321.0, 159.0, 65.0, 285.0, -29.25, 88.26),
    (342.0, 240.0, 68.0, 225.0, 23.27, 90.0),
    (436.0, 440.0, 79.0, 225.0, 23.27, 90.0),
    (138.0, 240.0, 103.0, 240.0, -146.7, 92.32),
    (18.0, 326.0, 95.0, 225.0, 140.71, 85.95),
]
# Voxel grid (x, y, z) and sampling rate of the tomogram / scores map of TS_03
REAL_GRID = (480, 464, 140)
REAL_PIXEL_SIZE = 10.8

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
