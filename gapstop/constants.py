EM_EXT = ".em"
MRC_EXT = ".mrc"

# Name of the particle list written by scipion-em-gapstop (cryocat.tmana.scores_extract_particles
# with output_type="emmotl"). Natively, the name is the one given as output_path.
PARTICLE_LIST_NAME = "particleList.em"
# Script written next to it by scipion-em-gapstop to run the extraction, whose call to
# EXTRACTION_FUNCTION records the scores map (ScoreTomogram) the particles were extracted from.
EXTRACTION_SCRIPT_NAME = "extractCoords.py"
EXTRACTION_FUNCTION = "scores_extract_particles"

# Columns of a motive list (TOM/AV3 convention, used by cryoCAT, STOPGAP and GAPSTOP(TM)): one
# row of 20 values per particle. x, y, z are 1-based voxel positions in the tomogram; the
# angles (degrees) are Euler ZXZ: phi (1st rotation around Z), theta (around X) and psi (2nd
# rotation around Z), stored in the order phi, psi, theta.
MOTL_COLUMNS = (
    "score",
    "geom1",
    "geom2",
    "subtomo_id",
    "tomo_id",
    "object_id",
    "subtomo_mean",
    "x",
    "y",
    "z",
    "shift_x",
    "shift_y",
    "shift_z",
    "geom3",
    "geom4",
    "geom5",
    "phi",
    "psi",
    "theta",
    "class",
)
N_MOTL_COLUMNS = len(MOTL_COLUMNS)
SCORE = MOTL_COLUMNS.index("score")
TOMO_ID = MOTL_COLUMNS.index("tomo_id")
X, Y, Z = (MOTL_COLUMNS.index(c) for c in ("x", "y", "z"))
SHIFT_X, SHIFT_Y, SHIFT_Z = (
    MOTL_COLUMNS.index(c) for c in ("shift_x", "shift_y", "shift_z")
)
PHI, PSI, THETA = (MOTL_COLUMNS.index(c) for c in ("phi", "psi", "theta"))

# Motive list positions are 1-based (cryoCAT writes the score map index + 1), while the CETS
# array frame is 0-based.
MOTL_POSITION_OFFSET = 1

# EM file format: a 512-byte header followed by the data, stored with x varying fastest.
EM_HEADER_SIZE = 512
EM_MACHINE_BYTE_ORDER = {
    0: ">",  # OS-9
    3: ">",  # SGI
    5: ">",  # Mac
    6: "<",  # PC
}
EM_DATA_TYPES = {1: "i1", 2: "i2", 4: "i4", 5: "f4", 9: "f8"}
