# cets-gapstop

Converters between [GAPSTOP(TM)](https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm) template
matching outputs and CETS, the cryo-electron tomography metadata standard defined in
[cets-data-models](https://github.com/TomoBabel/cets-data-models).

| Data | GAPSTOP(TM) files | CETS objects | Directions |
|---|---|---|---|
| Picked particles | motive list (`.em`, e.g. `particleList.em`, written by `cryocat.tmana.scores_extract_particles`) | `PointMatrixSet3D` (positions, orientations, scores) | GAPSTOP → CETS, CETS → GAPSTOP |
| Scores map | `scores_<index>_<tomo num>.mrc` | `ProbabilityMap3D` (`value_type: score`) | GAPSTOP → CETS, CETS → GAPSTOP |

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `gapstop` package and the `cets-gapstop` command. GAPSTOP(TM) and cryoCAT are
not needed: the motive lists are read and written directly.

## Command line

```text
cets-gapstop {coordinates,scores} {to-cets,from-cets} [options]
```

```bash
# Particles: GAPSTOP -> CETS -> GAPSTOP
cets-gapstop coordinates to-cets --motl particleList.em --tomo-id TS_03 \
    --annotated-object ribosome --score-annotation-id gapstop_scores_TS_03 -o coords.yaml
cets-gapstop coordinates from-cets -i coords.yaml -o out/particleList.em

# Scores map: GAPSTOP -> CETS -> GAPSTOP
cets-gapstop scores to-cets --scores-map scores_0_1.mrc --tomo-id TS_03 --threshold 0.1 -o scores.yaml
cets-gapstop scores from-cets -i scores.yaml --output-dir out
```

`to-cets` writes a CETS yaml file with the entry `coordinates` or `scores`; `from-cets` reads
it. Every sub-command has `--help`. Errors print a one-line message and exit with code 1; put
`--debug` before the data type to also print the traceback.

| `coordinates to-cets` option | Meaning |
|---|---|
| `--motl` (required) | Motive list (`.em`): the particle table of the TOM/AV3 toolboxes that cryoCAT uses, with one row per particle (20 columns) holding its score, tomogram number, position (1-based voxels plus sub-voxel shift) and orientation (ZXZ Euler angles phi, psi, theta). GAPSTOP(TM) writes it when extracting the particles from the scores map (`cryocat.tmana.scores_extract_particles`). |
| `--tomo-id` (required) | Id of the CETS tomogram (`source_tomogram_id`). It also names the coordinate systems (`<tomo id>_array`, `<tomo id>_physical`), so it must be the id of the CETS tomogram. |
| `--tomo-num` | Numeric tomogram id (`tomo_id` column) of the particles to convert; needed if the motive list has the particles of several tomograms. |
| `--pixel-size` | Sampling rate of the tomogram (Å). Default: the one of the reference map. Without any, only the array (voxel) coordinate system is attached. |
| `--reference-map` | mrc file on the voxel grid the particles were picked in: gives the sampling rate and checks that the particles are inside the grid. Default: the scores map recorded in the extraction script next to the motive list (`extractCoords.py`, written by scipion-em-gapstop), unless `--no-extraction-script`. |
| `--annotated-object` | What the template represents (e.g. `ribosome`); template matching does not record it. |
| `--score-annotation-id` | Id of the scores map annotation the particles were extracted from (`derived_from_annotation_id`). |
| `--annotation-id` | Id of the annotation. Default: `gapstop_coords_<tomo id>`. |

| Other options | Meaning |
|---|---|
| `coordinates from-cets --tomo-num/--object-id/--class-id` | Values of those motive list columns, which CETS does not store (default: 1). |
| `scores to-cets --threshold` | Score above which voxels were particle candidates in the extraction. |
| `scores to-cets --pixel-size` | Sampling rate (Å). Default: the one of the map header; an error is raised if neither is known. |
| `scores from-cets --tomo-num/--index` | Numbers of the map name `scores_<index>_<tomo num>.mrc` (defaults: 1 and 0). |
| `scores from-cets --copy` | Copy the map instead of linking it (it is always copied if its header sampling rate is not the CETS one). |

## Python API

```python
from gapstop.converters.coordinates3d import GapstopCoordinates3D
from gapstop.converters.score_tomogram import GapstopScoreTomogram

scores = GapstopScoreTomogram().gapstop_to_cets("scores_0_1.mrc", "TS_03")
coords = GapstopCoordinates3D().gapstop_to_cets(
    "particleList.em", "TS_03", reference_map="scores_0_1.mrc", score_annotation_id=scores.id
)
GapstopCoordinates3D().cets_to_gapstop(coords, "out/particleList.em")
```

`gapstop.utils.extraction.find_extraction_params` reads the scores map and threshold the
particles were extracted with from the extraction script next to the motive list.

## Conventions

- **Positions.** Motive list positions are 1-based voxel indices (cryoCAT writes the index of
  the scores map + 1), split into an integer position and a shift. In CETS they are 0-based
  positions in the tomogram (`origin3D`): position + shift − 1. When writing, each position is
  split again into the nearest voxel and the rest.
- **Orientations.** The angles are Euler ZXZ (phi, theta, psi), stored in the order phi, psi,
  theta. `matrix3D` is the rotation that brings the template into the particle,
  `R = Rz(psi) @ Rx(theta) @ Rz(phi)`. It is the inverse of the matrix Scipion stores, and the
  same rotation cets-scipion gives for the particle. When theta is 0 or 180 the angles written
  back may differ from the original ones, but they describe the same rotation.
- **Scores** are the template matching score of each particle (`score` column). Particles keep
  the order of the motive list.
- Motive lists are float32: values keep about 7 significant digits.

## Limitations

- The `subtomo_id`, `object_id`, `class` and `geom` columns are not kept.
- The angles map (`angles_<index>_<tomo num>.mrc`), the list of sampled angles and the template
  matching parameters (template, mask, angular sampling, wedge) are not converted, so a scores
  map written back is not enough to extract particles again.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest gapstop/tests/tests_*.py   # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy gapstop
```

### Tests

| File | What it tests |
|---|---|
| `gapstop/tests/tests_cets_gapstop.py` | GAPSTOP(TM) → CETS: the motive list reader (on `particleList.em`), the Euler angles and the coordinates. |
| `gapstop/tests/tests_cets_to_gapstop.py` | CETS → GAPSTOP(TM): motive lists and scores maps. |
| `gapstop/tests/tests_score_tomogram.py` | The scores map converter and the extraction script parser (on a synthetic script). |
| `gapstop/tests/tests_cli.py` | Every `cets-gapstop` sub-command, on small synthetic files (built in a temporary directory). |
| `gapstop/tests/tests_cli_fixtures.py` | Every `cets-gapstop` sub-command on the data of a real GAPSTOP(TM) template matching (`gapstop/tests/data`): the outputs are compared with the values of the input files and with reference CETS files, plus the round trips, the options and the errors. |

```bash
pytest gapstop/tests/tests_cli_fixtures.py                    # one file
pytest gapstop/tests/tests_cli_fixtures.py -k scores          # some tests
```

The test data (`gapstop/tests/data`, `DATA_DIR` in `gapstop/tests/__init__.py`) is the
template matching of the tomogram of TS_03:

| File | Content |
|---|---|
| `TS_03.mrc` | The tomogram (240 x 232 x 70 voxels, 21.6 Å/voxel). |
| `scores_0_1.mrc` | Scores map of the template matching, on the grid of the tomogram. |
| `particleList.em` | Motive list of the particles extracted from the scores map: 1042 particles of tomogram 1, with their orientation and score. |
| `TS_03_parameters.txt` | Acquisition of TS_03 and the voxel size of the tomogram (`--pixel-size 21.6`), the only one of its options `cets-gapstop` takes. |
| `gapstop_coords_TS_03_gapstop_to_cets.yaml` | Reference CETS coordinates of `particleList.em`. |
| `gapstop_scores_TS_03_gapstop_to_cets.yaml` | Reference CETS scores map of `scores_0_1.mrc` (its path is relative to the data directory). |

The reference CETS files hold what `to-cets` gives with `--tomo-id TS_03 --pixel-size 21.6`,
plus `--score-annotation-id gapstop_scores_TS_03` for the coordinates (they are linked to the
scores map) and, for the scores map, the threshold of the particle extraction
(`--threshold`, which only the reference holds: GAPSTOP(TM) records it in the extraction
script). The motive list written back from the coordinates has the values of
`particleList.em`. If the conversion changes on purpose, regenerate them with the Python API
(`write_coordinates_yaml` / `write_scores_yaml` of `gapstop/utils/utils.py`), make the path of
the scores map relative and review their diff. The files in `gapstop/tests/data` are never
modified by the tests.
