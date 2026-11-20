# -*- coding: utf-8 -*-
"""Tests of the ``cets-gapstop`` command line with the data of a real GAPSTOP(TM) template
matching of TS_03 (``gapstop/tests/data``), in both directions.

Fixtures in ``gapstop/tests/data``:

- ``TS_03.mrc``: the tomogram (240 x 232 x 70 voxels, 21.6 Å/voxel).
- ``scores_0_1.mrc``: the scores map of the template matching, on the grid of the tomogram.
- ``particleList.em``: the motive list of the particles extracted from the scores map
  (1042 particles of tomogram 1, with their orientation and score).
- ``TS_03_parameters.txt``: the acquisition of TS_03 and the voxel size of the tomogram
  (``--pixel-size``), the only one of its options ``cets-gapstop`` takes.
- ``gapstop_coords_TS_03_gapstop_to_cets.yaml``: reference CETS coordinates of
  ``particleList.em`` (linked to the scores map annotation).
- ``gapstop_scores_TS_03_gapstop_to_cets.yaml``: reference CETS scores map of
  ``scores_0_1.mrc``, with the threshold of the particle extraction (its path is relative to
  the data directory).

The expected values are read from the fixtures (``read_em_motl``, mrcfile), and the outputs
are also compared with the reference CETS files.
"""

import contextlib
import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    PointMatrixSet3D,
    ProbabilityMap3D,
    Scale,
)
from cets_data_model.utils.yaml_io import load_cets_object
from gapstop.cli import main
from gapstop.constants import (
    MOTL_COLUMNS,
    MOTL_POSITION_OFFSET,
    PHI,
    PSI,
    SCORE,
    SUBTOMO_ID,
    THETA,
    TOMO_ID,
    X,
    Z,
)
from gapstop.tests import DATA_DIR, write_mrc
from gapstop.utils.utils import euler_zxz_to_matrix, read_em_motl

TOMOGRAM = DATA_DIR / "TS_03.mrc"
SCORES_MAP = DATA_DIR / "scores_0_1.mrc"
MOTL = DATA_DIR / "particleList.em"
PARAMETERS = DATA_DIR / "TS_03_parameters.txt"
REF_COORDINATES = DATA_DIR / "gapstop_coords_TS_03_gapstop_to_cets.yaml"
REF_SCORES = DATA_DIR / "gapstop_scores_TS_03_gapstop_to_cets.yaml"

TS_ID = "TS_03"
TOMO_NUM = 1  # tomo_id column of the motive list
SCORES_ID = f"gapstop_scores_{TS_ID}"  # Default id of the scores map annotation
ATOL = 1e-6


def read_parameters() -> Dict[str, str]:
    """The options of TS_03_parameters.txt ({"--pixel-size": "21.6", ...}); the lines that
    are not options (the title) are skipped."""
    parameters = {}
    for line in PARAMETERS.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("--"):
            parameters[parts[0]] = parts[1]
    return parameters


def grid(path: Path) -> Tuple[Tuple[int, int, int], float]:
    """Dimensions (x, y, z) and voxel size of an mrc file."""
    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        header = mrc.header
        return (int(header.nx), int(header.ny), int(header.nz)), float(mrc.voxel_size.x)


def reference_threshold() -> float:
    """Threshold of the particle extraction, which only the reference CETS scores map
    holds (GAPSTOP(TM) records it in the extraction script)."""
    threshold = load_cets_object(REF_SCORES, "scores", ProbabilityMap3D).threshold
    assert threshold is not None
    return threshold


def array_to_physical_scale(obj: PointMatrixSet3D | ProbabilityMap3D) -> List[float]:
    scales = [
        t
        for t in obj.coordinate_transformations or []
        if isinstance(t, Scale) and t.name == "array_to_physical"
    ]
    if len(scales) != 1:
        raise AssertionError(f"Expected one array_to_physical scale, found {scales}")
    return list(scales[0].scale)


def tree_digest(directory: Path) -> Dict[str, str]:
    """sha256 of every file of a directory, to check that it is not modified."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-gapstop with the given arguments; returns the exit code and what it
    printed to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CetsGapstopCliFixturesTest(TestCase):
    pixel_size: float

    @classmethod
    def setUpClass(cls) -> None:
        cls.pixel_size = float(read_parameters()["--pixel-size"])

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_gapstop_cli_fixtures"))
        self.yaml = self.dir / "TS_03_cets.yaml"
        self.out = self.dir / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-gapstop, checks its exit code and returns what it printed to
        stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return err

    def _coords_args(self, *extra: str) -> List[str]:
        return [
            "coordinates",
            "to-cets",
            "--motl",
            str(MOTL),
            "--tomo-id",
            TS_ID,
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _coords_to_cets(self, *extra: str) -> PointMatrixSet3D:
        self.assert_run(self._coords_args(*extra))
        return load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)

    def _scores_args(self, *extra: str) -> List[str]:
        return [
            "scores",
            "to-cets",
            "--scores-map",
            str(SCORES_MAP),
            "--tomo-id",
            TS_ID,
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _scores_to_cets(self, *extra: str) -> ProbabilityMap3D:
        self.assert_run(self._scores_args(*extra))
        return load_cets_object(self.yaml, "scores", ProbabilityMap3D)

    # ------------------------------------------------------------------ fixtures
    def test_fixture_files(self) -> None:
        """The tomogram, the scores map and the motive list share the voxel grid and the
        voxel size of the parameters file."""
        tomo_dims, tomo_voxel = grid(TOMOGRAM)
        scores_dims, scores_voxel = grid(SCORES_MAP)
        self.assertEqual(scores_dims, tomo_dims)
        self.assertAlmostEqual(tomo_voxel, self.pixel_size, places=4)
        self.assertAlmostEqual(scores_voxel, self.pixel_size, places=4)
        motl = read_em_motl(MOTL)
        self.assertEqual(motl.shape[1], len(MOTL_COLUMNS))
        np.testing.assert_array_equal(motl[:, TOMO_ID], TOMO_NUM)
        # 1-based positions inside the grid
        positions = motl[:, X : Z + 1]
        self.assertTrue(np.all(positions >= 1))
        self.assertTrue(np.all(positions <= tomo_dims))
        # The particles were extracted above the threshold of the scores map
        self.assertTrue(np.all(motl[:, SCORE] >= reference_threshold() - ATOL))

    # ------------------------------------------------------------------ coordinates
    def test_coordinates_match_reference(self) -> None:
        coordinates = self._coords_to_cets(
            "--pixel-size",
            str(self.pixel_size),
            "--score-annotation-id",
            SCORES_ID,
        )
        reference = load_cets_object(REF_COORDINATES, "coordinates", PointMatrixSet3D)
        self.assertEqual(coordinates, reference)

    def test_coordinates_values(self) -> None:
        """The coordinates against the values of the motive list."""
        coordinates = self._coords_to_cets("--pixel-size", str(self.pixel_size))
        motl = read_em_motl(MOTL)
        self.assertEqual(coordinates.source_tomogram_id, TS_ID)
        self.assertEqual(coordinates.id, f"gapstop_coords_{TS_ID}")
        # 1-based motive list positions (+ shifts) -> 0-based array frame
        shifts = motl[:, [MOTL_COLUMNS.index(f"shift_{a}") for a in "xyz"]]
        np.testing.assert_allclose(
            coordinates.origin3D or [],
            motl[:, X : Z + 1] + shifts - MOTL_POSITION_OFFSET,
            atol=ATOL,
        )
        np.testing.assert_allclose(coordinates.scores or [], motl[:, SCORE], atol=ATOL)
        for matrix, row in zip(coordinates.matrix3D or [], motl):
            rotation = euler_zxz_to_matrix(row[PHI], row[THETA], row[PSI])
            np.testing.assert_allclose(matrix, rotation, atol=1e-6)
        np.testing.assert_allclose(
            array_to_physical_scale(coordinates), [self.pixel_size] * 3
        )

    def test_coordinates_reference_map(self) -> None:
        """--reference-map gives the voxel size (from its header) and checks the particles
        fit in its grid: the tomogram gives the same coordinates."""
        with_pixel_size = self._coords_to_cets("--pixel-size", str(self.pixel_size))
        self.yaml.unlink()
        with_map = self._coords_to_cets("--reference-map", str(TOMOGRAM))
        self.assertEqual(with_map, with_pixel_size)
        # A map whose grid is too small for the particles
        small = write_mrc(self.dir / "small.mrc", (16, 16, 8), self.pixel_size)
        self.yaml.unlink()
        err = self.assert_run(self._coords_args("--reference-map", str(small)), code=1)
        self.assertIn("outside", err)
        self.assertFalse(self.yaml.exists())

    def test_coordinates_options(self) -> None:
        coordinates = self._coords_to_cets(
            "--pixel-size",
            str(self.pixel_size),
            "--tomo-num",
            str(TOMO_NUM),
            "--annotated-object",
            "ribosome",
            "--annotation-id",
            "my_coords",
            "--no-extraction-script",
        )
        self.assertEqual(coordinates.id, "my_coords")
        assert coordinates.annotated_object is not None
        self.assertEqual(coordinates.annotated_object.name, "ribosome")
        self.assertEqual(len(coordinates.origin3D or []), len(read_em_motl(MOTL)))
        self.assertIsNone(coordinates.derived_from_annotation_id)

    def test_coordinates_round_trip(self) -> None:
        """to-cets + from-cets give back the motive list (all its columns)."""
        self._coords_to_cets("--pixel-size", str(self.pixel_size))
        written = self.out / "particleList.em"
        self.assert_run(
            ["coordinates", "from-cets", "-i", str(self.yaml), "-o", str(written)]
        )
        np.testing.assert_allclose(read_em_motl(written), read_em_motl(MOTL), atol=ATOL)

    def test_coordinates_from_reference(self) -> None:
        """The reference CETS coordinates give the motive list; --tomo-num, --object-id
        and --class-id fill their columns."""
        written = self.out / "particleList.em"
        args = ["coordinates", "from-cets", "-i", str(REF_COORDINATES)]
        self.assert_run(args + ["-o", str(written)])
        np.testing.assert_allclose(read_em_motl(written), read_em_motl(MOTL), atol=ATOL)
        self.assert_run(
            args
            + ["-o", str(written), "--tomo-num", "3"]
            + ["--object-id", "4", "--class-id", "5"]
        )
        motl = read_em_motl(written)
        np.testing.assert_array_equal(motl[:, TOMO_ID], 3)
        np.testing.assert_array_equal(motl[:, MOTL_COLUMNS.index("object_id")], 4)
        np.testing.assert_array_equal(motl[:, MOTL_COLUMNS.index("class")], 5)
        np.testing.assert_array_equal(motl[:, SUBTOMO_ID], np.arange(1, len(motl) + 1))

    def test_coordinates_errors(self) -> None:
        # No particles of that tomogram
        err = self.assert_run(self._coords_args("--tomo-num", "2"), code=1)
        self.assertIn("no particles", err)
        # Missing motive list
        args = self._coords_args()
        args[args.index("--motl") + 1] = str(self.dir / "missing.em")
        self.assert_run(args, code=1)
        self.assertFalse(self.yaml.exists())
        # Wrong output extension
        self.assert_run(
            [
                "coordinates",
                "from-cets",
                "-i",
                str(REF_COORDINATES),
                "-o",
                str(self.dir / "particleList.txt"),
            ],
            code=1,
        )
        # A scores map is not a set of coordinates
        self.assert_run(
            [
                "coordinates",
                "from-cets",
                "-i",
                str(REF_SCORES),
                "-o",
                str(self.out / "particleList.em"),
            ],
            code=1,
        )
        self.assertFalse((self.out / "particleList.em").exists())

    # ------------------------------------------------------------------ scores
    def test_scores_match_reference(self) -> None:
        scores = self._scores_to_cets(
            "--pixel-size",
            str(self.pixel_size),
            "--threshold",
            str(reference_threshold()),
        )
        reference = load_cets_object(REF_SCORES, "scores", ProbabilityMap3D)
        # The reference path is relative to the data directory
        self.assertEqual((DATA_DIR / (reference.path or "")).resolve(), SCORES_MAP)
        self.assertEqual(Path(scores.path or "").resolve(), SCORES_MAP)
        self.assertEqual(
            scores.model_copy(update={"path": None}),
            reference.model_copy(update={"path": None}),
        )

    def test_scores_values(self) -> None:
        """Without --pixel-size, the voxel size of the header."""
        scores = self._scores_to_cets()
        dims, voxel_size = grid(SCORES_MAP)
        self.assertEqual((scores.width, scores.height, scores.depth), dims)
        np.testing.assert_allclose(array_to_physical_scale(scores), [voxel_size] * 3)
        self.assertEqual(scores.id, SCORES_ID)
        self.assertEqual(scores.source_tomogram_id, TS_ID)
        self.assertIsNone(scores.threshold)

    def test_scores_from_cets(self) -> None:
        """The map is linked as scores_<index>_<tomo num>.mrc, or copied with --copy.

        The yaml written by to-cets is used, not the reference one: the relative path of the
        reference is resolved from the working directory, not from the yaml file."""
        self._scores_to_cets()
        args = ["scores", "from-cets", "-i", str(self.yaml)]
        self.assert_run(args + ["--output-dir", str(self.out)])
        linked = self.out / "scores_0_1.mrc"
        self.assertTrue(linked.is_symlink())
        self.assertEqual(linked.resolve(), SCORES_MAP)
        copied_dir = self.out / "copied"
        self.assert_run(
            args
            + ["--output-dir", str(copied_dir), "--copy"]
            + ["--index", "2", "--tomo-num", "3"]
        )
        copied = copied_dir / "scores_2_3.mrc"
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), SCORES_MAP.read_bytes())

    def test_scores_from_cets_other_voxel_size(self) -> None:
        """If the CETS voxel size is not the one of the header, the map is copied with it
        in its header and all its data."""
        voxel_size = self.pixel_size / 2
        self._scores_to_cets("--pixel-size", str(voxel_size))
        self.assert_run(
            ["scores", "from-cets", "-i", str(self.yaml), "--output-dir", str(self.out)]
        )
        written = self.out / "scores_0_1.mrc"
        self.assertFalse(written.is_symlink())
        self.assertAlmostEqual(grid(written)[1], voxel_size, places=4)
        with mrcfile.mmap(written, permissive=True) as mrc:
            with mrcfile.mmap(SCORES_MAP, permissive=True) as original:
                np.testing.assert_array_equal(mrc.data, original.data)

    def test_scores_errors(self) -> None:
        # A set of coordinates is not a scores map
        self.assert_run(
            [
                "scores",
                "from-cets",
                "-i",
                str(REF_COORDINATES),
                "--output-dir",
                str(self.out),
            ],
            code=1,
        )
        # Missing map / non-positive pixel size
        args = self._scores_args()
        args[args.index("--scores-map") + 1] = str(self.dir / "scores_0_1.mrc")
        self.assert_run(args, code=1)
        self.assert_run(self._scores_args("--pixel-size", "0"), code=1)
        self.assertFalse(self.yaml.exists())

    # ------------------------------------------------------------------ command line
    def test_input_files_are_not_modified(self) -> None:
        before = tree_digest(DATA_DIR)
        self._coords_to_cets("--reference-map", str(TOMOGRAM))
        self.assert_run(
            [
                "coordinates",
                "from-cets",
                "-i",
                str(self.yaml),
                "-o",
                str(self.dir / "p.em"),
            ]
        )
        self._scores_to_cets("--pixel-size", "5")
        self.assert_run(
            ["scores", "from-cets", "-i", str(self.yaml), "--output-dir", str(self.out)]
        )
        self.assertEqual(tree_digest(DATA_DIR), before)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["coordinates"],
            ["tomogram", "to-cets"],
            # --tomo-id is required
            ["coordinates", "to-cets", "--motl", str(MOTL), "-o", str(self.yaml)],
            self._coords_args("--tomo-num", "one"),  # an integer
            ["scores", "to-cets", "--tomo-id", TS_ID, "-o", str(self.yaml)],
            ["scores", "from-cets", "-i", str(REF_SCORES)],  # --output-dir
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as ctx:
                    run(args)
                self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(self.yaml.exists())

    def test_help(self) -> None:
        for args in (
            ["--help"],
            ["coordinates", "to-cets", "--help"],
            ["scores", "from-cets", "--help"],
        ):
            with self.subTest(args=args):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit) as ctx:
                        main(args)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("cets-gapstop", out.getvalue())
