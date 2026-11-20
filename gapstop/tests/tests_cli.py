# -*- coding: utf-8 -*-
import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase

import numpy as np

from cets_data_model.models.models import PointMatrixSet3D, ProbabilityMap3D
from cets_data_model.utils.yaml_io import load_cets_object
from gapstop.cli import main
from gapstop.tests import PIXEL_SIZE, TOMO_ID, motl_row, write_em_motl, write_mrc
from gapstop.tests.tests_score_tomogram import EXTRACTION_SCRIPT
from gapstop.utils.utils import read_em_motl

GRID = (40, 30, 20)
ROWS = [
    motl_row(score=0.3, tomo_id=1, x=5, y=6, z=7, phi=30, psi=45, theta=60),
    motl_row(score=0.2, tomo_id=1, x=10, y=20, z=15, phi=10, psi=-20, theta=90),
]
# (x, y, z, phi, psi, theta, score) of ROWS
ROW_COLUMNS = [7, 8, 9, 16, 17, 18, 0]


def run(args: List[str]) -> int:
    """Runs cets-gapstop with the given arguments, hiding its messages."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return main(args)


class CetsGapstopCliTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_gapstop_cli"))
        self.motl = write_em_motl(self.dir / "particleList.em", ROWS)
        self.scores_map = write_mrc(self.dir / "scores_0_1.mrc", GRID, PIXEL_SIZE)
        self.yaml = self.dir / "cets.yaml"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _coordinates_to_cets(self, *extra: str) -> int:
        return run(
            [
                "coordinates",
                "to-cets",
                "--motl",
                str(self.motl),
                "--tomo-id",
                TOMO_ID,
                "-o",
                str(self.yaml),
                *extra,
            ]
        )

    def test_coordinates_to_cets(self) -> None:
        code = self._coordinates_to_cets(
            "--reference-map",
            str(self.scores_map),
            "--annotated-object",
            "ribosome",
            "--score-annotation-id",
            "gapstop_scores_TS_01",
        )
        self.assertEqual(code, 0)
        coords = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        self.assertEqual(coords.origin3D, [[4.0, 5.0, 6.0], [9.0, 19.0, 14.0]])
        self.assertEqual(coords.source_tomogram_id, TOMO_ID)
        self.assertEqual(coords.derived_from_annotation_id, "gapstop_scores_TS_01")
        assert coords.annotated_object is not None
        self.assertEqual(coords.annotated_object.name, "ribosome")
        # Sampling rate read from the reference map: physical frame attached
        self.assertEqual(len(coords.coordinate_systems or []), 2)

    def test_coordinates_reference_map_from_extraction_script(self) -> None:
        """The scores map recorded in extractCoords.py is used as reference map"""
        project = self.dir / "project"
        results = (
            project / "Runs/000301_ProtGapStopTemplateMatching/extra/TS_03/results"
        )
        results.mkdir(parents=True)
        write_mrc(results / "scores_0_1.mrc", GRID, 10.8)
        extract_dir = project / "Runs/000388_ProtGapStopExtractCoords/extra/TS_03"
        extract_dir.mkdir(parents=True)
        (extract_dir / "extractCoords.py").write_text(EXTRACTION_SCRIPT)
        self.motl = write_em_motl(extract_dir / "particleList.em", ROWS)

        self.assertEqual(self._coordinates_to_cets(), 0)
        coords = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        self.assertEqual(len(coords.coordinate_systems or []), 2)

        self.assertEqual(self._coordinates_to_cets("--no-extraction-script"), 0)
        coords = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        self.assertEqual(len(coords.coordinate_systems or []), 1)

    def test_coordinates_round_trip(self) -> None:
        self.assertEqual(self._coordinates_to_cets("--pixel-size", "5.4"), 0)
        out = self.dir / "out" / "sub" / "particleList.em"  # created by the command
        code = run(
            [
                "coordinates",
                "from-cets",
                "-i",
                str(self.yaml),
                "-o",
                str(out),
                "--tomo-num",
                "3",
                "--class-id",
                "2",
            ]
        )
        self.assertEqual(code, 0)
        motl = read_em_motl(out)
        np.testing.assert_allclose(
            motl[:, ROW_COLUMNS], np.array(ROWS)[:, ROW_COLUMNS], atol=1e-4
        )
        self.assertEqual(motl[:, 4].tolist(), [3.0, 3.0])  # tomo_id
        self.assertEqual(motl[:, 19].tolist(), [2.0, 2.0])  # class

    def test_coordinates_of_several_tomograms(self) -> None:
        self.motl = write_em_motl(
            self.dir / "merged.em",
            [*ROWS, motl_row(score=0.1, tomo_id=2, x=1, y=1, z=1)],
        )
        # Several tomograms: one must be chosen
        self.assertEqual(self._coordinates_to_cets(), 1)
        self.assertEqual(self._coordinates_to_cets("--tomo-num", "2"), 0)
        coords = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        self.assertEqual(coords.origin3D, [[0.0, 0.0, 0.0]])
        # No particles of the tomogram
        self.assertEqual(self._coordinates_to_cets("--tomo-num", "7"), 1)

    def test_scores_round_trip(self) -> None:
        code = run(
            [
                "scores",
                "to-cets",
                "--scores-map",
                str(self.scores_map),
                "--tomo-id",
                TOMO_ID,
                "--threshold",
                "0.08",
                "--annotation-id",
                "my_scores",
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(code, 0)
        scores = load_cets_object(self.yaml, "scores", ProbabilityMap3D)
        self.assertEqual(scores.id, "my_scores")
        self.assertEqual(scores.threshold, 0.08)
        self.assertEqual((scores.width, scores.height, scores.depth), GRID)

        out_dir = self.dir / "tm"
        code = run(
            [
                "scores",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(out_dir),
                "--tomo-num",
                "4",
                "--index",
                "1",
            ]
        )
        self.assertEqual(code, 0)
        out = out_dir / "scores_1_4.mrc"
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), self.scores_map.resolve())

    def test_errors_give_exit_code_1(self) -> None:
        self.motl = self.dir / "missing.em"
        self.assertEqual(self._coordinates_to_cets(), 1)
        # Coordinates are not a scores map
        self.yaml.write_text("coordinates: {id: x}\n")
        no_scores = [
            "scores",
            "from-cets",
            "-i",
            str(self.yaml),
            "--output-dir",
            str(self.dir / "tm"),
        ]
        self.assertEqual(run(no_scores), 1)
        # The output is not a motive list (.em)
        self.motl = write_em_motl(self.dir / "particleList.em", ROWS)
        self.assertEqual(self._coordinates_to_cets(), 0)
        not_em = [
            "coordinates",
            "from-cets",
            "-i",
            str(self.yaml),
            "-o",
            str(self.dir / "out.txt"),
        ]
        self.assertEqual(run(not_em), 1)
