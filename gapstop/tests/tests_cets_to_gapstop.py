# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import ProbabilityMapValueType
from gapstop.converters.coordinates3d import GapstopCoordinates3D
from gapstop.converters.score_tomogram import GapstopScoreTomogram
from gapstop.tests import (
    REAL_MOTL_EXCERPT,
    REAL_PIXEL_SIZE,
    TOMO_ID,
    motl_row,
    write_em_motl,
    write_mrc,
)
from gapstop.utils.utils import (
    euler_zxz_to_matrix,
    matrix_to_euler_zxz,
    read_em_motl,
)


class CetsToGapstopBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_gapstop"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class MatrixToEulerTest(TestCase):
    def test_inverse_of_euler_zxz_to_matrix(self) -> None:
        for phi, theta, psi in [(285, 88.26, -29.25), (10, 1, 20), (359, 179, 180)]:
            np.testing.assert_allclose(
                matrix_to_euler_zxz(euler_zxz_to_matrix(phi, theta, psi)),
                (phi, theta, psi),
                atol=1e-9,
            )

    def test_gimbal_lock_keeps_the_rotation(self) -> None:
        """With theta 0 or 180 the angles are not unique, but the rotation is kept"""
        for phi, theta, psi in [(30, 0, 45), (30, 180, 45), (0, 0, 0)]:
            matrix = euler_zxz_to_matrix(phi, theta, psi)
            new_phi, new_theta, new_psi = matrix_to_euler_zxz(matrix)
            self.assertEqual(new_psi, 0.0)
            np.testing.assert_allclose(
                euler_zxz_to_matrix(new_phi, new_theta, new_psi), matrix, atol=1e-12
            )


class CetsToGapstopCoordinatesTest(CetsToGapstopBaseTest):
    def test_real_motl_roundtrip(self) -> None:
        """motive list -> CETS -> motive list gives back the original values"""
        converter = GapstopCoordinates3D()
        coords = converter.gapstop_to_cets(
            REAL_MOTL_EXCERPT, "TS_03", pixel_size=REAL_PIXEL_SIZE
        )
        assert coords is not None
        out_motl = converter.cets_to_gapstop(coords, self.test_dir / "particleList.em")
        np.testing.assert_allclose(
            read_em_motl(out_motl), read_em_motl(REAL_MOTL_EXCERPT), atol=1e-4
        )

    def test_positions_and_shifts(self) -> None:
        """Positions are 1-based and split into the nearest voxel and a sub-voxel shift"""
        motl = write_em_motl(
            self.test_dir / "in.em",
            [
                motl_row(score=0.3, tomo_id=1, x=1, y=1, z=1),
                motl_row(score=0.2, tomo_id=1, x=10, y=20, z=30, shift_x=0.25),
            ],
        )
        converter = GapstopCoordinates3D()
        coords = converter.gapstop_to_cets(motl, TOMO_ID)
        assert coords is not None
        self.assertEqual(coords.origin3D, [[0.0, 0.0, 0.0], [9.25, 19.0, 29.0]])
        out = read_em_motl(
            converter.cets_to_gapstop(coords, self.test_dir / "out.em", tomo_num=7)
        )
        np.testing.assert_allclose(out[:, 7:10], [[1, 1, 1], [10, 20, 30]])
        np.testing.assert_allclose(out[:, 10:13], [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        np.testing.assert_allclose(out[:, 0], [0.3, 0.2], atol=1e-6)
        # subtomo_id, tomo_id, object_id and class
        np.testing.assert_allclose(out[:, 3], [1, 2])
        np.testing.assert_allclose(out[:, [4, 5, 19]], [[7, 1, 1], [7, 1, 1]])

    def test_without_orientations_nor_scores(self) -> None:
        converter = GapstopCoordinates3D()
        coords = converter.gapstop_to_cets(REAL_MOTL_EXCERPT, "TS_03")
        assert coords is not None
        coords = coords.model_copy(update={"matrix3D": None, "scores": None})
        out = read_em_motl(converter.cets_to_gapstop(coords, self.test_dir / "out.em"))
        self.assertFalse(out[:, [0, 16, 17, 18]].any())

    def test_invalid_input(self) -> None:
        converter = GapstopCoordinates3D()
        coords = converter.gapstop_to_cets(REAL_MOTL_EXCERPT, "TS_03")
        assert coords is not None
        with self.assertRaises(ValueError):
            converter.cets_to_gapstop(coords, self.test_dir / "particles.star")
        with self.assertRaises(ValueError):
            converter.cets_to_gapstop(
                coords.model_copy(update={"scores": [0.1]}), self.test_dir / "out.em"
            )


class CetsToGapstopScoresTest(CetsToGapstopBaseTest):
    def _scores_map(self, voxel_size: float = 10.8) -> Path:
        return write_mrc(self.test_dir / "scores_0_1.mrc", (8, 8, 4), voxel_size)

    def test_linked_with_the_gapstop_name(self) -> None:
        converter = GapstopScoreTomogram()
        scores_map = self._scores_map()
        scores = converter.gapstop_to_cets(scores_map, "TS_03")
        out = converter.cets_to_gapstop(scores, self.test_dir / "results", tomo_num=2)
        self.assertEqual(out.name, "scores_0_2.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), scores_map.resolve())
        # Exporting it again is harmless; another file with that name is not overwritten
        self.assertEqual(
            converter.cets_to_gapstop(scores, self.test_dir / "results", tomo_num=2),
            out,
        )
        other = converter.gapstop_to_cets(
            write_mrc(self.test_dir / "other.mrc", (8, 8, 4), 10.8), "TS_03"
        )
        with self.assertRaises(FileExistsError):
            converter.cets_to_gapstop(other, self.test_dir / "results", tomo_num=2)

    def test_copied_if_the_sampling_rate_differs(self) -> None:
        """The header of the copy gets the sampling rate of the annotation"""
        converter = GapstopScoreTomogram()
        scores_map = self._scores_map(voxel_size=1.0)
        scores = converter.gapstop_to_cets(scores_map, "TS_03", pixel_size=10.8)
        out = converter.cets_to_gapstop(scores, self.test_dir / "results")
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 10.8, places=5)
        # The whole file is copied, not only its header
        self.assertEqual(out.stat().st_size, scores_map.stat().st_size)
        self.assertEqual(
            mrcfile.read(out).tobytes(), mrcfile.read(scores_map).tobytes()
        )
        with mrcfile.open(scores_map, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 1.0, places=5)

    def test_not_a_scores_map(self) -> None:
        converter = GapstopScoreTomogram()
        scores = converter.gapstop_to_cets(self._scores_map(), "TS_03")
        probabilities = scores.model_copy(
            update={"value_type": ProbabilityMapValueType.probability.value}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_gapstop(probabilities, self.test_dir / "results")
