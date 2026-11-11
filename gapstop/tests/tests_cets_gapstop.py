# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import yaml

from cets_data_model.models.models import (
    AnnotatedObject,
    AnnotationType,
    PointMatrixSet3D,
    Region,
)
from gapstop.converters.coordinates3d import GapstopCoordinates3D
from gapstop.tests import (
    PIXEL_SIZE,
    REAL_GRID,
    REAL_MOTL_EXCERPT,
    REAL_MOTL_EXCERPT_ROWS,
    REAL_PIXEL_SIZE,
    TOMO_ID,
    motl_row,
    write_em_motl,
    write_mrc,
)
from gapstop.utils.utils import (
    euler_zxz_to_matrix,
    read_em_motl,
    write_coordinates_yaml,
)


def rot_z(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class CetsGapstopBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_gapstop"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class CetsGapstopReaderTest(CetsGapstopBaseTest):
    def test_read_real_motl(self) -> None:
        motl = read_em_motl(REAL_MOTL_EXCERPT)
        self.assertEqual(motl.shape, (5, 20))
        np.testing.assert_allclose(
            motl[:, [7, 8, 9, 16, 17, 18]], REAL_MOTL_EXCERPT_ROWS, atol=1e-4
        )

    def test_reject_non_motl(self) -> None:
        not_motl = self.test_dir / "map.em"
        not_motl.write_bytes(bytes(512))  # machine code 0, data type code 0
        with self.assertRaises(ValueError):
            read_em_motl(not_motl)


class CetsGapstopRotationTest(TestCase):
    def test_euler_zxz(self) -> None:
        """R = Rz(psi) Rx(theta) Rz(phi): with theta = 0, a rotation of phi + psi around Z"""
        np.testing.assert_allclose(
            euler_zxz_to_matrix(30, 0, 45), rot_z(75), atol=1e-12
        )
        # theta = 90: Z goes to -Y (Rx(90) z = (0, -1, 0)), unaffected by phi
        rotated_z = euler_zxz_to_matrix(17, 90, 0) @ np.array([0, 0, 1])
        np.testing.assert_allclose(rotated_z, [0, -1, 0], atol=1e-12)

    def test_is_rotation(self) -> None:
        for phi, theta, psi in [
            (285, 88.26, -29.25),
            (225, 90, 23.27),
            (0, 180, 140.71),
        ]:
            r = euler_zxz_to_matrix(phi, theta, psi)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1.0)


class CetsGapstopCoordinatesTest(CetsGapstopBaseTest):
    def _motl(self, rows: list[list[float]]) -> Path:
        return write_em_motl(self.test_dir / "particleList.em", rows)

    def test_positions_are_zero_based_with_shifts(self) -> None:
        """origin3D = x + shift - 1 (1-based motive list -> 0-based array frame)"""
        motl = self._motl(
            [
                motl_row(score=0.3, tomo_id=1, x=1, y=1, z=1),
                motl_row(
                    score=0.2, tomo_id=1, x=10, y=20, z=30, shift_x=0.5, shift_z=-1
                ),
            ]
        )
        coords = GapstopCoordinates3D().gapstop_to_cets(motl, TOMO_ID)
        assert coords is not None
        self.assertEqual(coords.annotation_type, AnnotationType.point_matrix_set_3D)
        self.assertEqual(coords.origin3D, [[0.0, 0.0, 0.0], [9.5, 19.0, 28.0]])
        # One template matching score per particle, in the order of the positions
        assert coords.scores is not None
        np.testing.assert_allclose(coords.scores, [0.3, 0.2], atol=1e-6)
        self.assertEqual(coords.source_tomogram_id, TOMO_ID)
        self.assertEqual(coords.id, f"gapstop_coords_{TOMO_ID}")

    def test_orientations(self) -> None:
        motl = self._motl([motl_row(tomo_id=1, x=1, y=1, z=1, phi=30, psi=45, theta=0)])
        coords = GapstopCoordinates3D().gapstop_to_cets(motl, TOMO_ID)
        assert coords is not None
        np.testing.assert_allclose(coords.matrix3D[0], rot_z(75), atol=1e-6)

    def test_frames_and_pixel_size(self) -> None:
        motl = self._motl([motl_row(tomo_id=1, x=1, y=1, z=1)])
        converter = GapstopCoordinates3D()
        # No sampling rate: array frame only
        coords = converter.gapstop_to_cets(motl, TOMO_ID)
        assert coords is not None
        self.assertEqual(
            [cs.name for cs in coords.coordinate_systems], [f"{TOMO_ID}_array"]
        )
        self.assertEqual(coords.coordinate_transformations, [])
        # Given sampling rate
        coords = converter.gapstop_to_cets(motl, TOMO_ID, pixel_size=PIXEL_SIZE)
        assert coords is not None
        self.assertEqual(
            [cs.name for cs in coords.coordinate_systems],
            [f"{TOMO_ID}_array", f"{TOMO_ID}_physical"],
        )
        self.assertEqual(coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)
        # From the reference map header, unless given
        ref = write_mrc(self.test_dir / "scores.mrc", (4, 4, 4), 13.5)
        coords = converter.gapstop_to_cets(motl, TOMO_ID, reference_map=ref)
        assert coords is not None
        self.assertAlmostEqual(
            coords.coordinate_transformations[0].scale[0], 13.5, places=5
        )
        coords = converter.gapstop_to_cets(
            motl, TOMO_ID, pixel_size=PIXEL_SIZE, reference_map=ref
        )
        assert coords is not None
        self.assertEqual(coords.coordinate_transformations[0].scale[0], PIXEL_SIZE)
        with self.assertRaises(ValueError):
            converter.gapstop_to_cets(motl, TOMO_ID, pixel_size=-1.0)

    def test_reference_map_grid_check(self) -> None:
        """The last voxel of the grid (1-based x = nx) is inside; one more is outside"""
        ref = write_mrc(self.test_dir / "tomo.mrc", (4, 5, 6), PIXEL_SIZE)
        inside = self._motl([motl_row(tomo_id=1, x=4, y=5, z=6)])
        self.assertIsNotNone(
            GapstopCoordinates3D().gapstop_to_cets(inside, TOMO_ID, reference_map=ref)
        )
        outside = self._motl([motl_row(tomo_id=1, x=5, y=5, z=6)])
        with self.assertRaises(ValueError):
            GapstopCoordinates3D().gapstop_to_cets(outside, TOMO_ID, reference_map=ref)

    def test_several_tomograms(self) -> None:
        motl = self._motl(
            [
                motl_row(tomo_id=1, x=1, y=1, z=1),
                motl_row(tomo_id=2, x=2, y=2, z=2),
                motl_row(tomo_id=2, x=3, y=3, z=3),
            ]
        )
        converter = GapstopCoordinates3D()
        self.assertEqual(converter.get_tomo_nums(motl), [1, 2])
        with self.assertRaises(ValueError):
            converter.gapstop_to_cets(motl, TOMO_ID)  # tomo_num required
        coords = converter.gapstop_to_cets(motl, "TS_02", tomo_num=2)
        assert coords is not None
        self.assertEqual(coords.origin3D, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        self.assertIsNone(converter.gapstop_to_cets(motl, "TS_09", tomo_num=9))

    def test_annotated_object_and_id(self) -> None:
        motl = self._motl([motl_row(tomo_id=1, x=1, y=1, z=1)])
        coords = GapstopCoordinates3D().gapstop_to_cets(
            motl,
            TOMO_ID,
            annotated_object=AnnotatedObject(feature_type="other", name="ribosome"),
            annotation_id="run_2",
        )
        assert coords is not None
        self.assertEqual(coords.id, "run_2")
        self.assertEqual(coords.annotated_object.name, "ribosome")

    def test_real_motl_excerpt(self) -> None:
        converter = GapstopCoordinates3D()
        coords = converter.gapstop_to_cets(
            REAL_MOTL_EXCERPT, "TS_03", pixel_size=REAL_PIXEL_SIZE
        )
        assert coords is not None
        expected = np.array(REAL_MOTL_EXCERPT_ROWS)
        np.testing.assert_allclose(coords.origin3D, expected[:, :3] - 1)
        for matrix, (phi, psi, theta) in zip(coords.matrix3D, expected[:, 3:]):
            np.testing.assert_allclose(
                matrix, euler_zxz_to_matrix(phi, theta, psi), atol=1e-5
            )
        # All the particles are inside the voxel grid of TS_03
        self.assertTrue(np.all(np.array(coords.origin3D) <= np.array(REAL_GRID) - 1))
        # Real scores: one per particle, sorted in descending order by cryoCAT
        assert coords.scores is not None
        self.assertEqual(len(coords.scores), len(coords.origin3D))
        self.assertEqual(coords.scores, sorted(coords.scores, reverse=True))
        self.assertTrue(all(0 < score <= 1 for score in coords.scores))

    def test_yaml_roundtrip_in_region(self) -> None:
        coords = GapstopCoordinates3D().gapstop_to_cets(
            REAL_MOTL_EXCERPT, "TS_03", pixel_size=REAL_PIXEL_SIZE
        )
        assert coords is not None
        write_coordinates_yaml(coords, self.test_dir)
        with open(self.test_dir / f"{coords.id}_gapstop_to_cets.yaml") as f:
            data = yaml.safe_load(f)
        region = Region(id="TS_03", annotations=[data])
        reloaded = region.annotations[0]
        self.assertIsInstance(reloaded, PointMatrixSet3D)
        self.assertEqual(reloaded, coords)
