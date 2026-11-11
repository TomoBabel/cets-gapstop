# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from cets_data_model.models.models import (
    PointMatrixSet3D,
    ProbabilityMap3D,
    ProbabilityMapValueType,
    Region,
)
from gapstop.converters.coordinates3d import GapstopCoordinates3D
from gapstop.converters.score_tomogram import GapstopScoreTomogram
from gapstop.tests import TEST_DATA_DIR, motl_row, write_em_motl, write_mrc
from gapstop.utils.extraction import find_extraction_params, read_extraction_call

# Verbatim copy of the extraction script written by scipion-em-gapstop (TestGapStopTM, run 000388,
# TS_03). Its paths are relative to that Scipion project, so they are not found here.
REAL_EXTRACTION_SCRIPT = TEST_DATA_DIR / "TS_03_extractCoords.py.txt"

EXTRACTION_SCRIPT = """from cryocat import tmana

tmana.scores_extract_particles(
scores_map='Runs/000301_ProtGapStopTemplateMatching/extra/TS_03/results/scores_0_1.mrc',
angles_map='Runs/000301_ProtGapStopTemplateMatching/extra/TS_03/results/angles_0_1.mrc',
tomo_id=1,
particle_diameter=10.0,
scores_threshold=0.08449986168742374,
output_path='Runs/000388_ProtGapStopExtractCoords/extra/TS_03/particleList.em',
symmetry='C6',
tomo_mask=None)
"""


class CetsGapstopExtractionParamsTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_gapstop_scores"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _project(self, script: str = EXTRACTION_SCRIPT) -> tuple[Path, Path]:
        """A Scipion-like project: the scores map in the template matching run, and the motive
        list with its extraction script in the extraction run."""
        tm_results = (
            self.test_dir
            / "Runs/000301_ProtGapStopTemplateMatching/extra/TS_03/results"
        )
        tm_results.mkdir(parents=True)
        scores_map = write_mrc(tm_results / "scores_0_1.mrc", (8, 8, 4), 10.8)
        extract_dir = self.test_dir / "Runs/000388_ProtGapStopExtractCoords/extra/TS_03"
        extract_dir.mkdir(parents=True)
        (extract_dir / "extractCoords.py").write_text(script)
        motl = write_em_motl(
            extract_dir / "particleList.em",
            [motl_row(score=0.2, tomo_id=1, x=2, y=3, z=4)],
        )
        return motl, scores_map

    def test_real_extraction_script(self) -> None:
        kwargs = read_extraction_call(REAL_EXTRACTION_SCRIPT)
        assert kwargs is not None
        self.assertEqual(kwargs["tomo_id"], 1)
        self.assertAlmostEqual(kwargs["scores_threshold"], 0.08449986168742374)
        self.assertEqual(kwargs["symmetry"], "C6")
        self.assertTrue(kwargs["scores_map"].endswith("TS_03/results/scores_0_1.mrc"))
        # Next to a motive list, outside its project: the scores map is not found
        motl_dir = self.test_dir / "TS_03"
        motl_dir.mkdir()
        shutil.copy(REAL_EXTRACTION_SCRIPT, motl_dir / "extractCoords.py")
        params = find_extraction_params(motl_dir / "particleList.em")
        assert params is not None
        self.assertIsNone(params.scores_map)
        self.assertEqual(params.tomo_num, 1)
        self.assertEqual(params.particle_diameter, 10.0)

    def test_scores_map_resolved_in_project(self) -> None:
        motl, scores_map = self._project()
        params = find_extraction_params(motl)
        assert params is not None
        self.assertEqual(params.scores_map, scores_map)
        self.assertEqual(params.tomo_num, 1)
        self.assertAlmostEqual(params.scores_threshold or 0, 0.08449986168742374)

    def test_missing_or_unusable_script(self) -> None:
        """The link is optional: no script, or one that cannot be used, gives no parameters"""
        motl = write_em_motl(self.test_dir / "particleList.em", [motl_row(tomo_id=1)])
        self.assertIsNone(find_extraction_params(motl))
        script = self.test_dir / "extractCoords.py"
        for content in ("this is not python (", "print('no extraction call')"):
            script.write_text(content)
            self.assertIsNone(find_extraction_params(motl))
        # Non-literal arguments are ignored
        script.write_text("tmana.scores_extract_particles(scores_map=path, tomo_id=2)")
        params = find_extraction_params(motl)
        assert params is not None
        self.assertIsNone(params.scores_map)
        self.assertEqual(params.tomo_num, 2)

    def test_script_is_never_executed(self) -> None:
        marker = self.test_dir / "executed"
        script = f"open({str(marker)!r}, 'w').close()\n" + EXTRACTION_SCRIPT
        motl, _ = self._project(script)
        self.assertIsNotNone(find_extraction_params(motl))
        self.assertFalse(marker.exists())

    def test_scores_map_and_linked_coordinates(self) -> None:
        motl, scores_map = self._project()
        params = find_extraction_params(motl)
        assert params is not None and params.scores_map is not None
        scores = GapstopScoreTomogram().gapstop_to_cets(
            params.scores_map, "TS_03", threshold=params.scores_threshold
        )
        self.assertEqual(scores.id, "gapstop_scores_TS_03")
        self.assertEqual(scores.value_type, ProbabilityMapValueType.score)
        self.assertAlmostEqual(scores.threshold or 0, 0.08449986168742374)
        self.assertEqual((scores.width, scores.height, scores.depth), (8, 8, 4))
        self.assertAlmostEqual(
            scores.coordinate_transformations[0].scale[0], 10.8, places=5
        )
        coords = GapstopCoordinates3D().gapstop_to_cets(
            motl,
            "TS_03",
            reference_map=params.scores_map,
            score_annotation_id=scores.id,
        )
        assert coords is not None
        self.assertEqual(coords.derived_from_annotation_id, scores.id)
        # Sampling rate from the scores map header
        self.assertAlmostEqual(
            coords.coordinate_transformations[0].scale[0], 10.8, places=5
        )
        # Both validate together in a region, and the link resolves
        region = Region(
            id="TS_03",
            annotations=[
                scores.model_dump(mode="json"),
                coords.model_dump(mode="json"),
            ],
        )
        by_id = {a.id: a for a in region.annotations}
        self.assertIsInstance(by_id[scores.id], ProbabilityMap3D)
        self.assertIsInstance(by_id[coords.id], PointMatrixSet3D)
        self.assertIn(by_id[coords.id].derived_from_annotation_id, by_id)

    def test_coordinates_without_link(self) -> None:
        motl, _ = self._project()
        coords = GapstopCoordinates3D().gapstop_to_cets(motl, "TS_03")
        assert coords is not None
        self.assertIsNone(coords.derived_from_annotation_id)

    def test_invalid_pixel_size(self) -> None:
        _, scores_map = self._project()
        with self.assertRaises(ValueError):
            GapstopScoreTomogram().gapstop_to_cets(scores_map, "TS_03", pixel_size=0.0)
