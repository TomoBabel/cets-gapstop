import os
from typing import Optional

from cets_data_model.models.models import (
    AnnotatedObject,
    AnnotationType,
    ProbabilityMap3D,
    ProbabilityMapValueType,
)
from cets_data_model.utils.image_utils import get_mrc_info
from gapstop.constants import MRC_EXT
from gapstop.utils.utils import (
    gen_array_to_physical,
    gen_coordinate_systems,
    validate_file,
)


class GapstopScoreTomogram:
    def gapstop_to_cets(
        self,
        scores_map: os.PathLike,
        tomo_id: str,
        pixel_size: Optional[float] = None,
        threshold: Optional[float] = None,
        annotated_object: Optional[AnnotatedObject] = None,
        annotation_id: Optional[str] = None,
    ) -> ProbabilityMap3D:
        """Converts a GAPSTOP(TM) scores map (ScoreTomogram, scores_<i>_<tomo_num>.mrc) into
        CETS metadata.

        The scores map holds, for each voxel of the tomogram, the best template matching score
        over all the sampled orientations, so it is represented as a ``ProbabilityMap3D``
        annotation (``value_type: score``, stored under ``Region.annotations``) on the voxel
        grid of the tomogram, linked to it through ``source_tomogram_id``. The particles
        extracted from it refer to it through ``derived_from_annotation_id``.

        :param scores_map: path to the scores map.
        :type scores_map: os.PathLike.

        :param tomo_id: identifier of the CETS tomogram that was template-matched. It is used as
        ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of the
        CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param pixel_size: sampling rate (Å/px). If given, it is used; otherwise it is read from
        the mrc header (scipion-em-gapstop writes the tomogram sampling rate there).
        :type pixel_size: float, optional. Defaults to None.

        :param threshold: score above which the voxels were considered particle candidates in
        the coordinate extraction (scores_threshold), if known.
        :type threshold: float, optional. Defaults to None.

        :param annotated_object: what the template represents. Template matching does not
        record it.
        :type annotated_object: AnnotatedObject, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations. Defaults to ``get_annotation_id(tomo_id)``.
        :type annotation_id: str, optional. Defaults to None.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        scores_path = validate_file(scores_map, expected_ext=MRC_EXT)
        mrc_info = get_mrc_info(scores_path)
        voxel_size = pixel_size or mrc_info.apix_x
        if not voxel_size:
            raise ValueError(
                f"No pixel size was provided and {scores_path} does not contain it in its "
                f"header."
            )
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        return ProbabilityMap3D(
            id=annotation_id or self.get_annotation_id(tomo_id),
            name=f"GAPSTOP(TM) template matching scores for {tomo_id}",
            annotation_type=AnnotationType.probability_map_3D,
            source_tomogram_id=tomo_id,
            annotated_object=annotated_object,
            value_type=ProbabilityMapValueType.score,
            threshold=threshold,
            path=str(scores_path),
            width=mrc_info.size_x,
            height=mrc_info.size_y,
            depth=mrc_info.size_z,
            coordinate_systems=[array_cs, physical_cs],
            coordinate_transformations=[
                gen_array_to_physical(
                    voxel_size, array_cs.name, physical_cs.name, ndim=3
                )
            ],
        )

    @staticmethod
    def get_annotation_id(tomo_id: str) -> str:
        """Default id of the scores map annotation."""
        return f"gapstop_scores_{tomo_id}"
