import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from cets_data_model.models.models import (
    AnnotatedObject,
    AnnotationType,
    PointMatrixSet3D,
)
from cets_data_model.utils.image_utils import get_mrc_info
from gapstop.constants import (
    CLASS,
    EM_EXT,
    MOTL_POSITION_OFFSET,
    N_MOTL_COLUMNS,
    OBJECT_ID,
    PHI,
    PSI,
    SCORE,
    SHIFT_X,
    SHIFT_Y,
    SHIFT_Z,
    SUBTOMO_ID,
    THETA,
    TOMO_ID,
    X,
    Y,
    Z,
)
from gapstop.utils.utils import (
    euler_zxz_to_matrix,
    gen_array_to_physical,
    gen_coordinate_systems,
    matrix_to_euler_zxz,
    read_em_motl,
    validate_file,
    write_em_motl,
)


class GapstopCoordinates3D:
    def gapstop_to_cets(
        self,
        motl_file: os.PathLike,
        tomo_id: str,
        tomo_num: Optional[int] = None,
        pixel_size: Optional[float] = None,
        reference_map: Optional[os.PathLike] = None,
        annotated_object: Optional[AnnotatedObject] = None,
        score_annotation_id: Optional[str] = None,
        annotation_id: Optional[str] = None,
    ) -> Optional[PointMatrixSet3D]:
        """Converts the particles picked by GAPSTOP(TM) template matching (a motive list in EM
        format, written by cryocat.tmana.scores_extract_particles, e.g. particleList.em) for one
        tomogram into CETS metadata.

        The picked particles are represented as a ``PointMatrixSet3D`` annotation (stored under
        ``Region.annotations``) that links back to the tomogram through ``source_tomogram_id``:

        * ``origin3D``: the particle positions in the tomogram array (voxel) frame. Motive list
          positions are 1-based, so they are converted to the 0-based CETS array frame, adding
          the per-particle shifts first, as cryoCAT does to get the particle coordinates.
        * ``matrix3D``: the rotation matrix of each particle, built from its Euler ZXZ angles
          as R = Rz(psi) @ Rx(theta) @ Rz(phi) (see ``euler_zxz_to_matrix``).

        * ``scores``: the template matching score of each particle (score column).

        The particles keep the order of the motive list (descending score).

        :param motl_file: path to the motive list (.em).
        :type motl_file: os.PathLike.

        :param tomo_id: identifier of the CETS tomogram the particles were picked in. It is used
        as ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of
        the CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param tomo_num: numeric tomogram identifier (tomo_id column) of the particles to be
        converted, as a motive list may contain the particles of several tomograms. When
        omitted, the motive list must contain the particles of a single tomogram.
        :type tomo_num: int, optional. Defaults to None.

        :param pixel_size: sampling rate (Å/px) of the tomogram. If given, it is used;
        otherwise it is read from the header of ``reference_map``, if given. The motive list
        does not record it, so without any of them only the array (voxel) frame is attached.
        :type pixel_size: float, optional. Defaults to None.

        :param reference_map: an mrc file on the voxel grid the particles were picked in (the
        tomogram or the GAPSTOP(TM) scores map). Used to read the sampling rate when it is not
        given, and to check that all the particles are inside the grid.
        :type reference_map: os.PathLike, optional. Defaults to None.

        :param annotated_object: what the picked particles are (e.g. the template structure).
        Template matching does not record it.
        :type annotated_object: AnnotatedObject, optional. Defaults to None.

        :param score_annotation_id: id of the annotation of the scores map (ScoreTomogram) the
        particles were extracted from (see ``GapstopScoreTomogram``), stored as
        ``derived_from_annotation_id``. Optional: the motive list does not record it (see
        ``gapstop.utils.extraction.find_extraction_params``).
        :type score_annotation_id: str, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations. Defaults to ``get_annotation_id(tomo_id)``.
        :type annotation_id: str, optional. Defaults to None.

        :returns: the coordinates annotation, or None if there are no particles of the tomogram.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        motl_path = validate_file(motl_file, expected_ext=EM_EXT)
        motl = self._select_tomogram(read_em_motl(motl_path), tomo_num, motl_path)
        if motl.shape[0] == 0:
            return None

        positions = motl[:, [X, Y, Z]] + motl[:, [SHIFT_X, SHIFT_Y, SHIFT_Z]]
        origin_3d = positions - MOTL_POSITION_OFFSET
        matrix_3d = [
            euler_zxz_to_matrix(phi, theta, psi).tolist()
            for phi, theta, psi in motl[:, [PHI, THETA, PSI]]
        ]

        if reference_map is not None:
            mrc_info = get_mrc_info(reference_map)
            self._check_inside_grid(
                origin_3d,
                (mrc_info.size_x, mrc_info.size_y, mrc_info.size_z),
                reference_map,
            )
            if pixel_size is None and mrc_info.apix_x:
                pixel_size = mrc_info.apix_x

        # Picked coordinates are positions in the tomogram's array (voxel) frame. When the
        # tomogram sampling rate is known, the physical (Å) system and the canonical
        # array_to_physical scale are also attached, making the coordinates self-describing.
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        if pixel_size is not None:
            coordinate_systems = [array_cs, physical_cs]
            coordinate_transformations = [
                gen_array_to_physical(
                    pixel_size, array_cs.name, physical_cs.name, ndim=3
                )
            ]
        else:
            coordinate_systems = [array_cs]
            coordinate_transformations = []

        return PointMatrixSet3D(
            id=annotation_id or self.get_annotation_id(tomo_id),
            name=f"GAPSTOP(TM) picked coordinates for {tomo_id}",
            annotation_type=AnnotationType.point_matrix_set_3D,
            source_tomogram_id=tomo_id,
            annotated_object=annotated_object,
            derived_from_annotation_id=score_annotation_id,
            origin3D=origin_3d.tolist(),
            matrix3D=matrix_3d,
            scores=motl[:, SCORE].tolist(),
            coordinate_systems=coordinate_systems,
            coordinate_transformations=coordinate_transformations,
        )

    def cets_to_gapstop(
        self,
        coordinates: PointMatrixSet3D,
        motl_file: os.PathLike,
        tomo_num: int = 1,
        object_id: int = 1,
        class_id: int = 1,
    ) -> Path:
        """Writes the particles of a CETS ``PointMatrixSet3D`` annotation as a motive list in
        EM format, as cryocat.tmana.scores_extract_particles does (output_type="emmotl").

        Only the columns CETS holds are filled:

        * x, y, z and shift_x, shift_y, shift_z: ``origin3D`` converted to the 1-based
          positions of the motive list. Motive lists store the position as an integer voxel
          plus a sub-voxel shift, so each position is split into the nearest voxel and the
          rest (0 for the particles picked by GAPSTOP(TM), which are on voxels).
        * phi, psi, theta: the Euler ZXZ angles of ``matrix3D`` (see ``matrix_to_euler_zxz``).
          When theta is 0 or 180 they may differ from the original ones, but they describe
          the same rotation. 0 if the annotation has no orientations.
        * score: ``scores``. 0 if the annotation has no scores.
        * subtomo_id: 1..N, in the order of the annotation.
        * tomo_id, object_id and class: the given ``tomo_num``, ``object_id`` and
          ``class_id``, as CETS does not store them.

        The rest of the columns (geom1-5, subtomo_mean) are written as 0. EM motive lists are
        float32, so the values keep about 7 significant digits.

        :param coordinates: the picked particles.
        :type coordinates: PointMatrixSet3D.

        :param motl_file: path of the motive list (.em) to be written.
        :type motl_file: os.PathLike.

        :param tomo_num: numeric tomogram identifier (tomo_id column) of the particles.
        :type tomo_num: int, optional. Defaults to 1.

        :param object_id: object identifier (object_id column) of the particles.
        :type object_id: int, optional. Defaults to 1, as cryoCAT writes.

        :param class_id: class (class column) of the particles.
        :type class_id: int, optional. Defaults to 1, as cryoCAT writes.

        :returns: the path of the motive list written.
        """
        motl_path = Path(motl_file).expanduser()
        if motl_path.suffix != EM_EXT:
            raise ValueError(
                f"Invalid file extension '{motl_path.suffix}'. Expected: {EM_EXT}"
            )
        origin_3d = np.array(coordinates.origin3D or [], dtype=float)
        if origin_3d.size == 0:
            raise ValueError(f"The annotation [{coordinates.id}] has no particles.")
        n_particles = origin_3d.shape[0]
        for field, values in (
            ("matrix3D", coordinates.matrix3D),
            ("scores", coordinates.scores),
        ):
            if values is not None and len(values) != n_particles:
                raise ValueError(
                    f"The annotation [{coordinates.id}] has {n_particles} particles, but "
                    f"{len(values)} {field}."
                )

        motl = np.zeros((n_particles, N_MOTL_COLUMNS))
        positions = origin_3d + MOTL_POSITION_OFFSET
        voxels = np.floor(positions + 0.5)
        motl[:, [X, Y, Z]] = voxels
        motl[:, [SHIFT_X, SHIFT_Y, SHIFT_Z]] = positions - voxels
        if coordinates.matrix3D is not None:
            motl[:, [PHI, THETA, PSI]] = [
                matrix_to_euler_zxz(np.array(matrix)) for matrix in coordinates.matrix3D
            ]
        if coordinates.scores is not None:
            motl[:, SCORE] = coordinates.scores
        motl[:, SUBTOMO_ID] = np.arange(1, n_particles + 1)
        motl[:, TOMO_ID] = tomo_num
        motl[:, OBJECT_ID] = object_id
        motl[:, CLASS] = class_id
        write_em_motl(motl, motl_path)
        print(f"motive list successfully written! -> {motl_path}")
        return motl_path

    @staticmethod
    def get_annotation_id(tomo_id: str) -> str:
        """Default id of the coordinates annotation."""
        return f"gapstop_coords_{tomo_id}"

    @staticmethod
    def get_tomo_nums(motl_file: os.PathLike) -> List[int]:
        """Numeric tomogram identifiers (tomo_id column) present in a motive list."""
        motl = read_em_motl(validate_file(motl_file, expected_ext=EM_EXT))
        return sorted({int(t) for t in motl[:, TOMO_ID]})

    @staticmethod
    def _select_tomogram(
        motl: np.ndarray, tomo_num: Optional[int], motl_path: os.PathLike
    ) -> np.ndarray:
        tomo_nums = sorted({int(t) for t in motl[:, TOMO_ID]})
        if tomo_num is None:
            if len(tomo_nums) > 1:
                raise ValueError(
                    f"{motl_path} contains the particles of several tomograms {tomo_nums}; "
                    f"choose one with tomo_num."
                )
            return motl
        return motl[motl[:, TOMO_ID] == tomo_num]

    @staticmethod
    def _check_inside_grid(
        origin_3d: np.ndarray, dims: tuple[int, int, int], reference_map: os.PathLike
    ) -> None:
        outside = np.any((origin_3d < 0) | (origin_3d > np.array(dims) - 1), axis=1)
        if np.any(outside):
            raise ValueError(
                f"{int(outside.sum())} particles fall outside the {dims[0]} x {dims[1]} x "
                f"{dims[2]} voxel grid of {reference_map}; it is not the map they were "
                f"picked in."
            )
