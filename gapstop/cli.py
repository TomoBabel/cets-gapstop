"""Command line of the GAPSTOP(TM) <-> CETS converters (``cets-gapstop``).

Examples::

    cets-gapstop coordinates to-cets --motl particleList.em --tomo-id TS_03 \\
        --pixel-size 10.8 -o TS_03_coords.yaml
    cets-gapstop coordinates from-cets -i TS_03_coords.yaml -o out/particleList.em
    cets-gapstop scores to-cets --scores-map scores_0_1.mrc --tomo-id TS_03 \\
        --threshold 0.084 -o TS_03_scores.yaml
    cets-gapstop scores from-cets -i TS_03_scores.yaml --output-dir out --tomo-num 1

``to-cets`` writes a CETS yaml file with one entry per kind of object (``coordinates``,
``scores``), which is what ``from-cets`` reads.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cets_data_model.models.models import (
    AnnotatedObject,
    PointMatrixSet3D,
    ProbabilityMap3D,
    SegmentedFeatureType,
)
from cets_data_model.utils.cli import (
    FROM_CETS,
    TO_CETS,
    add_data_type,
    add_input_yaml,
    add_output_yaml,
    check_outputs,
    prepare_outputs,
    run_cli,
)
from cets_data_model.utils.yaml_io import load_cets_object, write_cets_yaml
from gapstop.converters.coordinates3d import GapstopCoordinates3D
from gapstop.converters.score_tomogram import GapstopScoreTomogram
from gapstop.utils.extraction import find_extraction_params

COORDINATES_KEY = "coordinates"
SCORES_KEY = "scores"


def _annotated_object(name: Optional[str]) -> Optional[AnnotatedObject]:
    if name is None:
        return None
    return AnnotatedObject(feature_type=SegmentedFeatureType.other, name=name)


def _add_annotation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tomo-id",
        required=True,
        help="Id of the CETS tomogram (source_tomogram_id). It names the coordinate "
        "systems, so it must match the id of the CETS tomogram.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px) of the tomogram.",
    )
    parser.add_argument(
        "--annotated-object",
        help="Name of what the template represents (e.g. ribosome). Template matching "
        "does not record it.",
    )
    parser.add_argument(
        "--annotation-id",
        help="Id of the annotation, unique within its region. Default: "
        "gapstop_coords_<tomo id> / gapstop_scores_<tomo id>.",
    )


# ------------------------------------------------------------------------- coordinates
def coordinates_to_cets(args: argparse.Namespace) -> None:
    converter = GapstopCoordinates3D()
    reference_map = args.reference_map
    if reference_map is None and not args.no_extraction_script:
        # scipion-em-gapstop records the scores map the particles were extracted from in
        # the extraction script next to the motive list
        params = find_extraction_params(args.motl)
        tomo_num = args.tomo_num
        if tomo_num is None:
            tomo_nums = converter.get_tomo_nums(args.motl)
            tomo_num = tomo_nums[0] if len(tomo_nums) == 1 else None
        if (
            params is not None
            and params.scores_map is not None
            and params.tomo_num in (None, tomo_num)
        ):
            reference_map = params.scores_map
            print(
                f"Using the scores map of the extraction as reference: {reference_map}"
            )
    coordinates = converter.gapstop_to_cets(
        motl_file=args.motl,
        tomo_id=args.tomo_id,
        tomo_num=args.tomo_num,
        pixel_size=args.pixel_size,
        reference_map=reference_map,
        annotated_object=_annotated_object(args.annotated_object),
        score_annotation_id=args.score_annotation_id,
        annotation_id=args.annotation_id,
    )
    if coordinates is None:
        which = f" of tomogram {args.tomo_num}" if args.tomo_num is not None else ""
        raise ValueError(f"{args.motl} has no particles{which}.")
    out = write_cets_yaml(args.output, {COORDINATES_KEY: coordinates})
    print(f"CETS coordinates written -> {out}")


def coordinates_from_cets(args: argparse.Namespace) -> None:
    coordinates = load_cets_object(args.input, COORDINATES_KEY, PointMatrixSet3D)
    outputs = prepare_outputs(args.output)
    GapstopCoordinates3D().cets_to_gapstop(
        coordinates,
        args.output,
        tomo_num=args.tomo_num,
        object_id=args.object_id,
        class_id=args.class_id,
    )
    check_outputs(outputs)


def _add_coordinates(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "coordinates",
        "Picked particles: motive list (.em) <-> PointMatrixSet3D.",
    )

    to_cets = directions.add_parser(
        TO_CETS, help="GAPSTOP(TM) motive list (.em) -> CETS yaml."
    )
    to_cets.add_argument(
        "--motl",
        required=True,
        type=Path,
        help="Motive list written by cryocat.tmana.scores_extract_particles (.em).",
    )
    _add_annotation_options(to_cets)
    to_cets.add_argument(
        "--tomo-num",
        type=int,
        help="Numeric tomogram id (tomo_id column) of the particles to convert. Needed "
        "if the motive list has the particles of several tomograms.",
    )
    to_cets.add_argument(
        "--reference-map",
        type=Path,
        help="mrc file on the voxel grid the particles were picked in (tomogram or scores "
        "map): gives the sampling rate if --pixel-size is not given, and checks that the "
        "particles are inside the grid. Default: the scores map recorded in the "
        "extraction script next to the motive list (extractCoords.py), if found.",
    )
    to_cets.add_argument(
        "--no-extraction-script",
        action="store_true",
        help="Do not look for the extraction script to find the reference map.",
    )
    to_cets.add_argument(
        "--score-annotation-id",
        help="Id of the annotation of the scores map the particles were extracted from "
        "(derived_from_annotation_id).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=coordinates_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS, help="CETS yaml -> GAPSTOP(TM) motive list (.em)."
    )
    add_input_yaml(from_cets, "CETS yaml with the coordinates.")
    from_cets.add_argument(
        "-o", "--output", required=True, type=Path, help="Motive list to write (.em)."
    )
    from_cets.add_argument(
        "--tomo-num",
        type=int,
        default=1,
        help="Numeric tomogram id (tomo_id column). Default: 1.",
    )
    from_cets.add_argument(
        "--object-id",
        type=int,
        default=1,
        help="Object id (object_id column). Default: 1.",
    )
    from_cets.add_argument(
        "--class-id", type=int, default=1, help="Class (class column). Default: 1."
    )
    from_cets.set_defaults(func=coordinates_from_cets)


# ------------------------------------------------------------------------- scores
def scores_to_cets(args: argparse.Namespace) -> None:
    scores = GapstopScoreTomogram().gapstop_to_cets(
        scores_map=args.scores_map,
        tomo_id=args.tomo_id,
        pixel_size=args.pixel_size,
        threshold=args.threshold,
        annotated_object=_annotated_object(args.annotated_object),
        annotation_id=args.annotation_id,
    )
    out = write_cets_yaml(args.output, {SCORES_KEY: scores})
    print(f"CETS scores map written -> {out}")


def scores_from_cets(args: argparse.Namespace) -> None:
    scores = load_cets_object(args.input, SCORES_KEY, ProbabilityMap3D)
    out = GapstopScoreTomogram().cets_to_gapstop(
        scores,
        args.output_dir,
        tomo_num=args.tomo_num,
        index=args.index,
        copy=args.copy,
    )
    print(f"GAPSTOP(TM) scores map written -> {out}")


def _add_scores(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "scores",
        "Template matching scores map (scores_<i>_<tomo num>.mrc) <-> ProbabilityMap3D.",
    )

    to_cets = directions.add_parser(
        TO_CETS, help="GAPSTOP(TM) scores map (.mrc) -> CETS yaml."
    )
    to_cets.add_argument(
        "--scores-map",
        required=True,
        type=Path,
        help="Scores map (scores_<i>_<tomo num>.mrc).",
    )
    _add_annotation_options(to_cets)
    to_cets.add_argument(
        "--threshold",
        type=float,
        help="Score above which the voxels were particle candidates in the coordinate "
        "extraction (scores_threshold).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=scores_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS,
        help="CETS yaml -> GAPSTOP(TM) scores map (scores_<index>_<tomo num>.mrc).",
    )
    add_input_yaml(from_cets, "CETS yaml with the scores map.")
    from_cets.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the scores map in.",
    )
    from_cets.add_argument(
        "--tomo-num",
        type=int,
        default=1,
        help="Numeric tomogram id used in the name of the map. Default: 1.",
    )
    from_cets.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the map (first number of its name). Default: 0.",
    )
    from_cets.add_argument(
        "--copy",
        action="store_true",
        help="Copy the map instead of linking it.",
    )
    from_cets.set_defaults(func=scores_from_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-gapstop",
        description="Converts GAPSTOP(TM) files to and from CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type", metavar="{coordinates,scores}", required=True
    )
    _add_coordinates(data_types)
    _add_scores(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
