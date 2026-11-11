"""Converts the GAPSTOP(TM) picked coordinates (motive lists in EM format) found in a directory
into CETS metadata, together with the scores maps (ScoreTomograms) they were extracted from,
when they can be found."""

import argparse
from pathlib import Path
from typing import Dict, Optional

from gapstop.constants import EM_EXT
from gapstop.converters.coordinates3d import GapstopCoordinates3D
from gapstop.converters.score_tomogram import GapstopScoreTomogram
from gapstop.utils.extraction import find_extraction_params
from gapstop.utils.utils import (
    read_em_motl,
    write_coordinates_yaml,
    write_scores_yaml,
)

### GAPSTOP TO CETS ################################################################
scratch_dir = Path("/home/jjimenez/CZII/cets_scratch_dir")
# Any directory containing motive lists written by cryocat.tmana.scores_extract_particles
# (output_type="emmotl"), searched recursively. Other EM files (e.g. templates) are skipped.
gapstop_dir = Path(
    "/home/jjimenez/ScipionUserData/projects/czii_re5_extract_subtomos/Runs/003282_ProtGapStopExtractCoords/"
)
# CETS tomogram id of each numeric tomogram id (tomo_id column) of the motive lists. The
# motive lists do not record it: when a number is not in this dict, the name of the directory
# containing the motive list is used (one motive list per tomogram directory).
tomo_ids: Dict[int, str] = {}


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return number


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--sampling-rate",
    type=positive_float,
    default=None,
    help="Sampling rate (Å/px) of the tomograms the particles were picked in. If not given, "
    "it is read from the header of the scores map, if found; otherwise, only the array "
    "(voxel) frame is attached.",
)
sampling_rate: Optional[float] = parser.parse_args().sampling_rate

converter = GapstopCoordinates3D()
scores_converter = GapstopScoreTomogram()
n_converted = 0
for em_file in sorted(gapstop_dir.rglob(f"*{EM_EXT}")):
    try:
        read_em_motl(em_file)
    except ValueError:
        continue  # not a motive list
    # Scores map (ScoreTomogram) the particles were extracted from: recorded by
    # scipion-em-gapstop in the extraction script next to the motive list (optional).
    params = find_extraction_params(em_file)
    scores_map = params.scores_map if params else None
    if scores_map is None:
        print(
            f"NOTE: no scores map found for {em_file}; converting the coordinates alone."
        )
    for tomo_num in converter.get_tomo_nums(em_file):
        tomo_id = tomo_ids.get(tomo_num, em_file.parent.name)
        score_annotation_id = None
        reference_map = None
        if params and scores_map and params.tomo_num in (None, tomo_num):
            # ScoreTomogram -> ProbabilityMap3D (Region.annotations)
            scores = scores_converter.gapstop_to_cets(
                scores_map=scores_map,
                tomo_id=tomo_id,
                pixel_size=sampling_rate,
                threshold=params.scores_threshold,
            )
            write_scores_yaml(scores, scratch_dir)
            score_annotation_id = scores.id
            # The particles are on its voxel grid: sampling rate and grid check
            reference_map = scores_map
        # Picked coordinates -> PointMatrixSet3D (Region.annotations), derived from the scores
        coordinates = converter.gapstop_to_cets(
            motl_file=em_file,
            tomo_id=tomo_id,
            tomo_num=tomo_num,
            pixel_size=sampling_rate,
            reference_map=reference_map,
            score_annotation_id=score_annotation_id,
        )
        if coordinates is not None:
            write_coordinates_yaml(coordinates, scratch_dir)
            n_converted += 1

if not n_converted:
    raise SystemExit(f"No GAPSTOP(TM) motive lists were found in {gapstop_dir}")
