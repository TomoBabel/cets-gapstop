import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gapstop.constants import EXTRACTION_FUNCTION, EXTRACTION_SCRIPT_NAME


@dataclass(frozen=True)
class ExtractionParams:
    """Parameters of the coordinate extraction (cryocat.tmana.scores_extract_particles) that
    produced a motive list. Only the ones relevant for CETS are kept."""

    scores_map: Optional[Path]
    """Scores map (GAPSTOP(TM) ScoreTomogram) the particles were extracted from. None if the
    recorded path cannot be found."""
    tomo_num: Optional[int]
    """Numeric tomogram id (tomo_id column of the motive list)."""
    scores_threshold: Optional[float]
    """Score above which the voxels were considered particle candidates."""
    particle_diameter: Optional[float]
    """Diameter (voxels) used to cluster the candidates."""
    symmetry: Optional[str]


def find_extraction_params(
    motl_file: os.PathLike,
) -> Optional[ExtractionParams]:
    """Finds the parameters of the extraction that produced a motive list.

    scipion-em-gapstop writes, next to each motive list, the script it runs to extract the
    coordinates (extractCoords.py), whose call to cryocat.tmana.scores_extract_particles
    records the scores map used, the numeric tomogram id and the threshold. The script is only
    read (parsed), never executed. Its paths are relative to the Scipion project directory, so
    they are resolved against the ancestors of the motive list directory.

    :returns: the extraction parameters, or None if there is no such script next to the
    motive list or it does not contain a call to scores_extract_particles with literal
    arguments.
    """
    motl_path = Path(motl_file)
    script = motl_path.parent / EXTRACTION_SCRIPT_NAME
    if not script.is_file():
        return None
    kwargs = read_extraction_call(script)
    if kwargs is None:
        return None
    scores_map = kwargs.get("scores_map")
    return ExtractionParams(
        scores_map=(
            _resolve(scores_map, motl_path.parent)
            if isinstance(scores_map, str)
            else None
        ),
        tomo_num=_as_int(kwargs.get("tomo_id")),
        scores_threshold=_as_float(kwargs.get("scores_threshold")),
        particle_diameter=_as_float(kwargs.get("particle_diameter")),
        symmetry=(
            kwargs.get("symmetry") if isinstance(kwargs.get("symmetry"), str) else None
        ),
    )


def read_extraction_call(script: os.PathLike) -> Optional[Dict[str, Any]]:
    """Keyword arguments of the scores_extract_particles call of an extraction script, keeping
    only those given as literals (strings, numbers, None). None if the script cannot be parsed
    or has no such call."""
    try:
        tree = ast.parse(Path(script).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == EXTRACTION_FUNCTION:
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                try:
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError:
                    continue  # not a literal
            return kwargs
    return None


def _call_name(func: ast.expr) -> Optional[str]:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _resolve(path: str, start_dir: Path) -> Optional[Path]:
    """An absolute path if it exists, or the first existing one relative to ``start_dir`` or
    any of its ancestors (scipion-em-gapstop writes paths relative to the project)."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for directory in (start_dir, *start_dir.parents):
        if (directory / candidate).is_file():
            return directory / candidate
    return None


def _as_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) else None


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None
