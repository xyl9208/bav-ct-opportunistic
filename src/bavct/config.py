"""Path configuration for the released research code.

All paths can be overridden with environment variables so that the code can run
against a local data layout without editing the source files.

Environment variables
---------------------
BAVCT_WORK              working directory (manifests, runs, results)
BAVCT_DATA              root of the cohort NIfTI tree
BAVCT_MASKS             TotalSegmentator output for the test cohorts
BAVCT_MISS_DX           delayed-diagnosis cohort root
BAVCT_MEDICALNET_DIR    local clone of the MedicalNet repository (optional)
BAVCT_MEDICALNET_WEIGHTS  MedicalNet ResNet-18 checkpoint (optional)

Default layout (relative to the repository root)::

    work/
      data/        manifests, processed arrays, cohort roster
      runs/        training runs
      results/     metrics, thresholds, pre-processing QC
    data/
      internal/    BAV/  TAV/
      external/    anji/ BAV/ TAV/ calib/
                   quzhou/ BAV/ TAV/ calib/
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(env, default):
    return Path(os.environ.get(env, default)).expanduser().resolve()


WORK = _path("BAVCT_WORK", REPO_ROOT / "work")
DATA = _path("BAVCT_DATA", REPO_ROOT / "data")

SRC_DIR = REPO_ROOT / "src"
RESULTS = WORK / "results"
RUNS_DIR = WORK / "runs"

MANIFEST_CSV = WORK / "data" / "manifest.csv"
COHORT_ROSTER = WORK / "data" / "cohorts.csv"
PREPROCESS_MANIFEST = WORK / "data" / "preprocess_manifest.csv"

MASKS_FINE = _path("BAVCT_MASKS", WORK / "masks_fine")
MISS_DX = _path("BAVCT_MISS_DX", WORK / "missdx")

INTERNAL = DATA / "internal"
EXTERNAL = DATA / "external"
METADATA_DIR = DATA / "metadata"

COHORTS = {
    "internal_BAV": INTERNAL / "BAV",
    "internal_TAV": INTERNAL / "TAV",
    "anji_BAV": EXTERNAL / "anji" / "BAV",
    "anji_TAV": EXTERNAL / "anji" / "TAV",
    "anji_calib": EXTERNAL / "anji" / "calib",
    "quzhou_BAV": EXTERNAL / "quzhou" / "BAV",
    "quzhou_TAV": EXTERNAL / "quzhou" / "TAV",
    "quzhou_calib": EXTERNAL / "quzhou" / "calib",
}

INPUT_SHAPE = (128, 128, 128)
N_CLASSES = 2

# Optional MedicalNet backbone (comparison arm only; the deployed model is the
# native 3D ResNet-18 in bavct.models.resnet_native).
MEDICALNET_DIR = _path(
    "BAVCT_MEDICALNET_DIR", REPO_ROOT / "third_party" / "MedicalNet"
)
MEDICALNET_WEIGHTS = _path(
    "BAVCT_MEDICALNET_WEIGHTS", REPO_ROOT / "weights" / "resnet_18_23dataset.pth"
)
