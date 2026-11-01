"""Backend version and user-overridable Scipion configuration variables."""

BACKEND_VERSION = "1.0.0"
# Verified commit of the official v1.0.0 release; never follow main.
SOURCE_COMMIT = "a35e73a72f9cbc2b13cf6f8d95cf963cb704ca17"
SOURCE_TAG = "v" + BACKEND_VERSION
SOURCE_URL = "https://github.com/YangLab-SDU/CryoCodex/archive/refs/tags/%s.tar.gz" % SOURCE_TAG
SOURCE_BUILD_DIR = "CryoCodex-" + BACKEND_VERSION
MODEL_URL = "https://yanglab.qd.sdu.edu.cn/CryoCodex/CryoCodex.pth"
CRYOCODEX_HOME = "CRYOCODEX_HOME"
CRYOCODEX_PYTHON = "CRYOCODEX_PYTHON"
CRYOCODEX_MODEL = "CRYOCODEX_MODEL"
CRYOCODEX_CONDA_EXE = "CRYOCODEX_CONDA_EXE"
INSTALL_MARKER = "cryocodex-installed.json"
CHIMERAX_COLOR_COMMAND = (
    'color sample #1 map #2 palette '
    '"#1B3A5F:#245A8D:#2F80C0:#6BAED6:#BFD9EA:#7F7F7F:#C9B458:'
    '#FEE191:#F9B233:#FC8E59:#F04438:#DC3223:#8B0000" key true ; '
    'key fontSize 11 size 0.65,0.02 pos 0.18,0.06 colorTreatment distinct'
)
OUTPUT_FILES = {
    "enhanced_volume": "cryocodex.mrc",
    "input_volume_local_quality": "cryocodex_in_score.mrc",
    "enhanced_volume_local_quality": "cryocodex_out_score.mrc",
    "molecular_mask": "cryocodex_in_mask.mrc",
}
