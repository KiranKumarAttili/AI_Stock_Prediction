"""Central file locations for the project (Dataset / Saved_Models / Results_and_Plots)."""

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
DATASET_DIR = CODE_DIR.parent / "Dataset"
MODELS_DIR = CODE_DIR / "Saved_Models"
RESULTS_DIR = CODE_DIR / "Results_and_Plots"

DATASET_FILES = {
    "tcs_stock_data.csv",
    "tcs_cleaned_data.csv",
    "tcs_with_indicators.csv",
}


class _ProjectDir:
    """Resolves a file name to the folder that holds it."""

    def __truediv__(self, name):
        name = str(name)
        if name in DATASET_FILES:
            folder = DATASET_DIR
        elif name.endswith((".pkl", ".keras")):
            folder = MODELS_DIR
        else:
            folder = RESULTS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    def __fspath__(self):
        return str(CODE_DIR)

    def __str__(self):
        return str(CODE_DIR)


PROJECT_DIR = _ProjectDir()
