import os
from pathlib import Path
from typing import Optional

# base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
STATE_FILE: Path = BASE_DIR / ".strata_state.json"

# hugging face token (optional, from environment)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# ensure models directory exists
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except OSError as err:
    # log failure or raise with descriptive error message
    raise OSError(f"failed to create models directory at {MODELS_DIR}: {err}") from err
