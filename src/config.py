import os
from pathlib import Path
from typing import Optional

# base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = Path.home() / ".strata" / "models"
LOGS_DIR: Path = Path.home() / ".strata" / "logs"
TRAINED_DIR: Path = Path.home() / ".strata" / "trained"
CONFIGS_DIR: Path = Path.home() / ".strata" / "configs"
DATASETS_DIR: Path = Path.home() / ".strata" / "datasets"

# openrouter key file
OPENROUTER_KEY_FILE: Path = Path.home() / ".strata" / "openrouter.json"

# agentrouter key file
AGENTROUTER_KEY_FILE: Path = Path.home() / ".strata" / "agentrouter.json"

# hugging face token (optional, from environment)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# ensure core directories exist
for _dir in (MODELS_DIR, LOGS_DIR, TRAINED_DIR, CONFIGS_DIR, DATASETS_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise OSError(f"failed to create directory at {_dir}: {err}") from err

# migrate legacy models if any exist
LEGACY_MODELS_DIR: Path = BASE_DIR / "models"
if LEGACY_MODELS_DIR.exists() and LEGACY_MODELS_DIR.is_dir():
    import shutil
    for path in LEGACY_MODELS_DIR.iterdir():
        dest_path = MODELS_DIR / path.name
        if not dest_path.exists():
            try:
                shutil.move(str(path), str(dest_path))
            except Exception as e:
                print(f"failed to migrate {path.name}: {e}")
    # clean up legacy directory if empty
    try:
        if not any(LEGACY_MODELS_DIR.iterdir()):
            LEGACY_MODELS_DIR.rmdir()
    except Exception:
        pass

# clean up legacy state file if it exists in the workspace
LEGACY_STATE_FILE: Path = BASE_DIR / ".strata_state.json"
if LEGACY_STATE_FILE.exists():
    try:
        LEGACY_STATE_FILE.unlink()
    except Exception:
        pass
