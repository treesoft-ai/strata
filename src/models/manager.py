from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.config import MODELS_DIR
from src.models.base import BaseModelHarness
from src.models.gguf import GGUFHarness
from src.models.transformers import TransformersHarness


class ModelManager:
    """
    coordinates model life cycles: discovering local models, maintaining active state,
    and routing inference calls to the loaded wrapper.
    """

    def __init__(self) -> None:
        self.active_harness: Optional[BaseModelHarness] = None
        self.active_name: Optional[str] = None

    def get_available_models(self) -> List[Dict[str, str]]:
        """
        scan models directory and identify downloaded formats.
        returns list of dictionaries detailing model name and type.
        """
        models: List[Dict[str, str]] = []
        if not MODELS_DIR.exists():
            return models

        for path in MODELS_DIR.iterdir():
            if path.is_dir():
                # check if directory contains a GGUF file
                gguf_files = list(path.glob("*.gguf"))
                if gguf_files:
                    models.append({"name": path.name, "type": "gguf"})
                else:
                    # check for common transformers config/weights files
                    config_file = path / "config.json"
                    if config_file.exists():
                        models.append({"name": path.name, "type": "transformers"})
                    else:
                        # check if it is a general directory with weights (like safetensors)
                        safetensors = list(path.glob("*.safetensors"))
                        bin_files = list(path.glob("*.bin"))
                        if safetensors or bin_files:
                            models.append({"name": path.name, "type": "transformers"})
                        else:
                            models.append({"name": path.name, "type": "unknown"})
        return models



    def load_model(self, name: str) -> BaseModelHarness:
        """
        instantiates and loads the harness for the specified model.
        unloads any previously active harness.
        """
        # unload previous model if active
        self.unload_current_model()

        model_path: Path = MODELS_DIR / name
        if not model_path.exists():
            raise FileNotFoundError(f"model folder does not exist at: {model_path}")

        # detect type
        model_type = "unknown"
        for m in self.get_available_models():
            if m["name"] == name:
                model_type = m["type"]
                break

        if model_type == "gguf":
            harness: BaseModelHarness = GGUFHarness(model_path)
        elif model_type == "transformers":
            harness = TransformersHarness(model_path)
        else:
            raise ValueError(f"unsupported or unrecognized model type for model: {name}")

        try:
            harness.load()
            self.active_harness = harness
            self.active_name = name
            return harness
        except Exception as err:
            raise RuntimeError(f"failed to load model {name}: {err}") from err

    def unload_current_model(self) -> None:
        """
        unload the in-memory model and free system resources.
        """
        if self.active_harness is not None:
            try:
                self.active_harness.unload()
            except Exception:
                # log or ignore failures during unload cleanup to ensure state reset
                pass
            self.active_harness = None
        self.active_name = None

    def delete_model(self, name: str) -> None:
        """
        delete the specified model folder from the local models directory.
        """
        model_path: Path = MODELS_DIR / name
        if not model_path.exists():
            raise FileNotFoundError(f"model '{name}' does not exist.")

        if self.active_name == name:
            self.unload_current_model()

        import shutil
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()

    def rename_model(self, old_name: str, new_name: str) -> None:
        """
        rename the specified model folder in the local models directory.
        """
        old_path: Path = MODELS_DIR / old_name
        new_path: Path = MODELS_DIR / new_name

        if not old_path.exists():
            raise FileNotFoundError(f"model '{old_name}' does not exist.")
        if new_path.exists():
            raise FileExistsError(f"a model named '{new_name}' already exists.")

        if self.active_name == old_name:
            self.unload_current_model()

        old_path.rename(new_path)

    def generate_response(self, prompt: str, model_name: Optional[str] = None, **kwargs: Any) -> str:
        """
        route inference execution to the loaded harness.
        if no harness is loaded, attempts to automatically load the specified model or falls back.
        """
        return "".join(self.generate_response_stream(prompt, model_name=model_name, **kwargs))

    def generate_response_stream(self, prompt: str, model_name: Optional[str] = None, **kwargs: Any) -> Iterator[str]:
        """
        route inference execution to the loaded harness and yield response tokens as they generate.
        """
        if model_name:
            self.load_model(model_name)
        elif self.active_harness is None:
            # find the first available model
            models = self.get_available_models()
            if not models:
                raise RuntimeError("no models have been downloaded yet. use download command first.")
            active_name = models[0]["name"]
            self.load_model(active_name)

        if self.active_harness is None:
            raise RuntimeError("no active model is loaded.")

        yield from self.active_harness.generate_stream(prompt, **kwargs)

    def count_tokens(self, text: str) -> int:
        """
        count the number of tokens in the given text string.
        """
        if self.active_harness is not None:
            return self.active_harness.count_tokens(text)
        return 0
