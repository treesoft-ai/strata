from pathlib import Path
from typing import Any, Iterator, Optional
from src.models.base import BaseModelHarness


class GGUFHarness(BaseModelHarness):
    """
    concrete implementation of BaseModelHarness for GGUF formats using llama-cpp-python.
    """

    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self._model: Optional[Any] = None

    def load(self) -> None:
        """
        load the gguf model. search for the gguf file if a folder path is given.
        """
        # resolve target file
        target_file: Path = self.model_path
        if target_file.is_dir():
            gguf_files = list(target_file.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"no gguf file found in directory {self.model_path}")
            target_file = gguf_files[0]

        # dynamic import to allow startup without llama-cpp installed
        try:
            from llama_cpp import Llama
            import llama_cpp
            # offload to GPU if supported
            n_gpu_layers = -1 if getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)() else 0
        except ImportError as err:
            raise ImportError(
                "llama-cpp-python is not installed. please run 'pip install llama-cpp-python' to run gguf models."
            ) from err

        try:
            # load the model using default local context length and gpu layers if supported
            self._model = Llama(
                model_path=str(target_file),
                n_ctx=2048,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            self.is_loaded = True
        except Exception as err:
            raise RuntimeError(f"failed to load gguf model from {target_file}: {err}") from err

    def unload(self) -> None:
        """
        free resources associated with the loaded llama model.
        """
        if self._model is not None:
            # clean up references to trigger garbage collection on C++ allocations
            del self._model
            self._model = None
        self.is_loaded = False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        generate text from the gguf model using the raw query prompt.
        """
        return "".join(self.generate_stream(prompt, **kwargs))

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        generate text from the gguf model using the raw query prompt and stream chunks.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("model is not loaded. execute load() before running queries.")

        try:
            # extract basic parameters with standard defaults
            max_tokens: int = kwargs.get("max_tokens", 256)
            temperature: float = kwargs.get("temperature", 0.7)

            # llama-cpp requires a non-empty prompt; use a single space as fallback
            effective_prompt = prompt if prompt else " "

            response_generator = self._model(
                effective_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in response_generator:
                choices = chunk.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
                    if text:
                        yield text
        except Exception as err:
            raise RuntimeError(f"inference run failed for gguf model: {err}") from err

    @property
    def device(self) -> str:
        if not self.is_loaded or self._model is None:
            return "cpu"
        try:
            n_gpu = getattr(self._model, "n_gpu_layers", 0)
            if n_gpu != 0:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                return "gpu"
        except Exception:
            pass
        return "cpu"

    def count_tokens(self, text: str) -> int:
        if not self.is_loaded or self._model is None:
            return 0
        try:
            return len(self._model.tokenize(text.encode("utf-8")))
        except Exception:
            return len(text.split())
