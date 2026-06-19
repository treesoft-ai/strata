from pathlib import Path
from typing import Any, Iterator, Optional
from src.models.base import BaseModelHarness


class TransformersHarness(BaseModelHarness):
    """
    concrete implementation of BaseModelHarness for HF Transformers formats using PyTorch.
    """

    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: str = "cpu"

    def load(self) -> None:
        """
        load tokenizer and weight binaries from the local model directory.
        automatically detects and utilizes CUDA GPU if available.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "transformers and torch are required. please install them to load transformers models."
            ) from err

        try:
            # detect device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # load tokenizer locally
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )

            # load model weights locally
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            # send to model compute hardware device
            self._model.to(self._device)
            self.is_loaded = True
        except Exception as err:
            raise RuntimeError(f"failed to load transformers model from {self.model_path}: {err}") from err

    def unload(self) -> None:
        """
        unload model weight tensors and garbage collect CUDA cache.
        """
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # trigger Python GC to free references
        gc.collect()

        # free CUDA memory if applicable
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self.is_loaded = False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        run query prompt through the tokenizer, execute forward pass on GPU/CPU, and decode response tokens.
        """
        return "".join(self.generate_stream(prompt, **kwargs))

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        run query prompt through the tokenizer, execute forward pass on GPU/CPU, and stream response tokens.
        """
        if not self.is_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("model is not loaded. execute load() before running queries.")

        try:
            import torch
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # extract basic hyperparameters with defaults
            max_tokens: int = kwargs.get("max_tokens", 256)
            temperature: float = kwargs.get("temperature", 0.7)
            do_sample: bool = temperature > 0.0

            # process input prompt tokens
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            # resolve missing padding token IDs (standard on causal model tokenizers like GPT/Llama)
            pad_token_id = self._tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self._tokenizer.eos_token_id

            # set up streamer
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

            # execute forward pass token generation in a separate thread
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )

            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            # yield tokens as they arrive
            for new_text in streamer:
                yield new_text

        except Exception as err:
            raise RuntimeError(f"inference run failed for transformers model: {err}") from err

    @property
    def device(self) -> str:
        return self._device

    def count_tokens(self, text: str) -> int:
        if not self.is_loaded or self._tokenizer is None:
            return 0
        try:
            return len(self._tokenizer.encode(text))
        except Exception:
            return len(text.split())
