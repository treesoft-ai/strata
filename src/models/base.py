from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator


class BaseModelHarness(ABC):
    """
    abstract base class for model inference wrappers.
    defines the core interface for loading, unloading, and running queries.
    """

    def __init__(self, model_path: Path) -> None:
        """
        initialize the harness with the path to the model directory or file.
        """
        self.model_path: Path = model_path
        self.is_loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """
        load the model into memory.
        must set self.is_loaded to True when successful.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        unload the model and release memory resources.
        must set self.is_loaded to False when successful.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        execute inference query on the model using the provided prompt.
        returns the generated text string.
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        execute inference query on the model using the provided prompt and stream back text.
        yields text chunks as they become available.
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """
        return the device name the model runs on (e.g. cpu, cuda).
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        count the number of tokens in the given text string.
        """
        pass
