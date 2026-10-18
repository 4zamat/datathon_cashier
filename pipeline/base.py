from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path

from pydantic import Field

from app.models import Detection, Model, Verdict


class ManualReviewRequired(ValueError):
    """The video was read, but visible evidence does not support a legal verdict."""


class Result(Model):
    violations: list[Detection] = Field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        return "violation" if self.violations else "no_violation"


class Pipeline(ABC):
    @abstractmethod
    def analyze(self, video_path: Path) -> Result:
        """Analyze a local video; return detections without normative references."""
        raise NotImplementedError


def load_pipeline(spec: str) -> Pipeline:
    """Load a no-argument pipeline class from trusted server configuration."""
    module_name, separator, class_name = spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("PIPELINE_CLASS must be module:ClassName")
    pipeline_class = getattr(import_module(module_name), class_name)
    if not isinstance(pipeline_class, type) or not issubclass(pipeline_class, Pipeline):
        raise TypeError("PIPELINE_CLASS must inherit pipeline.base.Pipeline")
    return pipeline_class()
