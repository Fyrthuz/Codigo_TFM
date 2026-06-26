import yaml
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    num_samples: int = 10
    num_classes: int = 2
    activation: str = "sigmoid"
    batch_size: int = 1


@dataclass
class ModelConfig:
    name: str = "unet"
    in_channels: int = 3
    out_channels: int = 1
    init_features: int = 32
    pretrained: bool = True


@dataclass
class FusionConfig:
    weighting_method: str = "inverse"
    beta: float = 1.0
    alpha: float = 1.0
    threshold_method: str = "naive"
    percentile: int = 50
    k: float = 0.5
    epsilon: float = 1e-6


@dataclass
class CRFConfig:
    sdims: tuple = (5, 5)
    schan: tuple = (5, 5, 5)
    n_iters: int = 5
    epsilon: float = 1e-8


@dataclass
class NoisyConfig:
    noise_std: float = 0.01


@dataclass
class MCDropoutConfig:
    p: float = 0.01


@dataclass
class SubsetConfig:
    subset_size: Optional[int] = None


@dataclass
class CertaintyRefinementConfig:
    enabled: bool = True
    alpha: float = 0.6
    threshold: float = 0.5


@dataclass
class PipelineConfig:
    paths: dict = field(default_factory=lambda: {
        "save_path": "./results",
        "root_path": "./MRI/filtered_data",
    })
    directories: list = field(default_factory=lambda: [
        "mc_dropout", "tta", "noisy", "base", "refined", "original"
    ])
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    crf: CRFConfig = field(default_factory=CRFConfig)
    noisy: NoisyConfig = field(default_factory=NoisyConfig)
    mc_dropout: MCDropoutConfig = field(default_factory=MCDropoutConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    certainty_refinement: CertaintyRefinementConfig = field(default_factory=CertaintyRefinementConfig)
    uncertainty_metric: dict = field(default_factory=lambda: {"threshold": 0.5})

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        inst = cls()
        if data is None:
            return inst
        if "paths" in data:
            inst.paths = data["paths"]
        if "directories" in data:
            inst.directories = data["directories"]
        if "inference" in data:
            for k, v in data["inference"].items():
                setattr(inst.inference, k, v)
        if "model" in data:
            for k, v in data["model"].items():
                setattr(inst.model, k, v)
        if "fusion" in data:
            for k, v in data["fusion"].items():
                setattr(inst.fusion, k, v)
        if "crf" in data:
            for k, v in data["crf"].items():
                if k in ("sdims", "schan"):
                    v = tuple(v)
                setattr(inst.crf, k, v)
        if "noisy" in data:
            for k, v in data["noisy"].items():
                setattr(inst.noisy, k, v)
        if "mc_dropout" in data:
            for k, v in data["mc_dropout"].items():
                setattr(inst.mc_dropout, k, v)
        if "subset" in data:
            for k, v in data["subset"].items():
                setattr(inst.subset, k, v)
        if "certainty_refinement" in data:
            for k, v in data["certainty_refinement"].items():
                setattr(inst.certainty_refinement, k, v)
        if "uncertainty_metric" in data:
            inst.uncertainty_metric = data["uncertainty_metric"]
        return inst
