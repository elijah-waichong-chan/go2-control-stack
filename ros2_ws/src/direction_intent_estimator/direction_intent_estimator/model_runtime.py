"""Shared runtime utilities for direction intent ONNX models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - runtime check
    np = None

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - runtime check
    ort = None

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - runtime check
    yaml = None


def require_runtime_dependencies() -> None:
    """Fail early with a clear message when runtime deps are missing."""
    if np is None:
        raise RuntimeError(
            "numpy is required for direction intent inference. "
            "Install it before running this node."
        )
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for direction intent inference. "
            "Install it before running this node."
        )
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required for direction intent inference. "
            "Install with: pip install onnxruntime"
        )


def as_int_mapping(raw: dict[Any, Any] | None) -> dict[int, int]:
    """Convert YAML label mappings to plain int-to-int dictionaries."""
    out: dict[int, int] = {}
    for key, value in (raw or {}).items():
        out[int(key)] = int(value)
    return out


def load_deploy_cfg(path: Path) -> dict[str, Any]:
    """Load and validate a model deploy YAML."""
    if not path.exists():
        raise RuntimeError(f"deploy.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("deploy.yaml must be a YAML mapping.")
    return cfg


def first_existing_path(paths: Iterable[Path]) -> Path:
    """Return the first existing path while preserving the declared fallback order."""
    ordered_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered_paths.append(path)
        if path.exists():
            return path
    if not ordered_paths:
        raise ValueError("Expected at least one candidate path.")
    return ordered_paths[0]


def resolve_deploy_path(model_dir: Path) -> Path:
    """Resolve the current or legacy deploy YAML path inside a model bundle."""
    params_dir = model_dir / "params"
    primary = params_dir / "deploy.yaml"
    if primary.exists():
        return primary

    legacy_candidates = sorted(params_dir.glob("*.deploy.yaml"))
    if legacy_candidates:
        return legacy_candidates[0]

    raise RuntimeError(
        f"deploy.yaml not found: {primary}. Checked legacy '*.deploy.yaml' files in {params_dir}."
    )


def resolve_model_path(deploy_path: Path, model_cfg: dict[str, Any]) -> Path:
    """Resolve the ONNX path for current and legacy bundle layouts."""
    raw_path = model_cfg.get("path")
    if not raw_path:
        raise RuntimeError("deploy.yaml model.path is required.")
    configured_path = Path(str(raw_path))
    candidates: list[Path] = []
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        candidates.append((deploy_path.parent / configured_path).resolve())
        if deploy_path.parent.name == "params":
            candidates.append((deploy_path.parent.parent / "exported" / configured_path.name).resolve())

    model_path = first_existing_path(candidates)
    if not model_path.exists():
        checked_paths = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"ONNX model not found. Checked: {checked_paths}")
    return model_path


@dataclass(frozen=True)
class ModelMetadata:
    """Model metadata loaded from a deploy bundle."""

    model_dir: Path
    model_path: Path
    input_name: str
    output_name: str
    num_features: int
    num_timesteps: int
    normalize: bool
    x_mean: Any
    x_std: Any
    index_to_label: dict[int, int]

    @property
    def output_dim(self) -> int | str:
        """Return the expected output width when labels are declared."""
        return len(self.index_to_label) or "unknown"


def load_model_metadata(model_dir: Path) -> ModelMetadata:
    """Load model metadata from a per-model bundle directory."""
    deploy_path = resolve_deploy_path(model_dir)
    deploy_cfg = load_deploy_cfg(deploy_path)
    model_cfg = deploy_cfg.get("model", {})
    preprocessing_cfg = deploy_cfg.get("preprocessing", {})
    labels_cfg = deploy_cfg.get("labels", {})

    num_features = int(model_cfg.get("num_features", 0))
    num_timesteps = int(model_cfg.get("num_timesteps", 0))
    x_mean = np.asarray(
        preprocessing_cfg.get("x_mean", [0.0] * num_features), dtype=np.float32
    )
    x_std = np.asarray(
        preprocessing_cfg.get("x_std", [1.0] * num_features), dtype=np.float32
    )
    x_std = np.where(x_std == 0.0, 1.0, x_std)

    if x_mean.shape != (num_features,) or x_std.shape != (num_features,):
        raise RuntimeError(
            "preprocessing x_mean/x_std size must match model.num_features in deploy.yaml"
        )

    return ModelMetadata(
        model_dir=model_dir,
        model_path=resolve_model_path(deploy_path, model_cfg),
        input_name=str(model_cfg.get("input_name", "input")),
        output_name=str(model_cfg.get("output_name", "logits")),
        num_features=num_features,
        num_timesteps=num_timesteps,
        normalize=bool(preprocessing_cfg.get("normalize", False)),
        x_mean=x_mean,
        x_std=x_std,
        index_to_label=as_int_mapping(labels_cfg.get("index_to_label")),
    )


class SlidingWindowIntentModel:
    """Shared ONNX-backed, time-resampled sliding-window classifier."""

    def __init__(
        self,
        *,
        logger: Any,
        model_dir: Path,
        onnx_intra_threads: int = 1,
        onnx_inter_threads: int = 1,
        sliding_window_ms: float = 300.0,
        sampling_hz: float = 200.0,
        publish_hz: float = 10.0,
    ) -> None:
        require_runtime_dependencies()
        self.logger = logger
        self.metadata = load_model_metadata(model_dir)
        self.sliding_window_ms = max(1.0, float(sliding_window_ms))
        self.sampling_hz = max(1.0, float(sampling_hz))
        self.publish_hz = max(1.0, float(publish_hz))
        self.sample_period_s = 1.0 / self.sampling_hz
        self.publish_period_s = 1.0 / self.publish_hz
        self.window_samples = int(round(self.sliding_window_ms * 1e-3 * self.sampling_hz))
        if self.window_samples != self.metadata.num_timesteps:
            raise RuntimeError(
                "Configured sliding window requires %d timesteps, but model expects %d."
                % (self.window_samples, self.metadata.num_timesteps)
            )
        self.window_span_s = max(0.0, (self.window_samples - 1) * self.sample_period_s)
        self.max_history_age_s = self.window_span_s + self.publish_period_s + self.sample_period_s
        self.history: deque[tuple[float, Any]] = deque()
        self.last_inference_time_s: float | None = None
        self.input_name = self.metadata.input_name
        self.output_name = self.metadata.output_name
        self.session = self._create_onnx_session(
            self.metadata.model_path,
            onnx_intra_threads,
            onnx_inter_threads,
        )

    def _create_onnx_session(
        self,
        model_path: Path,
        onnx_intra_threads: int,
        onnx_inter_threads: int,
    ):
        """Create an ONNX Runtime session and reconcile tensor names."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(onnx_intra_threads))
        opts.inter_op_num_threads = max(1, int(onnx_inter_threads))
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("ONNX model has no inputs or outputs.")
        if inputs[0].name != self.input_name:
            self.logger.warn(
                "deploy.yaml input_name=%s does not match ONNX input=%s; "
                "using ONNX name."
                % (self.input_name, inputs[0].name)
            )
            self.input_name = inputs[0].name
        if outputs[0].name != self.output_name:
            self.logger.warn(
                "deploy.yaml output_name=%s does not match ONNX output=%s; "
                "using ONNX name."
                % (self.output_name, outputs[0].name)
            )
            self.output_name = outputs[0].name
        return session

    def _prune_history(self, now_s: float) -> None:
        cutoff = now_s - self.max_history_age_s
        while len(self.history) > 1 and self.history[1][0] < cutoff:
            self.history.popleft()

    def _has_full_window(self, now_s: float) -> bool:
        if not self.history:
            return False
        oldest_needed_s = now_s - self.window_span_s
        return self.history[0][0] <= oldest_needed_s

    def _build_window(self, now_s: float) -> np.ndarray | None:
        if not self._has_full_window(now_s):
            return None
        entries = list(self.history)
        oldest_needed_s = now_s - self.window_span_s
        if entries[0][0] > oldest_needed_s:
            return None

        window = np.empty((self.window_samples, self.metadata.num_features), dtype=np.float32)
        entry_idx = 0
        current = entries[0][1]
        for i in range(self.window_samples):
            target_time_s = oldest_needed_s + i * self.sample_period_s
            while entry_idx + 1 < len(entries) and entries[entry_idx + 1][0] <= target_time_s:
                entry_idx += 1
                current = entries[entry_idx][1]
            window[i] = current
        return window

    def push(self, features: Any, sample_time_s: float) -> int | None:
        """Append one feature vector and return a throttled predicted label when ready."""
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self.metadata.num_features,):
            raise ValueError(
                "Expected %d features, got shape %s."
                % (self.metadata.num_features, vector.shape)
            )
        if not np.isfinite(sample_time_s):
            raise ValueError(f"Expected finite sample_time_s, got {sample_time_s!r}.")
        if self.history and sample_time_s < self.history[-1][0]:
            sample_time_s = self.history[-1][0]

        self.history.append((sample_time_s, vector.copy()))
        self._prune_history(sample_time_s)
        if not self._has_full_window(sample_time_s):
            return None
        if self.last_inference_time_s is not None:
            if (sample_time_s - self.last_inference_time_s) < self.publish_period_s:
                return None

        window = self._build_window(sample_time_s)
        if window is None:
            return None
        if self.metadata.normalize:
            window = (window - self.metadata.x_mean) / self.metadata.x_std

        model_input = np.transpose(window, (1, 0))[None, :, :]
        logits = self.session.run([self.output_name], {self.input_name: model_input})[0]
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[0]
        logits = logits.reshape(-1)
        pred_index = int(np.argmax(logits))
        self.last_inference_time_s = sample_time_s
        return self.metadata.index_to_label.get(pred_index, pred_index)
