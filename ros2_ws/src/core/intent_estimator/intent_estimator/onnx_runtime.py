"""Shared ONNX runtime utilities for direction intent models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def default_onnx_intra_threads() -> int:
    """Choose a conservative default for CPU inference threading."""
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def as_int_mapping(raw: dict[Any, Any] | None) -> dict[int, int]:
    """Convert YAML label mappings to plain int-to-int dictionaries."""
    out: dict[int, int] = {}
    for key, value in (raw or {}).items():
        out[int(key)] = int(value)
    return out


def load_deploy_cfg(path: Path) -> dict[str, Any]:
    """Load and validate a model deploy YAML."""
    if not path.exists():
        parent = path.parent
        if parent.exists():
            sibling_candidates = sorted(parent.glob("*.deploy.yaml"))
            if len(sibling_candidates) == 1:
                path = sibling_candidates[0]
            elif len(sibling_candidates) > 1:
                raise RuntimeError(
                    "deploy.yaml not found at %s, and multiple '*.deploy.yaml' siblings exist: %s"
                    % (path, ", ".join(candidate.name for candidate in sibling_candidates))
                )
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
    """Resolve the deploy YAML inside a flat model bundle directory."""
    flat_candidates = sorted(model_dir.glob("*.deploy.yaml"))
    if len(flat_candidates) == 1:
        return flat_candidates[0]
    if len(flat_candidates) > 1:
        valid_candidates: list[Path] = []
        invalid_candidates: list[str] = []
        for candidate in flat_candidates:
            try:
                deploy_cfg = load_deploy_cfg(candidate)
                model_cfg = deploy_cfg.get("model", {})
                model_path = resolve_model_path(candidate, model_cfg)
            except Exception as exc:
                invalid_candidates.append(f"{candidate.name} ({exc})")
                continue
            if model_path.exists():
                valid_candidates.append(candidate)
            else:
                invalid_candidates.append(
                    f"{candidate.name} (missing model: {model_path.name})"
                )

        if len(valid_candidates) == 1:
            return valid_candidates[0]

        details = ", ".join(str(path.name) for path in flat_candidates)
        if invalid_candidates:
            details += "; invalid candidates: %s" % ", ".join(invalid_candidates)
        raise RuntimeError(
            "Expected exactly one usable '*.deploy.yaml' in %s, found: %s"
            % (model_dir, details)
        )

    raise RuntimeError(
        "deploy.yaml not found in flat bundle %s. Expected exactly one '*.deploy.yaml' file."
        % model_dir
    )


def resolve_model_path(deploy_path: Path, model_cfg: dict[str, Any]) -> Path:
    """Resolve the ONNX path declared relative to a flat bundle deploy YAML."""
    raw_path = model_cfg.get("path")
    if not raw_path:
        raise RuntimeError("deploy.yaml model.path is required.")
    configured_path = Path(str(raw_path))
    candidates: list[Path] = []
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        candidates.append((deploy_path.parent / configured_path).resolve())

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
    raw_num_features: int
    delta_features: bool
    delta_feature_mode: str
    delta_formula: str
    x_mean: Any
    x_std: Any
    index_to_label: dict[int, int]
    output_type: str
    postprocess: str
    prediction_rule: str
    nonzero_prediction_threshold: float | None
    zero_index: int
    nonzero_indices: tuple[int, ...]
    label_confidence_thresholds: dict[int, float]
    fallback_label: int | None

    @property
    def input_num_features(self) -> int:
        """Return the feature width expected before deploy-time preprocessing."""
        return self.raw_num_features if self.delta_features else self.num_features

    @property
    def output_dim(self) -> int | str:
        """Return the expected output width when labels are declared."""
        return len(self.index_to_label) or "unknown"


@dataclass(frozen=True)
class SelectedFeature:
    """One named feature block declared in a deploy YAML."""

    name: str
    width: int
    source_start: int
    source_end: int
    source_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PredictionResult:
    """One throttled model inference result."""

    logits: Any
    prediction_scores: Any
    pred_index: int
    pred_label: int


def load_selected_features(deploy_cfg: dict[str, Any]) -> tuple[SelectedFeature, ...]:
    """Load ordered named feature blocks from a deploy YAML."""
    raw_selected = deploy_cfg.get("features", {}).get("selected", [])
    if raw_selected is None:
        return ()
    if not isinstance(raw_selected, list):
        raise RuntimeError("deploy.yaml features.selected must be a list when provided.")

    selected: list[SelectedFeature] = []
    next_default_start = 0
    for index, raw_feature in enumerate(raw_selected):
        if not isinstance(raw_feature, dict):
            raise RuntimeError(
                "deploy.yaml features.selected[%d] must be a mapping." % index
            )
        name = str(raw_feature.get("name", "")).strip()
        width = int(raw_feature.get("width", 0))
        raw_source_start = raw_feature.get("source_start")
        raw_source_end = raw_feature.get("source_end")
        raw_source_indices = raw_feature.get("source_indices")
        if not name:
            raise RuntimeError(
                "deploy.yaml features.selected[%d].name is required." % index
            )
        if width <= 0:
            raise RuntimeError(
                "deploy.yaml features.selected[%d].width must be > 0." % index
            )
        source_indices: tuple[int, ...] | None = None
        if raw_source_indices is not None:
            if not isinstance(raw_source_indices, list):
                raise RuntimeError(
                    "deploy.yaml features.selected[%d].source_indices must be a list."
                    % index
                )
            source_indices = tuple(int(source_index) for source_index in raw_source_indices)
            if len(source_indices) != width:
                raise RuntimeError(
                    "deploy.yaml features.selected[%d] width=%d does not match "
                    "len(source_indices)=%d."
                    % (index, width, len(source_indices))
                )
            if any(source_index < 0 for source_index in source_indices):
                raise RuntimeError(
                    "deploy.yaml features.selected[%d].source_indices entries must be >= 0."
                    % index
                )
            if len(set(source_indices)) != len(source_indices):
                raise RuntimeError(
                    "deploy.yaml features.selected[%d].source_indices must not contain duplicates."
                    % index
                )

        if raw_source_start is None and raw_source_end is None:
            source_start = next_default_start
            if source_indices is None:
                source_end = source_start + width
            else:
                source_start = min(source_indices)
                source_end = max(source_indices) + 1
        elif raw_source_start is None or raw_source_end is None:
            raise RuntimeError(
                "deploy.yaml features.selected[%d] must declare both source_start "
                "and source_end when either is provided." % index
            )
        else:
            source_start = int(raw_source_start)
            source_end = int(raw_source_end)

        if source_start < 0:
            raise RuntimeError(
                "deploy.yaml features.selected[%d].source_start must be >= 0." % index
            )
        if source_end <= source_start:
            raise RuntimeError(
                "deploy.yaml features.selected[%d].source_end must be > source_start."
                % index
            )
        if source_indices is None and (source_end - source_start) != width:
            raise RuntimeError(
                "deploy.yaml features.selected[%d] width=%d does not match "
                "source_end-source_start=%d."
                % (index, width, source_end - source_start)
            )
        if source_indices is not None:
            if any(
                source_index < source_start or source_index >= source_end
                for source_index in source_indices
            ):
                raise RuntimeError(
                    "deploy.yaml features.selected[%d].source_indices must lie within "
                    "[source_start, source_end)." % index
                )

        selected.append(
            SelectedFeature(
                name=name,
                width=width,
                source_start=source_start,
                source_end=source_end,
                source_indices=source_indices,
            )
        )
        next_default_start = source_end
    return tuple(selected)


def build_observation_vector(
    selected_features: Sequence[SelectedFeature],
    source_vectors: dict[str, Any],
):
    """Place named feature blocks into the deploy-declared observation slots."""
    if np is None:
        raise RuntimeError(
            "numpy is required for direction intent inference. "
            "Install it before running this node."
        )

    if not selected_features:
        return np.empty((0,), dtype=np.float32)

    observation_width = max(
        (
            max(feature.source_indices) + 1
            if feature.source_indices is not None
            else feature.source_end
        )
        for feature in selected_features
    )
    observation = np.zeros((observation_width,), dtype=np.float32)
    occupied = np.zeros((observation_width,), dtype=bool)

    for feature in selected_features:
        if feature.name not in source_vectors:
            raise ValueError(
                "Missing source vector for deploy feature %r." % feature.name
            )
        vector = np.asarray(source_vectors[feature.name], dtype=np.float32).reshape(-1)
        if vector.shape != (feature.width,):
            raise ValueError(
                "Expected source vector %r to have width %d, got shape %s."
                % (feature.name, feature.width, vector.shape)
            )

        if feature.source_indices is not None:
            indices = np.asarray(feature.source_indices, dtype=int)
            if occupied[indices].any():
                raise ValueError(
                    "Overlapping deploy feature indices for %r at %s."
                    % (feature.name, list(feature.source_indices))
                )
            observation[indices] = vector
            occupied[indices] = True
        else:
            start = feature.source_start
            end = feature.source_end
            if occupied[start:end].any():
                raise ValueError(
                    "Overlapping deploy feature slice for %r at [%d:%d]."
                    % (feature.name, start, end)
                )
            observation[start:end] = vector
            occupied[start:end] = True

    return observation


def build_selected_feature_vector(
    selected_features: Sequence[SelectedFeature],
    source_vectors: dict[str, Any],
):
    """Select deploy-configured observation slices in the declared feature order."""
    observation = build_observation_vector(selected_features, source_vectors)
    if observation.size == 0:
        return observation

    parts = [
        (
            observation[np.asarray(feature.source_indices, dtype=int)]
            if feature.source_indices is not None
            else observation[feature.source_start:feature.source_end]
        )
        for feature in selected_features
    ]
    return np.concatenate(parts, axis=0)


def load_model_metadata(model_dir: Path) -> ModelMetadata:
    """Load model metadata from a per-model bundle directory."""
    deploy_path = resolve_deploy_path(model_dir)
    deploy_cfg = load_deploy_cfg(deploy_path)
    model_cfg = deploy_cfg.get("model", {})
    preprocessing_cfg = deploy_cfg.get("preprocessing", {})
    labels_cfg = deploy_cfg.get("labels", {})
    inference_cfg = deploy_cfg.get("inference", {})

    num_features = int(model_cfg.get("num_features", 0))
    num_timesteps = int(model_cfg.get("num_timesteps", 0))
    delta_features = bool(preprocessing_cfg.get("delta_features", False))
    raw_num_features = int(preprocessing_cfg.get("num_raw_features", num_features))
    delta_feature_mode = str(preprocessing_cfg.get("delta_feature_mode", "")).strip()
    delta_formula = str(preprocessing_cfg.get("delta_formula", "")).strip()
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
    if raw_num_features <= 0:
        raise RuntimeError("preprocessing num_raw_features must be > 0 when provided.")
    if delta_features:
        if delta_feature_mode != "append":
            raise RuntimeError(
                "Unsupported preprocessing delta_feature_mode=%r. "
                "Only 'append' is currently supported." % delta_feature_mode
            )
        if delta_formula != "window_start_subtraction":
            raise RuntimeError(
                "Unsupported preprocessing delta_formula=%r. "
                "Only 'window_start_subtraction' is currently supported." % delta_formula
            )
        if num_features != 2 * raw_num_features:
            raise RuntimeError(
                "With delta_features append mode, model.num_features (%d) must equal "
                "2 * preprocessing.num_raw_features (%d)."
                % (num_features, raw_num_features)
            )

    prediction_rule = str(inference_cfg.get("prediction_rule", "argmax")).strip() or "argmax"
    output_type = str(inference_cfg.get("output_type", "logits")).strip() or "logits"
    postprocess = str(inference_cfg.get("postprocess", "")).strip()
    nonzero_prediction_threshold = inference_cfg.get("nonzero_prediction_threshold")
    if nonzero_prediction_threshold is not None:
        nonzero_prediction_threshold = float(nonzero_prediction_threshold)
    zero_index = int(inference_cfg.get("zero_index", 0))
    raw_nonzero_indices = inference_cfg.get("nonzero_indices", [])
    if raw_nonzero_indices is None:
        raw_nonzero_indices = []
    if not isinstance(raw_nonzero_indices, list):
        raise RuntimeError("inference.nonzero_indices must be a list when provided.")
    nonzero_indices = tuple(int(index) for index in raw_nonzero_indices)
    raw_label_confidence_thresholds = inference_cfg.get("label_confidence_thresholds", {})
    if raw_label_confidence_thresholds is None:
        raw_label_confidence_thresholds = {}
    if not isinstance(raw_label_confidence_thresholds, dict):
        raise RuntimeError(
            "inference.label_confidence_thresholds must be a mapping when provided."
        )
    label_confidence_thresholds = {
        int(label): float(threshold)
        for label, threshold in raw_label_confidence_thresholds.items()
    }
    fallback_label = inference_cfg.get("fallback_label")
    if fallback_label is not None:
        fallback_label = int(fallback_label)

    supported_prediction_rules = {
        "argmax",
        "nonzero_threshold",
        "label_confidence_thresholds",
    }
    if prediction_rule not in supported_prediction_rules:
        raise RuntimeError(
            "Unsupported inference.prediction_rule=%r. Supported values: %s."
            % (prediction_rule, ", ".join(sorted(supported_prediction_rules)))
        )

    if prediction_rule == "label_confidence_thresholds" and not label_confidence_thresholds:
        raise RuntimeError(
            "inference.prediction_rule=label_confidence_thresholds requires "
            "inference.label_confidence_thresholds."
        )

    return ModelMetadata(
        model_dir=model_dir,
        model_path=resolve_model_path(deploy_path, model_cfg),
        input_name=str(model_cfg.get("input_name", "input")),
        output_name=str(model_cfg.get("output_name", "logits")),
        num_features=num_features,
        num_timesteps=num_timesteps,
        normalize=bool(preprocessing_cfg.get("normalize", False)),
        raw_num_features=raw_num_features,
        delta_features=delta_features,
        delta_feature_mode=delta_feature_mode,
        delta_formula=delta_formula,
        x_mean=x_mean,
        x_std=x_std,
        index_to_label=as_int_mapping(labels_cfg.get("index_to_label")),
        output_type=output_type,
        postprocess=postprocess,
        prediction_rule=prediction_rule,
        nonzero_prediction_threshold=nonzero_prediction_threshold,
        zero_index=zero_index,
        nonzero_indices=nonzero_indices,
        label_confidence_thresholds=label_confidence_thresholds,
        fallback_label=fallback_label,
    )


class SlidingWindowIntentModel:
    """Shared ONNX-backed, time-resampled sliding-window classifier."""

    def __init__(
        self,
        *,
        logger: Any,
        model_dir: Path,
        onnx_intra_threads: int = default_onnx_intra_threads(),
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

        window = np.empty(
            (self.window_samples, self.metadata.input_num_features), dtype=np.float32
        )
        entry_idx = 0
        current = entries[0][1]
        for i in range(self.window_samples):
            target_time_s = oldest_needed_s + i * self.sample_period_s
            while entry_idx + 1 < len(entries) and entries[entry_idx + 1][0] <= target_time_s:
                entry_idx += 1
                current = entries[entry_idx][1]
            window[i] = current
        return window

    def _prepare_window_for_model(self, window: np.ndarray) -> np.ndarray:
        if not self.metadata.delta_features:
            return window

        if self.metadata.delta_feature_mode != "append":
            raise RuntimeError(
                "Unsupported preprocessing delta_feature_mode=%r."
                % self.metadata.delta_feature_mode
            )
        if self.metadata.delta_formula != "window_start_subtraction":
            raise RuntimeError(
                "Unsupported preprocessing delta_formula=%r."
                % self.metadata.delta_formula
            )

        baseline = window[0:1, :]
        delta = window - baseline
        prepared = np.concatenate((window, delta), axis=1)
        if prepared.shape != (self.window_samples, self.metadata.num_features):
            raise RuntimeError(
                "Prepared feature window has shape %s, expected (%d, %d)."
                % (
                    prepared.shape,
                    self.window_samples,
                    self.metadata.num_features,
                )
            )
        return prepared

    def _postprocess_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert model outputs into comparable prediction scores."""
        postprocess = self.metadata.postprocess.lower()
        if postprocess == "softmax":
            shifted = scores - np.max(scores)
            exp_scores = np.exp(shifted)
            denom = float(np.sum(exp_scores))
            if not np.isfinite(denom) or denom <= 0.0:
                raise RuntimeError("Softmax postprocess produced an invalid denominator.")
            return exp_scores / denom
        return scores

    def _select_prediction_index(self, scores: np.ndarray) -> int:
        """Apply the deploy-configured prediction rule to model outputs."""
        prediction_scores = self._postprocess_scores(scores)
        return self.select_prediction_index_from_scores(prediction_scores)

    def select_prediction_index_from_scores(self, prediction_scores: np.ndarray) -> int:
        """Apply the deploy-configured prediction rule to comparable prediction scores."""
        prediction_scores = np.asarray(prediction_scores, dtype=np.float32).reshape(-1)
        if self.metadata.prediction_rule == "label_confidence_thresholds":
            return self._select_prediction_index_with_label_thresholds(prediction_scores)
        if self.metadata.prediction_rule == "nonzero_threshold":
            return self._select_prediction_index_with_nonzero_threshold(prediction_scores)
        return int(np.argmax(prediction_scores))

    def _label_to_index(self, raw_label: int) -> int | None:
        for index, candidate_label in self.metadata.index_to_label.items():
            if int(candidate_label) == int(raw_label):
                return int(index)
        return None

    def _select_prediction_index_with_nonzero_threshold(
        self,
        prediction_scores: np.ndarray,
    ) -> int:
        nonzero_indices = tuple(int(index) for index in self.metadata.nonzero_indices)
        threshold = self.metadata.nonzero_prediction_threshold
        if not nonzero_indices or threshold is None:
            return int(np.argmax(prediction_scores))

        for index in (self.metadata.zero_index, *nonzero_indices):
            if not 0 <= int(index) < prediction_scores.shape[0]:
                raise RuntimeError(
                    "Prediction index %d is out of range for output width %d."
                    % (int(index), prediction_scores.shape[0])
                )

        best_nonzero_index = max(
            nonzero_indices,
            key=lambda index: float(prediction_scores[int(index)]),
        )
        if float(prediction_scores[int(best_nonzero_index)]) >= float(threshold):
            return int(best_nonzero_index)
        return int(self.metadata.zero_index)

    def _select_prediction_index_with_label_thresholds(
        self,
        prediction_scores: np.ndarray,
    ) -> int:
        threshold_by_index: dict[int, float] = {}
        for raw_label, threshold in self.metadata.label_confidence_thresholds.items():
            index = self._label_to_index(raw_label)
            if index is None:
                raise RuntimeError(
                    "Threshold configured for raw label %r, but it is not present in "
                    "labels.index_to_label." % raw_label
                )
            threshold_by_index[index] = float(threshold)

        for index in np.argsort(prediction_scores)[::-1]:
            score = float(prediction_scores[int(index)])
            threshold = threshold_by_index.get(int(index))
            if threshold is None or score >= threshold:
                return int(index)

        if self.metadata.fallback_label is not None:
            fallback_index = self._label_to_index(self.metadata.fallback_label)
            if fallback_index is None:
                raise RuntimeError(
                    "Fallback raw label %r is not present in labels.index_to_label."
                    % self.metadata.fallback_label
                )
            return fallback_index

        return int(np.argmax(prediction_scores))

    def push_result(self, features: Any, sample_time_s: float) -> PredictionResult | None:
        """Append one feature vector and return a throttled inference result when ready."""
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self.metadata.input_num_features,):
            raise ValueError(
                "Expected %d features, got shape %s."
                % (self.metadata.input_num_features, vector.shape)
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
        window = self._prepare_window_for_model(window)
        if self.metadata.normalize:
            window = (window - self.metadata.x_mean) / self.metadata.x_std

        model_input = np.transpose(window, (1, 0))[None, :, :]
        logits = self.session.run([self.output_name], {self.input_name: model_input})[0]
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[0]
        logits = logits.reshape(-1)
        prediction_scores = self._postprocess_scores(logits)
        pred_index = self._select_prediction_index(logits)
        self.last_inference_time_s = sample_time_s
        pred_label = self.metadata.index_to_label.get(pred_index, pred_index)
        return PredictionResult(
            logits=logits.copy(),
            prediction_scores=prediction_scores.copy(),
            pred_index=int(pred_index),
            pred_label=int(pred_label),
        )

    def push(self, features: Any, sample_time_s: float) -> int | None:
        """Append one feature vector and return a throttled predicted label when ready."""
        result = self.push_result(features, sample_time_s)
        if result is None:
            return None
        return int(result.pred_label)
