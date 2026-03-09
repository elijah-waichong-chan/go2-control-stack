"""Shared runtime utilities for direction intent ONNX models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool

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


def resolve_model_path(deploy_path: Path, model_cfg: dict[str, Any]) -> Path:
    """Resolve the ONNX path relative to the deploy YAML when needed."""
    raw_path = model_cfg.get("path")
    if not raw_path:
        raise RuntimeError("deploy.yaml model.path is required.")
    model_path = Path(str(raw_path))
    if not model_path.is_absolute():
        model_path = (deploy_path.parent / model_path).resolve()
    if not model_path.exists():
        raise RuntimeError(f"ONNX model not found: {model_path}")
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
    deploy_path = model_dir / "params" / "deploy.yaml"
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
    """Shared ONNX-backed sliding-window classifier for intent models."""

    def __init__(
        self,
        *,
        logger: Any,
        model_dir: Path,
        onnx_intra_threads: int = 1,
        onnx_inter_threads: int = 1,
    ) -> None:
        require_runtime_dependencies()
        self.logger = logger
        self.metadata = load_model_metadata(model_dir)
        self.history: deque[Any] = deque(maxlen=self.metadata.num_timesteps)
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

    def push(self, features: Any) -> int | None:
        """Append one feature vector and return a predicted label when ready."""
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self.metadata.num_features,):
            raise ValueError(
                "Expected %d features, got shape %s."
                % (self.metadata.num_features, vector.shape)
            )

        self.history.append(vector)
        if len(self.history) < self.metadata.num_timesteps:
            return None

        window = np.asarray(self.history, dtype=np.float32)
        if self.metadata.normalize:
            window = (window - self.metadata.x_mean) / self.metadata.x_std

        model_input = np.transpose(window, (1, 0))[None, :, :]
        logits = self.session.run([self.output_name], {self.input_name: model_input})[0]
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[0]
        logits = logits.reshape(-1)
        pred_index = int(np.argmax(logits))
        return self.metadata.index_to_label.get(pred_index, pred_index)


class RunningStatusHeartbeat:
    """Publish a periodic Bool heartbeat while a node is alive."""

    def __init__(
        self,
        *,
        node: Any,
        topic: str = "/status/intent_estimator/is_running",
        hz: float = 1.0,
    ) -> None:
        self.node = node
        self.topic = str(topic)
        self.hz = max(0.1, float(hz))
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = node.create_publisher(Bool, self.topic, qos)
        self.timer = node.create_timer(1.0 / self.hz, self.publish)
        self.publish()

    def publish(self) -> None:
        """Publish a True heartbeat sample."""
        self.publisher.publish(Bool(data=True))
