Model bundle layout for `intent_estimator`.

- `034`: left/right direction intent estimator
- `056`: up/down direction intent estimator

The current model bundles in this package use a flat per-bundle layout:

- `<model_name>.deploy.yaml`: model metadata and preprocessing config
- `<model_name>.onnx`: ONNX graph
- `<model_name>.onnx.data`: external ONNX tensor data when present

Example for `034`:

- `034/intent_left_right.deploy.yaml`
- `034/intent_left_right.onnx`
- `034/intent_left_right.onnx.data`

The deploy YAML declares the tensor names, feature layout, label mapping, and preprocessing stats used by the runtime.
