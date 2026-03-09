Model bundle layout for `direction_intent_estimator`.

- `012`: forward/backward direction intent estimator
- `034`: left/right direction intent estimator
- `056`: reserved for the future third model

Each model bundle is organized as:

- `params/deploy.yaml`: model metadata and preprocessing config
- `exported/policy.onnx`: ONNX graph
- `exported/policy.onnx.data`: external ONNX tensor data when present

This mirrors the existing `locomotion_controller` packaging pattern so each model is a self-contained, relocatable asset bundle.
