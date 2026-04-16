# Status Codes

This file is the central reference for status topics in this repo.

## Status Message Type

All current `/status/...` topics use `go2_msgs/LoopStatus`.

Fields:

- `status`: module-specific integer status code
- `avg_loop_ms`: average loop time in milliseconds
- `p99_loop_ms`: p99 loop time in milliseconds
- `max_loop_ms`: maximum loop time in milliseconds
- `budget_ms`: loop budget in milliseconds
- `deadline_miss_count`: number of iterations over budget
- `sample_count`: number of samples included in the timing stats

Current convention:

- `-1` in timing-related fields means "not recorded / not applicable"

## Published Status Topics

### `/status/loco_ctrl`

Message type: `go2_msgs/LoopStatus`

Source: `locomotion_controller/policy_controller.py`

- `0`: idle
- `1`: running
- `2`: waiting for fresh `/lowstate`
- `3`: waiting for `/status/standing_init` readiness

### `/status/standing_init`

Message type: `go2_msgs/LoopStatus`

Source: `locomotion_controller/standup_init.py`

- `1`: running stand-up sequence
- `2`: waiting for `/lowstate`
- `3`: stand-up complete

### `/status/state_estimator`

Message type: `go2_msgs/LoopStatus`

Source: `go2_odometry/scripts/inekf_odom.py`

- `1`: running
- `2`: waiting for `/status/standing_init` readiness

### `/status/arm_parser`

Message type: `go2_msgs/LoopStatus`

Source: `arm_controller/arm_feedback_parser.py`

- `1`: publishing parsed arm angles
- `2`: waiting for arm feedback input

### `/status/intent_estimator/left_right`

Message type: `go2_msgs/LoopStatus`

Source: `intent_estimator/left_right_intent_estimator.py`

- `1`: running
- `2`: waiting for required input topics

### `/status/intent_estimator/up_down`

Message type: `go2_msgs/LoopStatus`

Source: `intent_estimator/up_down_intent_estimator.py`

- `1`: running
- `2`: waiting for required input topics

## Dashboard-Derived Aggregate Status

The telemetry dashboard derives the "Intent Estimator" card from the freshest recent status among:

- `/status/intent_estimator/left_right`
- `/status/intent_estimator/up_down`

This aggregate view is not itself published on a ROS topic.

Source: `telemetry_dashboard/app.py`

- `1`: running
- `2`: waiting for required input topics
- other values: surfaced directly from the freshest estimator status

## Notes

- Status codes are module-specific. The same integer does not automatically mean the same thing across different topics.
- If a module starts recording loop timing later, update both its publisher and this document.
