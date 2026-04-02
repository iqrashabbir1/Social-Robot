# CS1: ROS2 plus Digital Twin Validation

## Goal
Validate that the ROS2-aligned digital-twin backbone is synchronized, measurable, and robust enough to support later caregiving modules.

## Interfaces Covered
- `/camera/image_raw`
- `/audio/stream`
- `/robot_pose`
- `/head_cmd`
- `/speech_cmd`
- `/event_log`
- `/system_health`

## Experiment Modes
- `M1`: simulator only
- `M2`: simulator plus control loop
- `M3`: simulator plus recorded playback
- `M4`: simulator plus injected delay, noise, and dropout faults

## Current Outputs
- `outputs/csv/cs1/latency_metrics.csv`
- `outputs/csv/cs1/sync_error_timeseries.csv`
- `outputs/csv/cs1/fault_injection_results.csv`
- `outputs/figures/cs1/latency_distribution.png`
- `outputs/figures/cs1/synchronization_error_over_time.png`
- `outputs/figures/cs1/task_success_comparison.png`
- `outputs/figures/cs1/resource_usage.png`

## Current Run Snapshot
- M1 mean latency: `19.1193 ms`
- M2 mean latency: `28.5686 ms`
- M3 mean latency: `32.5039 ms`
- M4 mean latency: `40.4978 ms`
- M4 message drop rate: `0.0531`
- M4 recovery rate: `0.0466`

## Interpretation
These are simulation-backed system-level measurements. They support a Paper 1 claim about measurable runtime behavior, not about final robot deployment.

## TODO for Real Integration
- connect the interface specification to a full ROS2 simulator
- replace software-equivalent topic generation with live ROS2 publishers and subscribers
- repeat the same metrics with simulator and hardware traces
