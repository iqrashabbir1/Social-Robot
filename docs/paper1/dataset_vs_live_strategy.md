# Dataset vs Live Strategy

## Why both are needed
- offline dataset evaluation provides controlled and repeatable perception evidence
- dataset replay validates that the same ROS2 graph can process controlled inputs end-to-end
- live hybrid runtime evidence demonstrates real transport, node orchestration, and logging behavior

## Paper 1 interpretation
- dataset evaluation should be the main controlled perception evidence
- ROS dataset replay should be the main bridge between offline scoring and ROS runtime validation
- hybrid live runtime should remain the main integration/runtime evidence
- `RAF-DB` is the recommended main controlled image dataset for this repo
- `CREMA-D` is the recommended replay-through-ROS dataset for this repo

## Claim boundary
- dataset evaluation does not prove live deployment readiness
- live runtime evidence does not replace controlled perception benchmarking
- Paper 1 is strongest when both are presented together
