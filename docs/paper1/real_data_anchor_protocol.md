# Real Data Anchor Protocol

## Goal
Add a minimal real local anchor without overstating the dataset scale.

## Protocol
1. Record a short webcam and microphone session on the target machine.
2. Log a placeholder robot-state/context stream with synchronized timestamps.
3. Export frame, audio-chunk, and context manifests.
4. Build a session manifest and session metadata file.
5. Use the session in:
   - CS2 synchronization analysis
   - CS3 pilot baseline inference

## Current local outcome
- a short session `paper1_anchor_demo` was collected locally
- it is sufficient for pilot demonstration only
- it is not large enough for generalizable model evaluation
