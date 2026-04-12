# Pilot Data Folder

This directory stores the small Paper 1 real-anchor dataset.

## Purpose
- provide a minimal real local anchor for CS2 synchronization
- provide a pilot baseline inference path for CS3
- ground the paper beyond fully synthetic data without overclaiming dataset scale

## Structure
- `sessions/<session_id>/frames/`
- `sessions/<session_id>/audio.wav`
- `sessions/<session_id>/video_frames.csv`
- `sessions/<session_id>/audio_chunks.csv`
- `sessions/<session_id>/robot_state_log.csv`
- `sessions/<session_id>/session_metadata.csv`
- `sessions/<session_id>/session_manifest.json`

## Claim boundary
These pilot sessions are only small local demonstrations. They are not a clinically representative dataset and should not be used to claim real-world caregiving effectiveness.
